"""Algorithm 1: Tool-Augmented CoT with Information-Gain Monitoring.

The agent processes each arithmetic sample step-by-step:
  1. Attempt the step with the base model (via vllm).
  2. Estimate IG using the supervisor.
  3. If IG > ε: accept the model's output.
     Else: defer to the Calculator tool.

Three evaluation modes are provided:
  * no_tool     – baseline: always use the model, never defer.
  * oracle      – always defer step 3 to the tool (oracle deferral, Phase 2).
  * threshold   – defer any step where estimated IG ≤ ε (Algorithm 1, Phase 3).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ig_tool_use.config import COT_SEP
from ig_tool_use.data.arithmetic import (
    ArithmeticSample,
    make_rollout_prompt,
    make_step_text,
    make_supervisor_state,
)
from ig_tool_use.supervisor.train import SupervisorModel, compute_ce_loss, compute_ig
from ig_tool_use.tools.calculator import Calculator


# ---------------------------------------------------------------------------
# Per-step result
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    step: int
    model_val: Optional[int]        # Model's raw answer (may be None if parse fails)
    final_val: int                  # Value used (model or tool)
    deferred: bool                  # True if tool was used
    ig: float                       # Estimated IG at this step
    gt_val: int                     # Ground-truth value


@dataclass
class SampleResult:
    x: int
    y: int
    steps: list[StepResult] = field(default_factory=list)

    @property
    def final_answer(self) -> Optional[int]:
        return self.steps[2].final_val if len(self.steps) == 3 else None

    @property
    def correct(self) -> bool:
        return self.final_answer == 3 * self.x + 2 * self.y

    @property
    def step3_deferred(self) -> bool:
        return len(self.steps) == 3 and self.steps[2].deferred


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_STEP_PATTERNS = {
    1: re.compile(r"3x\s*=\s*([\d.]+)", re.IGNORECASE),
    2: re.compile(r"2y\s*=\s*([\d.]+)", re.IGNORECASE),
    3: re.compile(r"3x\s*\+\s*2y\s*=\s*([\d.]+)", re.IGNORECASE),
}


def _parse_step_val(text: str, step: int) -> Optional[int]:
    m = _STEP_PATTERNS[step].search(text)
    if m is None:
        return None
    try:
        return int(float(m.group(1)))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------

class IGAgent:
    """Implements the three evaluation modes from Section 7.2 of the report."""

    def __init__(
        self,
        rollout,            # VLLMRollout instance
        supervisor: SupervisorModel,
        calculator: Calculator,
    ) -> None:
        self.rollout = rollout
        self.supervisor = supervisor
        self.calculator = calculator

    # ------------------------------------------------------------------
    # Mode 1: No tool (baseline)
    # ------------------------------------------------------------------

    def run_no_tool(self, sample: ArithmeticSample) -> SampleResult:
        """Use the model for all steps, never defer.  Reads from pre-parsed sample."""
        result = SampleResult(x=sample.x, y=sample.y)
        gt = [sample.step1_gt, sample.step2_gt, sample.step3_gt]
        model_vals = [sample.step1_model, sample.step2_model, sample.step3_model]
        states = [sample.sup_state_0, sample.sup_state_1, sample.sup_state_2, sample.sup_state_3]
        Y = sample.supervisor_target

        for step in (1, 2, 3):
            val = model_vals[step - 1]
            ig = compute_ig(self.supervisor, states[step - 1], states[step], Y)
            result.steps.append(
                StepResult(
                    step=step,
                    model_val=val,
                    final_val=val if val is not None else 0,
                    deferred=False,
                    ig=ig,
                    gt_val=gt[step - 1],
                )
            )
        return result

    # ------------------------------------------------------------------
    # Mode 2: Oracle deferral (Phase 2)
    # ------------------------------------------------------------------

    def run_oracle(self, sample: ArithmeticSample) -> SampleResult:
        """Always defer step 3 to the calculator; use model for steps 1 and 2.

        The IG is re-estimated with the tool's correct step-3 output so we can
        compare with the baseline.
        """
        result = SampleResult(x=sample.x, y=sample.y)
        gt = [sample.step1_gt, sample.step2_gt, sample.step3_gt]
        model_vals = [sample.step1_model, sample.step2_model, sample.step3_model]
        Y = sample.supervisor_target

        # Accumulated model supervisor state.
        step_vals: list[Optional[int]] = [None, None, None]

        for step in (1, 2):
            step_vals[step - 1] = model_vals[step - 1]

        # Steps 1 and 2: use model outputs.
        state_0 = sample.sup_state_0
        state_1 = sample.sup_state_1
        state_2 = sample.sup_state_2

        for step in (1, 2):
            states = [state_0, state_1, state_2]
            val = model_vals[step - 1]
            ig = compute_ig(self.supervisor, states[step - 1], states[step], Y)
            result.steps.append(
                StepResult(
                    step=step,
                    model_val=val,
                    final_val=val if val is not None else 0,
                    deferred=False,
                    ig=ig,
                    gt_val=gt[step - 1],
                )
            )

        # Step 3: always defer to tool.
        tool_val3 = self.calculator.execute_step(3, sample.x, sample.y,
                                                  step1_result=step_vals[0],
                                                  step2_result=step_vals[1])
        # Build tool-corrected state_3 for IG estimation.
        state_3_tool = make_supervisor_state(
            sample.x, sample.y,
            step1_val=step_vals[0],
            step2_val=step_vals[1],
            step3_val=tool_val3,
        )
        ig3 = compute_ig(self.supervisor, state_2, state_3_tool, Y)
        result.steps.append(
            StepResult(
                step=3,
                model_val=sample.step3_model,
                final_val=tool_val3,
                deferred=True,
                ig=ig3,
                gt_val=gt[2],
            )
        )
        return result

    # ------------------------------------------------------------------
    # Mode 3: Threshold-based deferral / Algorithm 1 (Phase 3)
    # ------------------------------------------------------------------

    def run_threshold(self, sample: ArithmeticSample, epsilon: float) -> SampleResult:
        """Algorithm 1: defer step t to tool if IG_M(t) ≤ ε.

        Generation is sequential: each step's vllm call uses the (potentially
        tool-corrected) accumulated state from previous steps.
        """
        result = SampleResult(x=sample.x, y=sample.y)
        Y = sample.supervisor_target

        # Accumulated state in two representations:
        #   rollout_prompt  – for feeding to vllm
        #   sup_states      – for supervisor CE computation
        rollout_prompt = make_rollout_prompt(sample.x, sample.y)
        current_sup_state = make_supervisor_state(sample.x, sample.y)

        step_vals: list[int] = []

        for step in (1, 2, 3):
            # --- Attempt: model generates the current step. ---
            raw = self.rollout.generate_step(rollout_prompt, step)
            model_val = _parse_step_val(raw, step)

            # Build the supervisor state with the model's output for the current step,
            # keeping previously finalized values for earlier steps.
            if model_val is not None:
                kw: dict = {}
                # Carry forward all already-finalized step values.
                if len(step_vals) >= 1:
                    kw["step1_val"] = step_vals[0]
                if len(step_vals) >= 2:
                    kw["step2_val"] = step_vals[1]
                # Overlay the model's output for the step being estimated.
                if step == 1:
                    kw["step1_val"] = model_val
                elif step == 2:
                    kw["step2_val"] = model_val
                else:
                    kw["step3_val"] = model_val
                sup_state_model = make_supervisor_state(sample.x, sample.y, **kw)
            else:
                sup_state_model = current_sup_state  # parse failed → IG ≈ 0

            # --- Estimate IG under model execution ---
            ig = compute_ig(self.supervisor, current_sup_state, sup_state_model, Y)

            # --- Decision: accept model or defer to tool ---
            if ig > epsilon and model_val is not None:
                # Accept model output.
                final_val = model_val
                deferred = False
                current_sup_state = sup_state_model
                rollout_prompt = rollout_prompt + make_step_text(step, final_val)
            else:
                # Defer to tool.
                s1_so_far = step_vals[0] if len(step_vals) > 0 else None
                s2_so_far = step_vals[1] if len(step_vals) > 1 else None
                final_val = self.calculator.execute_step(
                    step, sample.x, sample.y,
                    step1_result=s1_so_far,
                    step2_result=s2_so_far,
                )
                deferred = True
                # Update supervisor state with CORRECT tool value.
                _vals = {}
                if step >= 1:
                    _vals["step1_val"] = step_vals[0] if step > 1 else final_val
                if step >= 2:
                    _vals["step2_val"] = step_vals[1] if step > 2 else final_val
                if step == 3:
                    _vals["step3_val"] = final_val
                current_sup_state = make_supervisor_state(sample.x, sample.y, **_vals)
                rollout_prompt = rollout_prompt + make_step_text(step, final_val)

            step_vals.append(final_val)
            result.steps.append(
                StepResult(
                    step=step,
                    model_val=model_val,
                    final_val=final_val,
                    deferred=deferred,
                    ig=ig,
                    gt_val=[sample.step1_gt, sample.step2_gt, sample.step3_gt][step - 1],
                )
            )

        return result

    # ------------------------------------------------------------------
    # Batch wrappers
    # ------------------------------------------------------------------

    def run_no_tool_batch(self, samples: list[ArithmeticSample]) -> list[SampleResult]:
        return [self.run_no_tool(s) for s in samples]

    def run_oracle_batch(self, samples: list[ArithmeticSample]) -> list[SampleResult]:
        return [self.run_oracle(s) for s in samples]

    def run_threshold_batch(
        self, samples: list[ArithmeticSample], epsilon: float
    ) -> list[SampleResult]:
        return [self.run_threshold(s, epsilon) for s in samples]
