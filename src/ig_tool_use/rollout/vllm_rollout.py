"""vllm-based CoT generation for the base model.

Two generation modes:

  generate_full  – Generates all three steps at once. Used in Phase 1 to build
                   the training corpus for the supervisor.

  generate_step  – Generates a single step from an accumulated rollout prompt.
                   Used in Phase 3 (Algorithm 1) where tool deferral may change
                   the state between steps.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from tqdm import tqdm

from ig_tool_use.config import RolloutConfig
from ig_tool_use.data.arithmetic import (
    ArithmeticSample,
    make_rollout_prompt,
    make_step_text,
    parse_model_output,
)

if TYPE_CHECKING:
    pass

# Stop sequences per step:  prevent the model from running into the next step.
_STOP_SEQUENCES: dict[int, list[str]] = {
    1: ["\n2.", "\n\n"],
    2: ["\n3.", "\n\n"],
    3: ["\n\n", "\n\n\n"],
}


class VLLMRollout:
    """Wraps vllm.LLM for CoT generation on arithmetic prompts."""

    def __init__(self, cfg: RolloutConfig) -> None:
        # Lazy import so the module is importable without vllm installed.
        from vllm import LLM, SamplingParams  # noqa: F401

        self._LLM = LLM
        self._SamplingParams = SamplingParams
        self.cfg = cfg
        self._llm: object | None = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = self._LLM(
                model=self.cfg.model_name,
                tensor_parallel_size=self.cfg.tensor_parallel_size,
                gpu_memory_utilization=self.cfg.gpu_memory_utilization,
                trust_remote_code=True,
            )
        return self._llm

    # ------------------------------------------------------------------
    # Full CoT generation (Phase 1 / Phase 2 baseline)
    # ------------------------------------------------------------------

    def generate_full(self, samples: list[ArithmeticSample]) -> list[ArithmeticSample]:
        """Generate complete 3-step CoT for each sample in one vllm call."""
        llm = self._get_llm()
        sampling_params = self._SamplingParams(
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_new_tokens_full,
            stop=["\n\n\n"],
        )

        prompts = [make_rollout_prompt(s.x, s.y) for s in samples]
        outputs = llm.generate(prompts, sampling_params)

        for sample, output in zip(samples, outputs):
            raw = output.outputs[0].text
            parse_model_output(sample, raw)

        return samples

    # ------------------------------------------------------------------
    # Single-step generation (Phase 3 / Algorithm 1)
    # ------------------------------------------------------------------

    def generate_step(self, rollout_prompt: str, step: int) -> str:
        """Generate the response for a single step given the accumulated rollout prompt.

        The rollout_prompt ends just before the current step (e.g. ends with
        "Answer:\\n1. 3x = 69\\n" when about to generate step 2).
        Returns the raw text for this step only.
        """
        llm = self._get_llm()
        sampling_params = self._SamplingParams(
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_new_tokens_per_step,
            stop=_STOP_SEQUENCES.get(step, ["\n\n"]),
        )
        # Add the step prefix to guide the model.
        step_prefix = f"{step}. " if step == 1 else ""
        prompt = rollout_prompt + step_prefix
        outputs = llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

    def generate_step_batch(
        self, rollout_prompts: list[str], step: int
    ) -> list[str]:
        """Batch version of generate_step for throughput."""
        llm = self._get_llm()
        sampling_params = self._SamplingParams(
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_new_tokens_per_step,
            stop=_STOP_SEQUENCES.get(step, ["\n\n"]),
        )
        prompts = [p + (f"{step}. " if step == 1 else "") for p in rollout_prompts]
        outputs = llm.generate(prompts, sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def run_full_dataset(
        self, samples: list[ArithmeticSample], batch_size: int = 64, desc: str = "Rollout"
    ) -> list[ArithmeticSample]:
        """Run full CoT generation over a dataset in batches."""
        results: list[ArithmeticSample] = []
        for i in tqdm(range(0, len(samples), batch_size), desc=desc):
            batch = samples[i : i + batch_size]
            results.extend(self.generate_full(batch))
        return results
