"""Python calculator tool τ.

The tool is correct by construction: it computes 3x, 2y, 3x+2y exactly.
It corresponds to Λ_τ in the paper—a compositionally consistent update rule
that always executes arithmetic correctly, regardless of number magnitude.

This models the code-interpreter tool from Section 7.1 of the project report.
"""
from __future__ import annotations

from typing import Optional


class Calculator:
    """External arithmetic tool that correctly executes all three sub-tasks."""

    def execute_step(
        self,
        step: int,
        x: int,
        y: int,
        step1_result: Optional[int] = None,
        step2_result: Optional[int] = None,
    ) -> int:
        """Run the calculator for a given step.

        Args:
            step:          1, 2, or 3.
            x, y:          The original prompt values.
            step1_result:  Output of step 1 (needed for step 3).
            step2_result:  Output of step 2 (needed for step 3).

        Returns:
            Correct integer result for the step.
        """
        if step == 1:
            return self._step1(x)
        elif step == 2:
            return self._step2(y)
        elif step == 3:
            # Use tool-computed values for steps 1 and 2 if the caller did not
            # provide them (e.g. those steps were also deferred to the tool).
            v1 = step1_result if step1_result is not None else self._step1(x)
            v2 = step2_result if step2_result is not None else self._step2(y)
            return self._step3(v1, v2)
        else:
            raise ValueError(f"Unknown step: {step}")

    # ------------------------------------------------------------------
    # Ground-truth primitive tasks (Λ_τ)
    # ------------------------------------------------------------------

    @staticmethod
    def _step1(x: int) -> int:
        return 3 * x

    @staticmethod
    def _step2(y: int) -> int:
        return 2 * y

    @staticmethod
    def _step3(step1: int, step2: int) -> int:
        return step1 + step2
