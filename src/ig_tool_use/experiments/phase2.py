"""Phase 2: Oracle tool deferral.

For the 500 eval samples from Phase 1, replace step 3 with the Python
calculator's exact output and re-estimate IG(3) using the same supervisor.

Prediction (Section 7.3 of the report):
  * Î_π*(3) should be significantly positive.
  * Final accuracy should rise from ~42% to ~80% (ceiling limited by step-1 errors).

Artifacts saved under  {output_dir}/phase2/:
  results_oracle.pkl   – list[SampleResult]
  summary_oracle.json  – accuracy + mean IG per step
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

from ig_tool_use.agent.ig_agent import IGAgent, SampleResult
from ig_tool_use.config import ExperimentConfig
from ig_tool_use.supervisor.train import load_supervisor
from ig_tool_use.tools.calculator import Calculator

log = logging.getLogger(__name__)


def run(cfg: ExperimentConfig) -> dict:
    """Run Phase 2 and return accuracy + per-step IG summary."""
    phase1_dir = Path(cfg.output_dir) / "phase1"
    out_dir = Path(cfg.output_dir) / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load Phase 1 artifacts
    # ------------------------------------------------------------------
    eval_pkl = phase1_dir / "samples_eval.pkl"
    if not eval_pkl.exists():
        raise FileNotFoundError(
            f"Phase 1 eval data not found at {eval_pkl}. Run Phase 1 first."
        )
    with open(eval_pkl, "rb") as f:
        eval_samples = pickle.load(f)

    supervisor_dir = Path(cfg.checkpoint_dir) / "final"
    if not supervisor_dir.exists():
        raise FileNotFoundError(
            f"Supervisor checkpoint not found at {supervisor_dir}. Run Phase 1 first."
        )
    supervisor = load_supervisor(supervisor_dir, device=cfg.device)

    # ------------------------------------------------------------------
    # Oracle deferral
    # ------------------------------------------------------------------
    calculator = Calculator()
    # For oracle we don't need vllm (steps 1+2 come from Phase 1 rollout).
    agent = IGAgent(rollout=None, supervisor=supervisor, calculator=calculator)

    log.info("Running oracle deferral on %d eval samples …", len(eval_samples))
    results: list[SampleResult] = agent.run_oracle_batch(eval_samples)

    # ------------------------------------------------------------------
    # Summarise
    # ------------------------------------------------------------------
    summary = _summarise(results)
    log.info(
        "Oracle deferral: accuracy=%.1f%%  IG(1)=%.4f  IG(2)=%.4f  IG(3)=%.4f",
        summary["accuracy"] * 100,
        summary["mean_ig_1"], summary["mean_ig_2"], summary["mean_ig_3"],
    )

    results_path = out_dir / "results_oracle.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    summary_path = out_dir / "summary_oracle.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Phase 2 summary saved to %s", summary_path)

    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarise(results: list[SampleResult]) -> dict:
    n = len(results)
    accuracy = sum(r.correct for r in results) / n
    ig_sums = {1: 0.0, 2: 0.0, 3: 0.0}
    defer_counts = {1: 0, 2: 0, 3: 0}

    for r in results:
        for sr in r.steps:
            ig_sums[sr.step] += sr.ig
            if sr.deferred:
                defer_counts[sr.step] += 1

    return {
        "accuracy": accuracy,
        "mean_ig_1": ig_sums[1] / n,
        "mean_ig_2": ig_sums[2] / n,
        "mean_ig_3": ig_sums[3] / n,
        "deferral_rate_1": defer_counts[1] / n,
        "deferral_rate_2": defer_counts[2] / n,
        "deferral_rate_3": defer_counts[3] / n,
    }
