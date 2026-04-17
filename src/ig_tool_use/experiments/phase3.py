"""Phase 3: Threshold-based automatic deferral (Algorithm 1).

For each ε ∈ {0, 0.01, 0.05, 0.1, 0.5}, run Algorithm 1 on the 500 eval samples:
  * The agent attempts each step with the model.
  * If IG(t) ≤ ε: re-execute using the Calculator tool.
  * Record final accuracy, per-step deferral rates, and IG profiles.

Artifacts saved under  {output_dir}/phase3/:
  results_eps_{ε}.pkl        – list[SampleResult] for each ε
  summary_phase3.json        – accuracy, deferral rates, IG for all ε values
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

from tqdm import tqdm

from ig_tool_use.agent.ig_agent import IGAgent, SampleResult
from ig_tool_use.config import ExperimentConfig
from ig_tool_use.rollout.vllm_rollout import VLLMRollout
from ig_tool_use.supervisor.train import load_supervisor
from ig_tool_use.tools.calculator import Calculator

log = logging.getLogger(__name__)


def run(cfg: ExperimentConfig) -> dict:
    """Run Phase 3 for all epsilon values. Returns full summary dict."""
    phase1_dir = Path(cfg.output_dir) / "phase1"
    out_dir = Path(cfg.output_dir) / "phase3"
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
    # vllm rollout (needed for sequential step generation)
    # ------------------------------------------------------------------
    rollout = VLLMRollout(cfg.rollout)
    calculator = Calculator()
    agent = IGAgent(rollout=rollout, supervisor=supervisor, calculator=calculator)

    # ------------------------------------------------------------------
    # Sweep over ε
    # ------------------------------------------------------------------
    all_summaries: dict[float, dict] = {}

    for epsilon in cfg.epsilon_values:
        eps_str = str(epsilon).replace(".", "_")
        result_path = out_dir / f"results_eps_{eps_str}.pkl"

        if result_path.exists():
            log.info("ε=%.3f – loading cached results.", epsilon)
            with open(result_path, "rb") as f:
                results: list[SampleResult] = pickle.load(f)
        else:
            log.info("ε=%.3f – running Algorithm 1 on %d samples …", epsilon, len(eval_samples))
            results = []
            for s in tqdm(eval_samples, desc=f"Phase3 ε={epsilon}"):
                results.append(agent.run_threshold(s, epsilon))

            with open(result_path, "wb") as f:
                pickle.dump(results, f)

        summary = _summarise(results, epsilon)
        all_summaries[epsilon] = summary
        log.info(
            "ε=%.3f  acc=%.1f%%  defer3=%.1f%%  precision3=%.3f  recall3=%.3f",
            epsilon,
            summary["accuracy"] * 100,
            summary["deferral_rate_3"] * 100,
            summary["precision_step3"],
            summary["recall_step3"],
        )

    # Save combined summary.
    summary_path = out_dir / "summary_phase3.json"
    with open(summary_path, "w") as f:
        # Convert float keys to strings for JSON.
        json.dump({str(k): v for k, v in all_summaries.items()}, f, indent=2)
    log.info("Phase 3 summary saved to %s", summary_path)

    return all_summaries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarise(results: list[SampleResult], epsilon: float) -> dict:
    n = len(results)
    accuracy = sum(r.correct for r in results) / n

    ig_sums = {1: 0.0, 2: 0.0, 3: 0.0}
    defer_counts = {1: 0, 2: 0, 3: 0}

    # Precision/recall for step 3 specifically.
    tp3 = fp3 = fn3 = tn3 = 0

    for r in results:
        for sr in r.steps:
            ig_sums[sr.step] += sr.ig
            if sr.deferred:
                defer_counts[sr.step] += 1
            if sr.step == 3:
                step_wrong = sr.model_val != sr.gt_val
                if sr.deferred and step_wrong:
                    tp3 += 1
                elif sr.deferred and not step_wrong:
                    fp3 += 1
                elif not sr.deferred and step_wrong:
                    fn3 += 1
                else:
                    tn3 += 1

    precision3 = tp3 / (tp3 + fp3) if (tp3 + fp3) > 0 else float("nan")
    recall3 = tp3 / (tp3 + fn3) if (tp3 + fn3) > 0 else float("nan")

    return {
        "epsilon": epsilon,
        "accuracy": accuracy,
        "mean_ig_1": ig_sums[1] / n,
        "mean_ig_2": ig_sums[2] / n,
        "mean_ig_3": ig_sums[3] / n,
        "deferral_rate_1": defer_counts[1] / n,
        "deferral_rate_2": defer_counts[2] / n,
        "deferral_rate_3": defer_counts[3] / n,
        "precision_step3": precision3,
        "recall_step3": recall3,
        "tp3": tp3, "fp3": fp3, "fn3": fn3, "tn3": tn3,
    }
