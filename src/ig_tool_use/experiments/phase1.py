"""Phase 1: Baseline information-gain profiling.

Steps:
  1. Generate N_train=2000 CoT traces from Llama-3-8B (no tool access).
  2. Train the GPT-2 supervisor on those traces.
  3. Evaluate per-step IG on N_eval=500 held-out samples.

Expected outcome (Table 2 of [1]):
  IG(step 1) ≈ 0.67  (positive, step 1 is mostly correct at 80%)
  IG(step 2) ≈ 0.24  (positive, step 2 is mostly correct at 98%)
  IG(step 3) ≈ 0.03  (near-zero, step 3 fails 58% of the time)

Artifacts saved under  {output_dir}/phase1/:
  samples_train.pkl    – ArithmeticSamples with rollout data
  samples_eval.pkl     – eval split
  ig_baseline.json     – mean IG per step on eval set
  supervisor/          – trained checkpoint
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

from ig_tool_use.config import ExperimentConfig
from ig_tool_use.data.arithmetic import generate_samples
from ig_tool_use.rollout.vllm_rollout import VLLMRollout
from ig_tool_use.supervisor.dataset import build_training_texts
from ig_tool_use.supervisor.train import (
    SupervisorModel,
    compute_dataset_ig,
    train_supervisor,
)

log = logging.getLogger(__name__)


def run(cfg: ExperimentConfig) -> dict[int, float]:
    """Run Phase 1 and return mean IG per step on the eval set."""
    out_dir = Path(cfg.output_dir) / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Generate CoT traces
    # ------------------------------------------------------------------
    train_pkl = out_dir / "samples_train.pkl"
    eval_pkl = out_dir / "samples_eval.pkl"

    if train_pkl.exists() and eval_pkl.exists():
        log.info("Loading cached rollout data from %s", out_dir)
        with open(train_pkl, "rb") as f:
            train_samples = pickle.load(f)
        with open(eval_pkl, "rb") as f:
            eval_samples = pickle.load(f)
    else:
        log.info(
            "Generating %d train + %d eval CoT traces …",
            cfg.rollout.n_train,
            cfg.rollout.n_eval,
        )
        total_n = cfg.rollout.n_train + cfg.rollout.n_eval
        all_samples = generate_samples(total_n, seed=cfg.rollout.seed)

        rollout = VLLMRollout(cfg.rollout)
        all_samples = rollout.run_full_dataset(all_samples, desc="Phase1-rollout")

        train_samples = all_samples[: cfg.rollout.n_train]
        eval_samples = all_samples[cfg.rollout.n_train :]

        with open(train_pkl, "wb") as f:
            pickle.dump(train_samples, f)
        with open(eval_pkl, "wb") as f:
            pickle.dump(eval_samples, f)

    _log_accuracy(train_samples, "train")
    _log_accuracy(eval_samples, "eval")

    # ------------------------------------------------------------------
    # 2. Train supervisor
    # ------------------------------------------------------------------
    supervisor_dir = Path(cfg.checkpoint_dir)
    supervisor_final = supervisor_dir / "final"

    if supervisor_final.exists():
        log.info("Supervisor checkpoint found at %s, skipping training.", supervisor_final)
        from ig_tool_use.supervisor.train import load_supervisor
        supervisor = load_supervisor(supervisor_final, device=cfg.device)
    else:
        log.info("Training supervisor …")
        train_texts = build_training_texts(train_samples)
        log.info("Total supervisor training strings: %d", len(train_texts))
        supervisor = train_supervisor(
            train_texts, cfg.supervisor, save_dir=supervisor_dir
        )
        supervisor.to(cfg.device)

    # ------------------------------------------------------------------
    # 3. Estimate per-step IG on eval set
    # ------------------------------------------------------------------
    log.info("Estimating information-gain on eval set …")
    ig_per_step = compute_dataset_ig(supervisor, eval_samples)
    log.info("Mean IG  step1=%.4f  step2=%.4f  step3=%.4f",
             ig_per_step[1], ig_per_step[2], ig_per_step[3])

    ig_path = out_dir / "ig_baseline.json"
    with open(ig_path, "w") as f:
        json.dump({str(k): v for k, v in ig_per_step.items()}, f, indent=2)
    log.info("Saved IG profile to %s", ig_path)

    # Also compute and save per-sample IG for Phase 4 analysis.
    _save_per_sample_ig(supervisor, eval_samples, out_dir / "ig_per_sample.pkl")

    return ig_per_step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_accuracy(samples, split: str) -> None:
    n = len(samples)
    acc1 = sum(s.step1_correct for s in samples) / n
    acc2 = sum(s.step2_correct for s in samples) / n
    acc3 = sum(s.step3_correct for s in samples) / n
    log.info(
        "[%s] Step accuracy: step1=%.1f%%  step2=%.1f%%  step3=%.1f%%",
        split, acc1 * 100, acc2 * 100, acc3 * 100,
    )


def _save_per_sample_ig(supervisor, eval_samples, path: Path) -> None:
    """Save per-sample IG values for all three steps."""
    from ig_tool_use.supervisor.train import compute_ig

    records = []
    for s in eval_samples:
        Y = s.supervisor_target
        states = [s.sup_state_0, s.sup_state_1, s.sup_state_2, s.sup_state_3]
        rec = {"x": s.x, "y": s.y, "step1_correct": s.step1_correct,
               "step2_correct": s.step2_correct, "step3_correct": s.step3_correct}
        for step in (1, 2, 3):
            if states[step - 1] and states[step]:
                ig = compute_ig(supervisor, states[step - 1], states[step], Y)
            else:
                ig = float("nan")
            rec[f"ig_{step}"] = ig
        records.append(rec)

    with open(path, "wb") as f:
        pickle.dump(records, f)
    log.info("Saved per-sample IG to %s", path)
