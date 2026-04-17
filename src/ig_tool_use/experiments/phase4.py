"""Phase 4: Comparison and analysis — generates all result plots.

Figures produced (saved under {output_dir}/phase4/):

  fig1_ig_baseline.png     – Per-step IG profile under no-tool baseline (bar chart)
  fig2_ig_oracle.png       – Per-step IG profile under oracle deferral
  fig3_accuracy_vs_eps.png – Final accuracy vs. ε curve
  fig4_deferral_vs_eps.png – Per-step deferral rate vs. ε
  fig5_precision_recall.png – Precision and recall of deferral criterion vs. ε
  fig6_ig_heatmap.png      – Sample-wise IG heatmap (baseline, analogous to Fig 3a of [1])
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ig_tool_use.config import ExperimentConfig

log = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", font_scale=1.2)

STEP_LABELS = ["Step 1\n3x ✓", "Step 2\n2y ✓", "Step 3\n3x+2y ✗"]


def run(cfg: ExperimentConfig) -> None:
    """Load Phase 1–3 artifacts and generate all analysis plots."""
    out_dir = Path(cfg.output_dir) / "phase4"
    out_dir.mkdir(parents=True, exist_ok=True)

    phase1_dir = Path(cfg.output_dir) / "phase1"
    phase2_dir = Path(cfg.output_dir) / "phase2"
    phase3_dir = Path(cfg.output_dir) / "phase3"

    # ------------------------------------------------------------------
    # Figure 1: Baseline IG profile
    # ------------------------------------------------------------------
    ig_baseline_path = phase1_dir / "ig_baseline.json"
    if ig_baseline_path.exists():
        with open(ig_baseline_path) as f:
            ig_baseline = {int(k): v for k, v in json.load(f).items()}
        _plot_ig_bar(ig_baseline, title="Baseline IG (no tool)",
                     path=out_dir / "fig1_ig_baseline.png")
    else:
        log.warning("Phase 1 IG data not found; skipping Figure 1.")

    # ------------------------------------------------------------------
    # Figure 2: Oracle IG profile
    # ------------------------------------------------------------------
    oracle_summary_path = phase2_dir / "summary_oracle.json"
    if oracle_summary_path.exists():
        with open(oracle_summary_path) as f:
            oracle = json.load(f)
        ig_oracle = {1: oracle["mean_ig_1"], 2: oracle["mean_ig_2"], 3: oracle["mean_ig_3"]}
        _plot_ig_bar(ig_oracle, title="Oracle Deferral IG (step 3 → tool)",
                     path=out_dir / "fig2_ig_oracle.png")
    else:
        log.warning("Phase 2 summary not found; skipping Figure 2.")

    # ------------------------------------------------------------------
    # Figures 3–5: Phase 3 sweeps
    # ------------------------------------------------------------------
    phase3_summary_path = phase3_dir / "summary_phase3.json"
    if phase3_summary_path.exists():
        with open(phase3_summary_path) as f:
            phase3 = json.load(f)

        eps_vals = sorted(float(k) for k in phase3.keys())
        records = [phase3[str(e)] for e in eps_vals]

        _plot_accuracy_vs_eps(
            eps_vals, [r["accuracy"] for r in records],
            baseline_acc=ig_baseline_path,
            oracle_acc=oracle_summary_path,
            path=out_dir / "fig3_accuracy_vs_eps.png",
        )
        _plot_deferral_vs_eps(eps_vals, records, path=out_dir / "fig4_deferral_vs_eps.png")
        _plot_precision_recall(eps_vals, records, path=out_dir / "fig5_precision_recall.png")
    else:
        log.warning("Phase 3 summary not found; skipping Figures 3–5.")

    # ------------------------------------------------------------------
    # Figure 6: Sample-wise IG heatmap (baseline)
    # ------------------------------------------------------------------
    per_sample_path = phase1_dir / "ig_per_sample.pkl"
    if per_sample_path.exists():
        with open(per_sample_path, "rb") as f:
            per_sample = pickle.load(f)
        _plot_ig_heatmap(per_sample, path=out_dir / "fig6_ig_heatmap.png")
    else:
        log.warning("Per-sample IG data not found; skipping Figure 6.")

    log.info("All figures saved to %s", out_dir)


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def _plot_ig_bar(ig: dict[int, float], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    steps = [1, 2, 3]
    vals = [ig[s] for s in steps]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in vals]
    bars = ax.bar(STEP_LABELS, vals, color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Mean Information-Gain")
    ax.set_title(title)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005 * (1 if val >= 0 else -1),
                f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def _plot_accuracy_vs_eps(
    eps_vals: list[float],
    accuracies: list[float],
    baseline_acc: Path,
    oracle_acc: Path,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eps_vals, [a * 100 for a in accuracies], "o-", color="#3498db",
            label="IG-deferral", linewidth=2, markersize=8)

    # Add baseline / oracle horizontal reference lines if available.
    if baseline_acc.exists():
        with open(baseline_acc) as f:
            ig = json.load(f)
    if oracle_acc.exists():
        with open(oracle_acc) as f:
            orac = json.load(f)
        ax.axhline(orac["accuracy"] * 100, color="#e74c3c", linestyle="--",
                   label=f"Oracle ({orac['accuracy']*100:.1f}%)")

    ax.set_xscale("symlog", linthresh=0.005)
    ax.set_xlabel("Threshold ε")
    ax.set_ylabel("Final Accuracy (%)")
    ax.set_title("Accuracy vs. Deferral Threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def _plot_deferral_vs_eps(eps_vals: list[float], records: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for step, color, label in [(1, "#2ecc71", "Step 1 (3x)"),
                                 (2, "#3498db", "Step 2 (2y)"),
                                 (3, "#e74c3c", "Step 3 (3x+2y)")]:
        rates = [r[f"deferral_rate_{step}"] * 100 for r in records]
        ax.plot(eps_vals, rates, "o-", color=color, label=label, linewidth=2, markersize=8)
    ax.set_xscale("symlog", linthresh=0.005)
    ax.set_xlabel("Threshold ε")
    ax.set_ylabel("Deferral Rate (%)")
    ax.set_title("Per-step Deferral Rate vs. ε")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def _plot_precision_recall(eps_vals: list[float], records: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    prec = [r["precision_step3"] for r in records]
    rec = [r["recall_step3"] for r in records]
    ax.plot(eps_vals, prec, "s-", color="#9b59b6", label="Precision (step 3)", linewidth=2)
    ax.plot(eps_vals, rec, "^-", color="#e67e22", label="Recall (step 3)", linewidth=2)
    ax.set_xscale("symlog", linthresh=0.005)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Threshold ε")
    ax.set_ylabel("Score")
    ax.set_title("Deferral Criterion: Precision & Recall at Step 3")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def _plot_ig_heatmap(per_sample: list[dict], path: Path, n_display: int = 30) -> None:
    """Heatmap of sample-wise IG trajectories (analogous to Figure 3a of [1])."""
    # Take a random subset for readability.
    rng = np.random.default_rng(42)
    idx = rng.choice(len(per_sample), size=min(n_display, len(per_sample)), replace=False)
    subset = [per_sample[i] for i in sorted(idx)]

    matrix = np.array([[r.get(f"ig_{s}", 0.0) for s in (1, 2, 3)] for r in subset])

    fig, ax = plt.subplots(figsize=(5, max(4, n_display * 0.2)))
    vmax = np.nanpercentile(np.abs(matrix), 95)
    sns.heatmap(
        matrix, ax=ax, cmap="RdYlGn", center=0, vmin=-vmax, vmax=vmax,
        xticklabels=["Step 1", "Step 2", "Step 3"],
        yticklabels=[f"Sample {i}" for i in range(len(subset))],
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Information-Gain"},
    )
    ax.set_title("Sample-wise IG (baseline, no tool)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)
