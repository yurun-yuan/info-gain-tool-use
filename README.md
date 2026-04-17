# Information-Gain Guided Tool Deferral

Implementation of the experiments from the project report *"Information-Gain Guided Tool Deferral for Tool-Augmented LLM Agents"* (Yurun Yuan & Shutong Wu, ECE/Math 888, UW-Madison Spring 2026), built on the IG framework of [Ton, Taufiq & Liu, ICML 2025](https://arxiv.org/abs/2411.11984).

---

## Overview

The experiments test whether a lightweight GPT-2 supervisor, trained to predict final answer correctness, can guide a Llama-3-8B agent to defer arithmetic sub-steps to a Python calculator when information-gain (IG) is low.

Four phases:

| Phase | What it does |
|-------|-------------|
| 1 | Generate 2000 CoT traces, train the GPT-2 supervisor, profile baseline per-step IG |
| 2 | Oracle deferral — always replace step 3 with the calculator, measure IG and accuracy gain |
| 3 | Algorithm 1 — threshold-based deferral, sweep ε ∈ {0, 0.01, 0.05, 0.1, 0.5} |
| 4 | Generate comparison plots from Phase 1–3 artifacts |

---

## Requirements

- Python ≥ 3.10
- A CUDA GPU with ≥ 24 GB VRAM (for Llama-3-8B via vllm; Phase 4 plotting runs on CPU)
- [Hugging Face access to Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) — accept the licence and run `huggingface-cli login`

---

## Installation

```bash
# 1. Clone the repo
git clone <repo-url>
cd info-gain-tool-use

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install the package in editable mode
pip install -e .

# 4. (Optional) install dev extras
pip install -e ".[dev]"
```

> **Note:** `vllm` pulls in its own pinned versions of `torch` and `transformers`. If you encounter conflicts, install vllm first and then re-install this package:
> ```bash
> pip install vllm
> pip install -e .
> ```

---

## Usage

### Run all phases sequentially

```bash
ig-experiment all
```

### Run individual phases

```bash
ig-experiment phase1
ig-experiment phase2
ig-experiment phase3
ig-experiment phase4
```

### Common options

All commands accept the following flags (shown with defaults):

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `results/` | Root directory for all artifacts |
| `--device` | `cuda` | PyTorch device for supervisor (`cuda` / `cpu`) |
| `--model` | `meta-llama/Meta-Llama-3-8B` | Base model loaded by vllm |
| `--supervisor-model` | `gpt2` | Supervisor model (HF hub name) |
| `--n-train` | `2000` | Number of training CoT traces |
| `--n-eval` | `500` | Number of evaluation CoT traces |

Phase 3 additionally accepts:

| Flag | Default | Description |
|------|---------|-------------|
| `--epsilon` | `0,0.01,0.05,0.1,0.5` | Comma-separated ε values to sweep |

#### Examples

```bash
# Quick smoke-test with smaller dataset on CPU (Phases 1-3 still need a GPU for vllm)
ig-experiment phase1 --n-train 100 --n-eval 50 --output-dir results_small

# Custom epsilon sweep
ig-experiment phase3 --epsilon "0,0.02,0.1,1.0" --output-dir results

# Plots only (CPU-friendly, requires Phase 1-3 artifacts to exist)
ig-experiment phase4 --device cpu --output-dir results
```

---

## Output layout

After a full run the `results/` directory contains:

```
results/
├── phase1/
│   ├── samples_train.pkl       # 2000 ArithmeticSamples with model rollouts
│   ├── samples_eval.pkl        # 500 held-out samples
│   ├── ig_baseline.json        # mean IG per step under no-tool baseline
│   └── ig_per_sample.pkl       # per-sample IG for all three steps
├── phase2/
│   ├── results_oracle.pkl      # SampleResult list for oracle deferral
│   └── summary_oracle.json     # accuracy + mean IG
├── phase3/
│   ├── results_eps_<ε>.pkl     # SampleResult list for each threshold
│   └── summary_phase3.json     # accuracy, deferral rates, precision/recall per ε
├── phase4/
│   ├── fig1_ig_baseline.png
│   ├── fig2_ig_oracle.png
│   ├── fig3_accuracy_vs_eps.png
│   ├── fig4_deferral_vs_eps.png
│   ├── fig5_precision_recall.png
│   └── fig6_ig_heatmap.png
└── checkpoints/supervisor/final/   # trained GPT-2 LoRA checkpoint
```

Phases cache their heavy artifacts (pickle files) so they can be re-run or re-started without recomputing upstream work.

---

## Project structure

```
info-gain-tool-use/
├── pyproject.toml
├── scripts/
│   └── run_experiment.py       # thin wrapper (same as ig-experiment CLI)
└── src/ig_tool_use/
    ├── config.py               # RolloutConfig, SupervisorConfig, ExperimentConfig
    ├── cli.py                  # Typer CLI (ig-experiment entry point)
    ├── data/
    │   └── arithmetic.py       # ArithmeticSample, prompt builders, output parser
    ├── rollout/
    │   └── vllm_rollout.py     # vllm-based CoT generation (full + per-step)
    ├── supervisor/
    │   ├── dataset.py          # SupervisorDataset, build_training_texts
    │   └── train.py            # GPT-2 + LoRA training, compute_ig, compute_ce_loss
    ├── tools/
    │   └── calculator.py       # Exact Python calculator for the three steps
    ├── agent/
    │   └── ig_agent.py         # IGAgent: no-tool, oracle, and threshold (Alg. 1) modes
    └── experiments/
        ├── phase1.py
        ├── phase2.py
        ├── phase3.py
        └── phase4.py
```

---

## How it works

**Information-Gain** for step *t* is:

```
IG(t) = CE(Y | X_{t-1}) − CE(Y | X_t)
```

where CE is cross-entropy estimated by the supervisor `g_sup(· | X_t)`, and `Y` is the correct final answer.  A near-zero or negative IG signals that the model's step *t* output was uninformative — i.e., the step is a candidate for tool deferral.

**Algorithm 1** (Phase 3): at each step the agent first runs the base model, estimates IG, and defers to the Python calculator if `IG ≤ ε`.

---

## References

- Ton, J.-F., Taufiq, M., & Liu, Y. (2025). *Understanding Chain-of-Thought in LLMs through Information Theory.* ICML 2025. [arXiv:2411.11984](https://arxiv.org/abs/2411.11984)
- Yuan, Y. & Wu, S. (2026). *Information-Gain Guided Tool Deferral for Tool-Augmented LLM Agents.* ECE/Math 888 Project Report, UW-Madison.
