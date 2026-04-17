"""Experiment configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Separator tokens used in supervisor training format (mirrors the paper).
SEP_TOKEN: str = "#|>"
COT_SEP: str = " || "


@dataclass
class RolloutConfig:
    """Config for vllm-based CoT generation from the base model."""

    model_name: str = "meta-llama/Meta-Llama-3-8B"
    n_train: int = 2000
    n_eval: int = 500
    seed: int = 42
    # Per-step generation: keep short so the model doesn't run past the current step.
    max_new_tokens_per_step: int = 40
    # Full CoT generation for Phase 1 baseline.
    max_new_tokens_full: int = 120
    temperature: float = 0.0  # greedy decoding
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85


@dataclass
class SupervisorConfig:
    """Config for the GPT-2 supervisor model."""

    model_name: str = "gpt2"
    max_length: int = 512
    train_batch_size: int = 8
    eval_batch_size: int = 32
    learning_rate: float = 2e-4
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_accumulation_steps: int = 4
    val_split: float = 0.1
    save_steps: int = 200
    eval_steps: int = 200
    logging_steps: int = 50


@dataclass
class ExperimentConfig:
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("results"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/supervisor"))
    # Thresholds ε to sweep in Phase 3.
    epsilon_values: list[float] = field(
        default_factory=lambda: [0.0, 0.01, 0.05, 0.1, 0.5]
    )
    device: str = "cuda"
