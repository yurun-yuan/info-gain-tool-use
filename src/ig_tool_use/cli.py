"""CLI entry point for the IG-guided tool-deferral experiments.

Usage:
    ig-experiment phase1 [--device cuda]
    ig-experiment phase2 [--device cuda]
    ig-experiment phase3 [--device cuda]
    ig-experiment phase4
    ig-experiment all   [--device cuda]
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.logging import RichHandler

from ig_tool_use.config import ExperimentConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger(__name__)

app = typer.Typer(help="Information-Gain Guided Tool Deferral experiments.")


def _make_cfg(
    output_dir: Path,
    device: str,
    model: str,
    supervisor_model: str,
    n_train: int,
    n_eval: int,
) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.output_dir = output_dir
    cfg.checkpoint_dir = output_dir / "checkpoints" / "supervisor"
    cfg.device = device
    cfg.rollout.model_name = model
    cfg.supervisor.model_name = supervisor_model
    cfg.rollout.n_train = n_train
    cfg.rollout.n_eval = n_eval
    return cfg


@app.command()
def phase1(
    output_dir: Path = typer.Option(Path("results"), help="Root results directory."),
    device: str = typer.Option("cuda", help="PyTorch device for supervisor."),
    model: str = typer.Option("meta-llama/Meta-Llama-3-8B", help="Base model (vllm)."),
    supervisor_model: str = typer.Option("gpt2", help="Supervisor model (HF)."),
    n_train: int = typer.Option(2000, help="Training CoT traces."),
    n_eval: int = typer.Option(500, help="Evaluation CoT traces."),
) -> None:
    """Phase 1: Generate CoT traces, train supervisor, profile baseline IG."""
    from ig_tool_use.experiments import phase1 as p1
    cfg = _make_cfg(output_dir, device, model, supervisor_model, n_train, n_eval)
    p1.run(cfg)


@app.command()
def phase2(
    output_dir: Path = typer.Option(Path("results")),
    device: str = typer.Option("cuda"),
    model: str = typer.Option("meta-llama/Meta-Llama-3-8B"),
    supervisor_model: str = typer.Option("gpt2"),
    n_train: int = typer.Option(2000),
    n_eval: int = typer.Option(500),
) -> None:
    """Phase 2: Oracle tool deferral — always replace step 3 with the calculator."""
    from ig_tool_use.experiments import phase2 as p2
    cfg = _make_cfg(output_dir, device, model, supervisor_model, n_train, n_eval)
    p2.run(cfg)


@app.command()
def phase3(
    output_dir: Path = typer.Option(Path("results")),
    device: str = typer.Option("cuda"),
    model: str = typer.Option("meta-llama/Meta-Llama-3-8B"),
    supervisor_model: str = typer.Option("gpt2"),
    n_train: int = typer.Option(2000),
    n_eval: int = typer.Option(500),
    epsilon: Optional[str] = typer.Option(
        None,
        help="Comma-separated ε values to sweep; defaults to 0,0.01,0.05,0.1,0.5.",
    ),
) -> None:
    """Phase 3: Threshold-based IG deferral (Algorithm 1) for several ε values."""
    from ig_tool_use.experiments import phase3 as p3
    cfg = _make_cfg(output_dir, device, model, supervisor_model, n_train, n_eval)
    if epsilon is not None:
        cfg.epsilon_values = [float(e.strip()) for e in epsilon.split(",")]
    p3.run(cfg)


@app.command()
def phase4(
    output_dir: Path = typer.Option(Path("results")),
    device: str = typer.Option("cpu", help="Device for loading supervisor (plots only)."),
    model: str = typer.Option("meta-llama/Meta-Llama-3-8B"),
    supervisor_model: str = typer.Option("gpt2"),
    n_train: int = typer.Option(2000),
    n_eval: int = typer.Option(500),
) -> None:
    """Phase 4: Generate comparison plots from Phase 1–3 artifacts."""
    from ig_tool_use.experiments import phase4 as p4
    cfg = _make_cfg(output_dir, device, model, supervisor_model, n_train, n_eval)
    p4.run(cfg)


@app.command()
def all(
    output_dir: Path = typer.Option(Path("results")),
    device: str = typer.Option("cuda"),
    model: str = typer.Option("meta-llama/Meta-Llama-3-8B"),
    supervisor_model: str = typer.Option("gpt2"),
    n_train: int = typer.Option(2000),
    n_eval: int = typer.Option(500),
) -> None:
    """Run all four phases sequentially."""
    from ig_tool_use.experiments import phase1 as p1
    from ig_tool_use.experiments import phase2 as p2
    from ig_tool_use.experiments import phase3 as p3
    from ig_tool_use.experiments import phase4 as p4

    cfg = _make_cfg(output_dir, device, model, supervisor_model, n_train, n_eval)
    log.info("=== Phase 1 ===")
    p1.run(cfg)
    log.info("=== Phase 2 ===")
    p2.run(cfg)
    log.info("=== Phase 3 ===")
    p3.run(cfg)
    log.info("=== Phase 4 ===")
    p4.run(cfg)


if __name__ == "__main__":
    app()
