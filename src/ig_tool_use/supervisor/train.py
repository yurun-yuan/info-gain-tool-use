"""Supervisor model: training with HF Trainer + PEFT LoRA, and inference utilities.

The supervisor model  g_sup  approximates  p(Y | X_t^M).
  * Training: fine-tune GPT-2 on (state, Y) pairs with loss masked to Y tokens.
  * Inference: compute per-sample CE loss to estimate information-gain.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from ig_tool_use.config import SEP_TOKEN, SupervisorConfig
from ig_tool_use.supervisor.dataset import SupervisorDataset, build_training_texts

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thin wrapper to keep model + tokenizer together
# ---------------------------------------------------------------------------

class SupervisorModel:
    def __init__(self, model, tokenizer, device: str = "cuda", max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.model.eval()

    def to(self, device: str) -> "SupervisorModel":
        self.device = device
        self.model = self.model.to(device)
        return self


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_supervisor(
    train_texts: list[str],
    cfg: SupervisorConfig,
    save_dir: Path,
    val_texts: Optional[list[str]] = None,
) -> SupervisorModel:
    """Fine-tune GPT-2 with LoRA on the supervisor training texts.

    Args:
        train_texts: Strings of the form "{X_t^M} #|> {Y}".
        cfg:         SupervisorConfig.
        save_dir:    Where to save the best checkpoint.
        val_texts:   Optional validation texts (split from train_texts if None).

    Returns:
        Trained SupervisorModel.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)

    # LoRA adapts the attention projections; cheap to train on GPT-2.
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=cfg.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Optional train/val split.
    if val_texts is None:
        split = int(len(train_texts) * (1.0 - cfg.val_split))
        val_texts = train_texts[split:]
        train_texts = train_texts[:split]

    train_dataset = SupervisorDataset(train_texts, tokenizer, cfg.max_length)
    val_dataset = SupervisorDataset(val_texts, tokenizer, cfg.max_length)

    log.info("Train examples: %d, Val examples: %d", len(train_dataset), len(val_dataset))

    # HF Trainer expects a DataCollator that handles padding.
    # SupervisorDataset already pads via collate_fn, but we use the
    # default DataCollatorWithPadding here.  The labels have -100 padding
    # baked in so cross-entropy is correct.
    training_args = TrainingArguments(
        output_dir=str(save_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=_pad_collate,
    )
    trainer.train()

    # Save the final adapter weights.
    model.save_pretrained(str(save_dir / "final"))
    tokenizer.save_pretrained(str(save_dir / "final"))

    return SupervisorModel(model, tokenizer, max_length=cfg.max_length)


def _pad_collate(batch: list[dict]) -> dict:
    """Pad a batch to the same length, keeping -100 labels for masked positions."""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    out: dict[str, list[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in batch:
        n = item["input_ids"].shape[0]
        pad = max_len - n
        out["input_ids"].append(F.pad(item["input_ids"], (0, pad), value=0))
        out["attention_mask"].append(F.pad(item["attention_mask"], (0, pad), value=0))
        out["labels"].append(F.pad(item["labels"], (0, pad), value=-100))
    return {k: torch.stack(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_supervisor(
    checkpoint_dir: Path,
    device: str = "cuda",
    max_length: int = 512,
) -> SupervisorModel:
    """Load a fine-tuned supervisor from a saved LoRA checkpoint."""
    from peft import PeftModel

    checkpoint_dir = Path(checkpoint_dir)
    base_model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base, str(checkpoint_dir))
    model = model.merge_and_unload()  # fuse LoRA weights for faster inference

    return SupervisorModel(model, tokenizer, device=device, max_length=max_length)


# ---------------------------------------------------------------------------
# Information-gain estimation
# ---------------------------------------------------------------------------

def compute_ce_loss(
    sup: SupervisorModel,
    state: str,
    target: str,
) -> float:
    """Compute the average CE loss of  target  tokens given  state.

    Implements:  -E[log p(Y | X_t^M)]  via cross-entropy on Y tokens.
    The loss is averaged over target tokens (consistent with HF's default).
    """
    sep = f" {SEP_TOKEN} "
    full_text = state + sep + target
    prefix_text = state + sep

    tokenizer = sup.tokenizer
    device = sup.device

    enc = tokenizer(
        full_text,
        truncation=True,
        max_length=sup.max_length,
        return_tensors="pt",
    )
    prefix_enc = tokenizer(
        prefix_text,
        truncation=True,
        max_length=sup.max_length,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels = input_ids.clone()

    prefix_len = prefix_enc["input_ids"].shape[1]
    labels[0, :prefix_len] = -100

    if (labels != -100).sum() == 0:
        # Target was fully truncated; return a large penalty.
        return 1e6

    with torch.no_grad():
        outputs = sup.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    return outputs.loss.item()


def compute_ig(
    sup: SupervisorModel,
    state_prev: str,
    state_curr: str,
    target: str,
) -> float:
    """Estimate information-gain at a step.

    IG(t) = CE(Y | X_{t-1}^M) - CE(Y | X_t^M)

    Positive IG → step t was informative about the correct answer.
    Negative IG → step t was uninformative or misleading.
    """
    ce_prev = compute_ce_loss(sup, state_prev, target)
    ce_curr = compute_ce_loss(sup, state_curr, target)
    return ce_prev - ce_curr


def compute_dataset_ig(
    sup: SupervisorModel,
    samples,  # list[ArithmeticSample]
) -> dict[int, float]:
    """Compute mean IG per step over a list of ArithmeticSamples.

    Returns {1: mean_ig_1, 2: mean_ig_2, 3: mean_ig_3}.
    """
    ig_sums = {1: 0.0, 2: 0.0, 3: 0.0}
    counts = {1: 0, 2: 0, 3: 0}

    for s in samples:
        Y = s.supervisor_target
        states = [s.sup_state_0, s.sup_state_1, s.sup_state_2, s.sup_state_3]
        for step in (1, 2, 3):
            if states[step - 1] and states[step]:
                ig = compute_ig(sup, states[step - 1], states[step], Y)
                ig_sums[step] += ig
                counts[step] += 1

    return {
        step: (ig_sums[step] / counts[step] if counts[step] > 0 else float("nan"))
        for step in (1, 2, 3)
    }
