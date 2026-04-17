"""Supervisor training dataset.

For each ArithmeticSample, we create 4 training examples (one per state t = 0..3):

  "{X_t^M} #|> {Y}"

where X_t^M is the supervisor state at step t (model's accumulated CoT using "||"
separators) and Y is the CORRECT final answer "3x + 2y = {correct_val}".

The loss is computed only on the tokens after "#|>", so the model learns
    p(Y | X_t^M)
for all t simultaneously.  Positive IG at step t means the supervisor is more
confident in Y after seeing step t.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from ig_tool_use.config import SEP_TOKEN
from ig_tool_use.data.arithmetic import ArithmeticSample


def build_training_texts(samples: list[ArithmeticSample]) -> list[str]:
    """Return one training string per (sample, step) pair.

    Each string has format:  "{state} #|> {correct_Y}"
    States with un-parsed model outputs (None values) are included with
    whatever text is available (None → the state may contain "None" as
    a string, which is acceptable; we filter those out).
    """
    texts: list[str] = []
    for s in samples:
        Y = s.supervisor_target
        for state in (s.sup_state_0, s.sup_state_1, s.sup_state_2, s.sup_state_3):
            if state:  # skip empty states (rollout not yet run)
                texts.append(f"{state} {SEP_TOKEN} {Y}")
    return texts


class SupervisorDataset(Dataset):
    """PyTorch Dataset for supervisor training.

    Tokenises each text and masks the input prefix (before #|>) in the labels
    so that the loss is computed only on the target Y tokens.
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer,
        max_length: int = 512,
    ) -> None:
        self.samples: list[dict] = []
        sep = f" {SEP_TOKEN} "

        for text in texts:
            sep_pos = text.find(sep)
            if sep_pos == -1:
                continue

            prefix_text = text[: sep_pos + len(sep)]  # everything up to and including "#|> "

            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            prefix_enc = tokenizer(
                prefix_text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()

            # Mask all prefix tokens so loss is only on Y.
            prefix_len = prefix_enc["input_ids"].shape[1]
            labels[:prefix_len] = -100

            # Skip samples where no target tokens survived truncation.
            if (labels != -100).sum() == 0:
                continue

            self.samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def collate_fn(batch: list[dict]) -> dict:
    """Pad a batch of variable-length samples to the same length."""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    padded: dict[str, list] = {"input_ids": [], "attention_mask": [], "labels": []}

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len
        padded["input_ids"].append(
            torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
        )
        padded["attention_mask"].append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        padded["labels"].append(
            torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )

    return {k: torch.stack(v) for k, v in padded.items()}
