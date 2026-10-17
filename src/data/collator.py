"""Batch hazırlama: soru kayıtlarını model girdisine çevirir, sola/sağa padding yapar."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence


def pad_input_ids(seqs: list[torch.Tensor], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """seqs: B adet (T_i,) long -> input_ids (B, T_max), attention_mask (B, T_max) bool (sağdan padding)."""
    ids = pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    mask = pad_sequence([torch.ones_like(s, dtype=torch.bool) for s in seqs], batch_first=True, padding_value=False)
    return ids, mask


class QuestionCollator:
    """Kayıt listesini {input_ids, attention_mask, pixel_values, items} sözlüğüne çevirir."""

    def __init__(self, tokenizer, image_processor, dataset_root: Path) -> None:
        self.tokenizer, self.image_processor, self.root = tokenizer, image_processor, dataset_root
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        from src.backbone.loader import prepare_inputs

        seqs, pixels = [], []
        for it in items:
            ids, pix = prepare_inputs(it, self.root, self.tokenizer, self.image_processor)
            seqs.append(ids[0])
            pixels.append(pix[0])
        input_ids, attention_mask = pad_input_ids(seqs, self.pad_id)
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "pixel_values": torch.stack(pixels), "items": items}
