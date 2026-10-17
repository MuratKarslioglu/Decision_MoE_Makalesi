"""Bir soruyu router ile çalıştırır: bağlam kurar, cevap üretir, routing kayıtlarını döndürür."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.backbone.loader import prepare_inputs
from src.routing.phase import detect_phases
from src.routing.router import DecisionMoERouter, SequenceContext, StepRecord, Transition

N_IMAGE_TOKENS = 576                      # CLIP ViT-L/14-336
SHORT_ANSWER_SUFFIX = " Answer with a single word or short phrase."
ANSWER_MARKER = "ASSISTANT:"


def build_sequence_context(input_ids: torch.Tensor, tokenizer, image_token_index: int,
                           n_image_tokens: int = N_IMAGE_TOKENS) -> SequenceContext:
    """input_ids (T,) -> SequenceContext (genişletilmiş uzunluk T - 1 + n_image_tokens)."""
    phases = detect_phases(
        input_ids, image_token_index, n_image_tokens,
        question_mark_ids=tokenizer.encode("?", add_special_tokens=False),
        answer_marker_ids=tokenizer.encode(ANSWER_MARKER, add_special_tokens=False),
    )
    pos = int((input_ids == image_token_index).nonzero()[0])
    modality = torch.zeros_like(phases)
    modality[pos : pos + n_image_tokens] = 1
    return SequenceContext(modality=modality, phases=phases)


@dataclass
class ItemResult:
    answer: str
    records: list[StepRecord]
    transitions: list[Transition]
    n_generated: int


def run_item(model, router: DecisionMoERouter, tokenizer, image_processor, item: dict[str, Any],
             dataset_root: Path, max_new_tokens: int = 10, short_answer: bool = True) -> ItemResult:
    """Tek soru: router bağlamını kurar, greedy üretir, cevabı ilk satıra keser."""
    from moellava.constants import IMAGE_TOKEN_INDEX

    it = dict(item)
    if short_answer:
        it["question"] = it["question"] + SHORT_ANSWER_SUFFIX
    ids, pix = prepare_inputs(it, dataset_root, tokenizer, image_processor)
    ctx = build_sequence_context(ids[0], tokenizer, IMAGE_TOKEN_INDEX)
    router.begin_episode(ctx, max_steps=max_new_tokens + 1)
    with torch.no_grad():
        out = model.generate(ids, images=pix, do_sample=False, max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.eos_token_id)
    records = router.end_episode()
    text = tokenizer.decode([t for t in out[0].tolist() if 0 <= t < len(tokenizer)], skip_special_tokens=True)
    answer = text.split("ASSISTANT:")[-1].strip().split("\n")[0].strip()
    return ItemResult(answer, records, list(router.transitions), out.shape[1])
