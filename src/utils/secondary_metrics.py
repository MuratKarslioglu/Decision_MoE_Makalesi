"""İkincil metrikler: katı eşleşmenin kaçırdığı anlamca doğru cevapları ölçer (birincil metrik `metrics.is_correct` kalır).

Not: Faz 1'de `src/utils/metrics.py` içine taşınacak. Şimdilik ayrı dosya (koşan hattı etkilememek için).
"""
from __future__ import annotations

from collections import Counter

from src.utils.metrics import normalize_answer


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD tarzı token F1 ∈ [0,1] (normalize edilmiş kelimeler üzerinde)."""
    p, r = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not p or not r:
        return float(p == r)
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def semantic_similarity(predictions: list[str], references: list[str], model=None) -> list[float]:
    """MiniLM (all-MiniLM-L6-v2) kosinüs benzerliği ∈ [-1,1], çiftler üzerinde eleman eleman."""
    if len(predictions) != len(references):
        raise ValueError("predictions ve references aynı uzunlukta olmalı")
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    a = model.encode(predictions, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    b = model.encode(references, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    return (a * b).sum(dim=-1).tolist()
