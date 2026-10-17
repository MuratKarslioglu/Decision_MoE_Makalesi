"""Değerlendirme metrikleri: doğruluk, hallucination, FLOPs sayacı (Briefing §3.4, §8)."""
from __future__ import annotations

import re
import string

NEGATIVE_ANSWERS = {"no", "none", "not", "nothing", "there is no", "there are no"}
NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}
_ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    """Küçük harf, noktalama/artikel temizliği, sayı kelimeleri -> rakam."""
    text = text.lower().strip().translate(str.maketrans("", "", string.punctuation))
    tokens = [NUMBER_WORDS.get(t, t) for t in text.split() if t not in _ARTICLES]
    return " ".join(tokens)


def is_correct(prediction: str, reference: str) -> bool:
    """Normalize eşleşme veya referansın tahmin içinde tam-kelime olarak geçmesi (A_t ∈ {0,1})."""
    p, r = normalize_answer(prediction), normalize_answer(reference)
    if not r:
        return False
    return p == r or re.search(rf"(?<!\w){re.escape(r)}(?!\w)", p) is not None


def is_negative(prediction: str) -> bool:
    """Tahmin 'görselde yok' diyor mu? (T4 için)"""
    p = normalize_answer(prediction)
    first = p.split()[0] if p else ""
    return first in NEGATIVE_ANSWERS or any(p.startswith(n) for n in NEGATIVE_ANSWERS if " " in n)


def is_hallucination(prediction: str, question_type: str) -> bool:
    """H_t: T4 (olmayan nesne) sorusuna olumsuz yanıt vermemek hallucination sayılır; diğer tiplerde 0."""
    return question_type == "T4" and not is_negative(prediction)


def accuracy(predictions: list[str], references: list[str]) -> float:
    """Ortalama doğruluk ∈ [0,1]."""
    if len(predictions) != len(references):
        raise ValueError("predictions ve references aynı uzunlukta olmalı")
    return sum(is_correct(p, r) for p, r in zip(predictions, references)) / max(1, len(references))


def hallucination_rate(predictions: list[str], question_types: list[str]) -> float:
    """Yalnızca T4 sorularında hallucination oranı ∈ [0,1]; T4 yoksa 0."""
    t4 = [p for p, t in zip(predictions, question_types) if t == "T4"]
    return sum(is_hallucination(p, "T4") for p in t4) / len(t4) if t4 else 0.0


def expert_flops_fraction(selected_counts: float, n_experts: int) -> float:
    """Seçilen expert sayısının tüm expertlere oranı (FLOPs_baseline = tüm expertler)."""
    return selected_counts / n_experts
