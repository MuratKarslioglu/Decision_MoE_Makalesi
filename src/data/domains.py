"""Çok-alanlı uzmanlaşma deneyi için alan tanımı: görüntü kaynağı + cevap biçimine göre 4 alan.

Her alan bir expert'e atanır (alan d -> expert d). Yalnızca mevcut soru kayıtlarından türetilir.
"""
from __future__ import annotations

DOMAIN_NAMES = ("diagram", "scene_text", "photo_reasoning", "photo_yesno")


def domain_of(item: dict) -> int:
    """0 diyagram (AI2D), 1 sahne metni (TextVQA), 2 fotoğraf sayma/uzamsal/çıkarım (T1-T3), 3 fotoğraf evet/hayır-varlık (T4, T5)."""
    if item["source"] == "ai2d":
        return 0
    if item["source"] == "textvqa":
        return 1
    return 3 if item["question_type"] in ("T4", "T5") else 2
