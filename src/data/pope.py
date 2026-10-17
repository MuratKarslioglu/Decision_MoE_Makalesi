"""POPE yükleyici ve metrikleri (evet/hayır nesne varlığı sorusu; hallucination ölçer)."""
from __future__ import annotations

import json
import random
from pathlib import Path

POPE_ROOT = Path(__file__).resolve().parents[2] / "pope_data"
POPE_SPLITS = ("random", "popular", "adversarial")


def load_pope(split: str, n: int | None = None, seed: int = 42, root: Path = POPE_ROOT) -> list[dict]:
    """Bir POPE alt kümesini okur; n verilirse görüntü başına dengeli olmasa da sabit seed ile rastgele alt örnekler."""
    if split not in POPE_SPLITS:
        raise ValueError(f"split {POPE_SPLITS} içinden olmalı: {split!r}")
    items = json.load(open(root / f"{split}.json"))
    if n is not None and n < len(items):
        items = random.Random(seed).sample(items, n)
    return items


def parse_yes_no(text: str) -> str:
    """Model cevabından 'yes' / 'no' / 'other' çıkarır (ilk anlamlı kelimeye göre)."""
    words = text.lower().replace(",", " ").replace(".", " ").split()
    for w in words:
        if w in ("yes", "yeah", "yep"):
            return "yes"
        if w in ("no", "not", "nope", "none"):
            return "no"
    return "other"


def pope_metrics(predictions: list[str], labels: list[str]) -> dict[str, float]:
    """POPE standart metrikleri: doğruluk, kesinlik, duyarlılık, F1 ve 'yes' oranı. Pozitif sınıf = 'yes'.

    Ayrıştırılamayan cevap ('other') hata sayılır (etikete göre yanlış, 'yes' olarak sayılmaz).
    """
    if len(predictions) != len(labels):
        raise ValueError("predictions ve labels aynı uzunlukta olmalı")
    p = [parse_yes_no(x) for x in predictions]
    tp = sum(a == "yes" and b == "yes" for a, b in zip(p, labels))
    fp = sum(a == "yes" and b == "no" for a, b in zip(p, labels))
    fn = sum(a != "yes" and b == "yes" for a, b in zip(p, labels))
    correct = sum(a == b for a, b in zip(p, labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n = max(1, len(labels))
    return {"accuracy": correct / n, "precision": precision, "recall": recall, "f1": f1,
            "yes_ratio": sum(a == "yes" for a in p) / n, "unparsed": sum(a == "other" for a in p) / n}
