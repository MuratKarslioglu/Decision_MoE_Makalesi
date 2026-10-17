"""questions.json / splits/*.json okuyan PyTorch Dataset (Briefing §5)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

DATASET_ROOT = Path(__file__).resolve().parents[2] / "decisionmoe_dataset"
SPLITS = ("train", "val", "test")
REQUIRED_KEYS = ("id", "image_id", "image_path", "source", "question_type", "question", "answer", "language")


class DecisionMoEDataset(Dataset):
    """Bir split'in soru kayıtlarını döndürür; görüntü yüklemesi collator'a bırakılır (lazy)."""

    def __init__(self, split: str, root: Path = DATASET_ROOT, question_types: tuple[str, ...] | None = None,
                 limit: int | None = None) -> None:
        if split not in SPLITS:
            raise ValueError(f"split {SPLITS} içinden olmalı: {split!r}")
        self.root = Path(root)
        with open(self.root / "splits" / f"{split}.json", encoding="utf-8") as f:
            items: list[dict[str, Any]] = json.load(f)
        for it in items[:1]:
            missing = [k for k in REQUIRED_KEYS if k not in it]
            if missing:
                raise ValueError(f"kayıtta eksik alanlar: {missing}")
        if question_types is not None:
            items = [it for it in items if it["question_type"] in question_types]
        self.items = items[:limit] if limit else items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]
