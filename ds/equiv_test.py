"""Eşdeğerlik: yama (topk, K=6) = yamasız çıktı (10 soru). Kullanım: venv_ds/bin/python ds/equiv_test.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ds"))
import dsloader  # noqa: E402
from dsrouter import DSRouter  # noqa: E402

from src.data.dataset import DecisionMoEDataset  # noqa: E402

proc, model = dsloader.load_ds()
ds = DecisionMoEDataset("val")
items = ds.items[:: len(ds.items) // 10][:10]
base = [dsloader.answer(proc, model, ds.root / it["image_path"], it["question"]) for it in items]
r = DSRouter(model, "topk")
patched = [dsloader.answer(proc, model, ds.root / it["image_path"], it["question"]) for it in items]
r.close()
print("özdeş:", sum(a == b for a, b in zip(base, patched)), "/", len(items))
for a, b in zip(base, patched):
    print(repr(a), repr(b))
