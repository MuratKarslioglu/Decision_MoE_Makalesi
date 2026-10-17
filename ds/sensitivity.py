"""DeepSeek-VL2-Tiny duyarlılık testi: expert SEÇİMİ doğruluğu etkiliyor mu? (400 val, eşleştirilmiş bootstrap)

Kullanım: venv_ds/bin/python ds/sensitivity.py [n=400]   Çıktı: experiments/results/ds_sensitivity_val.json
Karşılaştırma referansı: topk_k6 (modelin orijinal ayarı). random/bottom/skip_top "yanlış expert seçimi" kontrolleridir.
"""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ds"))
import dsloader  # noqa: E402
from dsrouter import DSRouter  # noqa: E402

from src.data.dataset import DecisionMoEDataset  # noqa: E402
from src.utils.metrics import is_correct  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
OUT = ROOT / "experiments" / "results" / "ds_sensitivity_val.json"
CONFIGS = [("topk", 6), ("random", 6), ("bottom", 6), ("skip_top", 6), ("topk", 3), ("random", 3), ("topk", 2), ("topk", 1), ("random", 1)]


def main() -> None:
    proc, model = dsloader.load_ds()
    ds = DecisionMoEDataset("val")
    items = ds.items[:: max(1, len(ds.items) // N)][:N]
    res = {}
    for mode, k in CONFIGS:
        name = f"{mode}_k{k}"
        r = DSRouter(model, mode, k)
        t0, ok = time.time(), []
        for it in items:
            ok.append(float(is_correct(dsloader.answer(proc, model, ds.root / it["image_path"], it["question"]), it["answer"])))
        r.close()
        res[name] = {"correct": ok, "accuracy": sum(ok) / len(ok), "seconds": time.time() - t0}
        print(f"{name:14s} acc={res[name]['accuracy']:.3f} {res[name]['seconds']:.0f}s", flush=True)
        OUT.write_text(json.dumps({"n": len(items), "results": res}))
    ref = torch.tensor(res["topk_k6"]["correct"])
    g = torch.Generator().manual_seed(0)
    for name, v in res.items():
        d = torch.tensor(v["correct"]) - ref
        m = d[torch.randint(0, len(d), (2000, len(d)), generator=g)].mean(1)
        print(f"{name:14s} Δ={100*d.mean():+5.1f}pp [{100*m.quantile(.025):+5.1f},{100*m.quantile(.975):+5.1f}]", flush=True)


if __name__ == "__main__":
    main()
