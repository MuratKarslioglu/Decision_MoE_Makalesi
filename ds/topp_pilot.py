"""Token-düzeyi uyarlanır K pilotu (Ada-K benzeri sezgisel): K_token = en küçük j, ilk-6 gate kütlesinin >= tau'su. Val 400.
Soru: token/katman granülerliğinde uyarlama, tek tip Top-K sınırının (doğruluk vs token-ağırlıklı FLOPs) üstüne çıkıyor mu?
Kullanım: venv_ds/bin/python ds/topp_pilot.py [n=400]   Çıktı: experiments/results/ds_topp_pilot.json"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ds"))
import dsloader  # noqa: E402
from dsrouter import DSRouter  # noqa: E402

from src.data.dataset import DecisionMoEDataset  # noqa: E402
from src.utils.metrics import is_correct  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
OUT = ROOT / "experiments" / "results" / "ds_topp_pilot.json"
CONFIGS = [(f"topp_t{t}", dict(mode="topp", k=6, tau=t)) for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.15, 0.25, 0.3)] + [("topk_k4", dict(mode="topk", k=4)),
                                                                                                    ("topk_k5", dict(mode="topk", k=5))]


def main() -> None:
    proc, model = dsloader.load_ds()
    ds = DecisionMoEDataset("val")
    items = ds.items[:: max(1, len(ds.items) // N)][:N]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, kw in CONFIGS:
        if name in res:
            continue
        r = DSRouter(model, **kw)
        ok, t0 = [], time.time()
        for it in items:
            ok.append(int(is_correct(dsloader.answer(proc, model, ds.root / it["image_path"], it["question"]), it["answer"])))
        fl = r.ksum / (r.kcnt * 6) if r.kcnt else kw["k"] / 6
        r.close()
        res[name] = {"ids": [it["id"] for it in items], "correct": ok, "accuracy": sum(ok) / len(ok), "flops": fl, "seconds": time.time() - t0}
        print(f"{name:10s} acc={res[name]['accuracy']:.3f} flops={fl:.3f} {res[name]['seconds']:.0f}s", flush=True)
        OUT.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
