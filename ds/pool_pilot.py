"""Aşama 0 pilotu: katman başına expert havuzu küçültmenin doğruluğa etkisi + gate Top-6'nın soru başına farklı expert sayısı (IO göstergesi).
Havuz = prefill'de toplam gate kütlesi en yüksek m expert (pool) veya rastgele m expert (pool_random); token başına Top-6 yalnız havuzda.
Kullanım: venv_ds/bin/python ds/pool_pilot.py [n=400]   Çıktı: experiments/results/ds_pool_pilot.json (val; soru bazında correct + farklı expert sayısı)"""
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
OUT = ROOT / "experiments" / "results" / "ds_pool_pilot.json"
CONFIGS = [("topk_k6", dict(mode="topk", k=6))] + [(f"pool_m{m}", dict(mode="pool", k=6, pool_m=m)) for m in (32, 24, 16, 12, 8)] \
    + [(f"poolrand_m{m}", dict(mode="pool_random", k=6, pool_m=m)) for m in (32, 16)]


def main() -> None:
    proc, model = dsloader.load_ds()
    ds = DecisionMoEDataset("val")
    items = ds.items[:: max(1, len(ds.items) // N)][:N]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, kw in CONFIGS:
        if name in res:
            continue
        r = DSRouter(model, **kw)
        ok, dist, t0 = [], [], time.time()
        for it in items:
            r.begin_episode()
            ok.append(int(is_correct(dsloader.answer(proc, model, ds.root / it["image_path"], it["question"]), it["answer"])))
            dist.append(r.used.sum(1).float().mean().item())             # katman ortalaması farklı expert sayısı
        r.close()
        res[name] = {"ids": [it["id"] for it in items], "correct": ok, "accuracy": sum(ok) / len(ok),
                     "distinct_experts_per_layer": sum(dist) / len(dist), "seconds": time.time() - t0}
        print(f"{name:14s} acc={res[name]['accuracy']:.3f} distinct/layer={res[name]['distinct_experts_per_layer']:.1f} {res[name]['seconds']:.0f}s", flush=True)
        OUT.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
