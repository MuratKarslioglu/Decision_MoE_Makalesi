"""DeepSeek-VL2 üzerinde TOPSIS (64 expert) ile expert seçimi: gate Top-K ile aynı K'da karşılaştırma (val).
Kriterler (expert başına, katman başına): gate olasılığı (benefit), alan önceliği = train'de alanın expert kullanım payı (benefit),
uzmanlık = 1 - H(alan|expert)/log4 (benefit), yük = genel kullanım payı (cost). Alan ÖRACLE olarak bilinir (üst sınır: önceliğin
alan etiketini bilen en iyi hali). Seçilen expertlerin ağırlığı orijinal gate skorundan gelir.
Kullanım: venv_ds/bin/python ds/mcdm.py [n_val=400] [prior_per_domain=150]   Çıktı: artifacts/ds_domain_prior.pt, experiments/results/ds_mcdm_val.json"""
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
from src.data.domains import domain_of  # noqa: E402
from src.utils.metrics import is_correct  # noqa: E402

NV = int(sys.argv[1]) if len(sys.argv) > 1 else 400
PER = int(sys.argv[2]) if len(sys.argv) > 2 else 150
PRIOR = ROOT / "artifacts" / "ds_domain_prior.pt"
OUT = ROOT / "experiments" / "results" / "ds_mcdm_val.json"
K = 3
# (gate, alan önceliği, uzmanlık, yük)
CONFIGS = {"topsis_gate_only": (1, 0, 0, 0), "topsis_g7_p3": (.7, .3, 0, 0), "topsis_g5_p5": (.5, .5, 0, 0),
           "topsis_g4_p3_s2_l1": (.4, .3, .2, .1), "topsis_g2_p8": (.2, .8, 0, 0)}


def build_prior(proc, model):
    ds = DecisionMoEDataset("train")
    use = []
    for d in range(4):
        items = [it for it in ds.items if domain_of(it) == d][:PER]
        r = DSRouter(model, "topk", 6)
        for it in items:
            dsloader.answer(proc, model, ds.root / it["image_path"], it["question"])
        use.append(r.utilization.cpu().float()); r.close()
        print("öncelik alan", d, len(items), flush=True)
    u = torch.stack(use, 1)                                   # (L, D, E)
    p = u / u.sum(2, keepdim=True)                            # alan içi kullanım payı
    q = u / u.sum(1, keepdim=True).clamp_min(1)               # expert başına alan dağılımı
    ent = -(q * (q + 1e-9).log()).sum(1) / torch.log(torch.tensor(4.0))
    prior = {"prior": p, "spec": 1 - ent, "load": u.sum(1) / u.sum((1, 2))[:, None]}
    torch.save(prior, PRIOR)
    return prior


def main() -> None:
    proc, model = dsloader.load_ds()
    prior = torch.load(PRIOR) if PRIOR.exists() else build_prior(proc, model)
    ds = DecisionMoEDataset("val")
    items = ds.items[:: max(1, len(ds.items) // NV)][:NV]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    runs = [("topk_k3", dict(mode="topk", k=K))] + [(n, dict(mode="mcdm", k=K, mcdm={**prior, "w": torch.tensor(w, dtype=torch.float32)}))
                                                  for n, w in CONFIGS.items()]
    for name, kw in runs:
        if name in res:
            continue
        r = DSRouter(model, **kw)
        ok, t0 = [], time.time()
        for it in items:
            r.domain = domain_of(it)
            ok.append(int(is_correct(dsloader.answer(proc, model, ds.root / it["image_path"], it["question"]), it["answer"])))
        r.close()
        res[name] = {"correct": ok, "accuracy": sum(ok) / len(ok), "seconds": time.time() - t0}
        print(f"{name:20s} acc={res[name]['accuracy']:.3f} {res[name]['seconds']:.0f}s", flush=True)
        OUT.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
