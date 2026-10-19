"""DeepSeek-VL2 doğal uzmanlaşma ölçümü (val): alan başına expert kullanımı, alanlar arası örtüşme, hedefli ablasyon.

1) Her alanda (src/data/domains.py) Top-6 routing ile (L, 64) expert kullanım sayıları.
2) Örtüşme: katman başına alanların en çok kullanılan N experti arasında Jaccard (ortalama).
3) Hedefli ablasyon: alan d'ye özgü N expert (kullanım payı − diğer alanların ortalaması en büyük) katman başına yasaklanır;
   değerlendirme alanı × ablasyon kümesi doğruluk matrisi; kontrol: rastgele N expert.
Kullanım: venv_ds/bin/python ds/specialization.py [alan başına soru=150] [N=10]   Çıktı: experiments/results/ds_specialization.json
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ds"))
import dsloader  # noqa: E402
from dsrouter import DSRouter  # noqa: E402

from src.data.dataset import DecisionMoEDataset  # noqa: E402
from src.data.domains import DOMAIN_NAMES, domain_of  # noqa: E402
from src.utils.metrics import is_correct  # noqa: E402

PER = int(sys.argv[1]) if len(sys.argv) > 1 else 150
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
OUT = ROOT / "experiments" / "results" / "ds_specialization.json"


def run(proc, model, items, ds, **kw):
    r = DSRouter(model, "topk", 6, **kw)
    ok = [int(is_correct(dsloader.answer(proc, model, ds.root / it["image_path"], it["question"]), it["answer"])) for it in items]
    u = r.utilization.cpu().float()
    r.close()
    return ok, u


def main() -> None:
    proc, model = dsloader.load_ds()
    ds = DecisionMoEDataset("val")
    per = {d: [it for it in ds.items if domain_of(it) == d][:PER] for d in range(4)}
    usage, base = {}, {}
    for d in range(4):
        base[d], usage[d] = run(proc, model, per[d], ds)
        print(DOMAIN_NAMES[d], "Top-6 doğruluk", round(sum(base[d]) / len(base[d]), 3), flush=True)
    p = {d: usage[d] / usage[d].sum(1, keepdim=True) for d in range(4)}               # (L, E) kullanım payı
    L, E = p[0].shape
    top = {d: p[d].topk(N, dim=1).indices for d in range(4)}
    jac = torch.zeros(4, 4)
    for a in range(4):
        for b in range(4):
            j = [len(set(top[a][l].tolist()) & set(top[b][l].tolist())) / len(set(top[a][l].tolist()) | set(top[b][l].tolist())) for l in range(L)]
            jac[a, b] = sum(j) / L
    print("Jaccard (en çok kullanılan %d expert, katman ortalaması):" % N)
    print(jac)
    spec = {d: p[d] - sum(p[o] for o in range(4) if o != d) / 3 for d in range(4)}
    sets = {f"ablate_{DOMAIN_NAMES[d]}": spec[d].topk(N, dim=1).indices for d in range(4)}
    g = torch.Generator().manual_seed(0)
    sets["ablate_random"] = torch.stack([torch.randperm(E, generator=g)[:N] for _ in range(L)])
    abl = {"baseline": [sum(base[d]) / len(base[d]) for d in range(4)]}
    detail = {"baseline": base}
    for name, idx in sets.items():
        banned = torch.zeros(L, E, dtype=torch.bool)
        banned.scatter_(1, idx, True)
        row, det = [], {}
        for d in range(4):
            ok, _ = run(proc, model, per[d], ds, banned=banned.cuda())
            det[d] = ok
            row.append(sum(ok) / len(ok))
        abl[name], detail[name] = row, det
        print(f"{name:26s}", " ".join(f"{DOMAIN_NAMES[d][:9]}={row[d]:.3f}" for d in range(4)), flush=True)
    OUT.write_text(json.dumps({"per": PER, "N": N, "jaccard": jac.tolist(), "accuracy": abl, "correct": {k: {str(d): v for d, v in x.items()} for k, x in detail.items()},
                               "usage_share": {str(d): p[d].tolist() for d in range(4)}}))
    print("SPEC_TAMAM", flush=True)


if __name__ == "__main__":
    main()
