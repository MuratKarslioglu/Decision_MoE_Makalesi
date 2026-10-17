"""Alan duyarlılığı: soru alanı (diyagram/sahne metni/fotoğraf-çıkarım/fotoğraf-evet-hayır) × zorlanan expert doğruluk matrisi.

Beklenti (uzmanlaşma varsa): expert d alan d'de en iyi; oracle (alan→expert d) > Top-2 > rastgele.
Kullanım: python scripts/domain_sensitivity.py <tag> [lora_yolu|-] [alan başına soru=100] [yapılandırmalar: virgüllü]
Çıktı: experiments/results/domain_sensitivity_<tag>.json
"""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backbone.expert_lora import load_expert_lora  # noqa: E402
from src.backbone.loader import load_backbone, load_model_config  # noqa: E402
from src.data.dataset import DecisionMoEDataset  # noqa: E402
from src.data.domains import DOMAIN_NAMES, domain_of  # noqa: E402
from src.rl.policy import FixedActionPolicy  # noqa: E402
from src.routing.router import DecisionMoERouter  # noqa: E402
from src.routing.runner import run_item  # noqa: E402
from src.utils.metrics import is_correct  # noqa: E402

TAG = sys.argv[1]
LORA = None if len(sys.argv) < 3 or sys.argv[2] == "-" else Path(sys.argv[2])
PER_DOM = int(sys.argv[3]) if len(sys.argv) > 3 else 100
OUT = ROOT / "experiments" / "results" / f"domain_sensitivity_{TAG}.json"


def configs() -> dict[str, dict]:
    c = {f"topk_k{k}": dict(mode="topk", k=k) for k in (1, 2, 4)}
    c.update({"random_k1": dict(mode="random", k=1), "fixed_mcdm_k2": dict(mode="fixed_mcdm", k=2)})
    for e in range(4):
        u = [1e-6] * 4
        u[e] = 1.0
        c[f"pin{e}"] = dict(mode="direct", policy=FixedActionPolicy(u, 1))
    keep = sys.argv[4].split(",") if len(sys.argv) > 4 else None
    return {k: v for k, v in c.items() if keep is None or k in keep}


def main() -> None:
    tokenizer, model, image_processor = load_backbone(load_model_config())
    if LORA is not None:
        load_expert_lora(model, LORA)
    artifact = torch.load(ROOT / "artifacts" / "prototypes.pt")
    ds = DecisionMoEDataset("val")
    per: dict[int, list] = {d: [] for d in range(4)}
    for it in ds.items:
        if len(per[domain_of(it)]) < PER_DOM:
            per[domain_of(it)].append(it)
    items = [it for d in range(4) for it in per[d]]
    doms = [domain_of(it) for it in items]
    print({DOMAIN_NAMES[d]: len(v) for d, v in per.items()}, flush=True)
    res: dict[str, list] = {}
    for name, kw in configs().items():
        torch.manual_seed(0)
        router = DecisionMoERouter(model, artifact, **kw)
        t0 = time.time()
        res[name] = [float(is_correct(run_item(model, router, tokenizer, image_processor, it, ds.root).answer, it["answer"]))
                     for it in items]
        router.close()
        row = [sum(c for c, d in zip(res[name], doms) if d == k) / max(1, doms.count(k)) for k in range(4)]
        print(f"{name:14s} " + " ".join(f"{DOMAIN_NAMES[k][:8]}={row[k]:.3f}" for k in range(4)) +
              f"  ort={sum(res[name]) / len(items):.3f} {time.time() - t0:.0f}s", flush=True)
        OUT.write_text(json.dumps({"tag": TAG, "lora": str(LORA), "domains": doms, "correct": res}))
    if not all(f"pin{d}" in res for d in range(4)):
        return
    orc = [res[f"pin{d}"][i] for i, d in enumerate(doms)]
    print(f"ORACLE (alan d -> expert d) ort={sum(orc) / len(orc):.3f}")
    res["oracle"] = orc
    OUT.write_text(json.dumps({"tag": TAG, "lora": str(LORA), "domains": doms, "correct": res}))


if __name__ == "__main__":
    main()
