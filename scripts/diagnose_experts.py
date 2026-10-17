"""Expert tanısı: aynı girdide expert çıktıları ne kadar farklı? (hipotez A: özdeş/değiştirilebilir)

Her MoE katmanında prefill girdileri için 4 expert'in çıktıları hesaplanır; çiftler arası ortalama kosinüs benzerliği ve
bağıl L2 farkı (||yi-yj||/||yi||) raporlanır. Çıktı: experiments/results/expert_diagnosis[_<tag>].json
Kullanım: python scripts/diagnose_experts.py [n_soru=60] [tag]
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backbone.loader import load_backbone, load_model_config  # noqa: E402
from src.data.dataset import DecisionMoEDataset  # noqa: E402
from src.routing.router import DecisionMoERouter  # noqa: E402
from src.routing.runner import run_item  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
OUT = ROOT / "experiments" / "results" / f"expert_diagnosis{'_' + TAG if TAG else ''}.json"


def main() -> None:
    tokenizer, model, image_processor = load_backbone(load_model_config())
    artifact = torch.load(ROOT / "artifacts" / "prototypes.pt")
    ds = DecisionMoEDataset("val")
    items = [ds[i] for i in range(0, len(ds), max(1, len(ds) // N))][:N]
    router = DecisionMoERouter(model, artifact, mode="topk", k=2)
    n_e = router.n_experts
    pairs = list(combinations(range(n_e), 2))
    cos = torch.zeros(len(router.layers), len(pairs))
    rel = torch.zeros_like(cos)
    cnt = torch.zeros(len(router.layers))
    handles = []
    for pos, (_, moe) in enumerate(router.layers):
        layer = moe.deepspeed_moe

        def hook(_mod, args, _pos=pos, _layer=layer):
            x = args[0].reshape(-1, args[0].shape[-1])
            if x.shape[0] < 8:          # decode adımlarını atla
                return
            with torch.no_grad():
                ys = []
                for e in range(n_e):
                    y = _layer.experts.deepspeed_experts[e](x)
                    ys.append((y[0] if isinstance(y, tuple) else y).float())
                for j, (a, b) in enumerate(pairs):
                    cos[_pos, j] += F.cosine_similarity(ys[a], ys[b], dim=-1).mean().cpu()
                    rel[_pos, j] += ((ys[a] - ys[b]).norm(dim=-1) / ys[a].norm(dim=-1).clamp_min(1e-6)).mean().cpu()
            cnt[_pos] += 1
        handles.append(layer.register_forward_pre_hook(hook))
    for i, it in enumerate(items):
        run_item(model, router, tokenizer, image_processor, it, ds.root, max_new_tokens=1)
        if i % 10 == 0:
            print(i, flush=True)
    for h in handles:
        h.remove()
    router.close()
    c = cnt.clamp_min(1).unsqueeze(1)
    res = {"n": len(items), "pairs": pairs, "cosine": (cos / c).tolist(), "rel_l2": (rel / c).tolist(),
           "mean_cosine": float((cos / c).mean()), "mean_rel_l2": float((rel / c).mean())}
    OUT.write_text(json.dumps(res))
    for l in range(len(router.layers)):
        print(f"katman {l}: kosinüs {(cos / c)[l].mean():.3f}  bağıl L2 {(rel / c)[l].mean():.3f}")
    print(f"ORTALAMA kosinüs {res['mean_cosine']:.3f}  bağıl L2 {res['mean_rel_l2']:.3f}")


if __name__ == "__main__":
    main()
