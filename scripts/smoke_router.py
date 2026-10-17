"""Router smoke testi (gerçek model): topk / fixed_mcdm / policy modlarını küçük bir val alt kümesinde karşılaştırır."""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backbone.loader import load_backbone, load_model_config  # noqa: E402
from src.data.dataset import DecisionMoEDataset  # noqa: E402
from src.routing.router import DecisionMoERouter  # noqa: E402
from src.routing.runner import run_item  # noqa: E402
from src.rl.policy import RoutingPolicy  # noqa: E402
from src.utils.metrics import accuracy, hallucination_rate  # noqa: E402

N_ITEMS = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def main() -> None:
    tokenizer, model, image_processor = load_backbone(load_model_config())
    artifact = torch.load(ROOT / "artifacts" / "prototypes.pt")
    ds = DecisionMoEDataset("val")
    step = max(1, len(ds) // N_ITEMS)
    items = [ds[i] for i in range(0, len(ds), step)][:N_ITEMS]
    torch.manual_seed(0)
    configs = {
        "topk_k2": dict(mode="topk", k=2),
        "fixed_mcdm_k2": dict(mode="fixed_mcdm", k=2),
        "fixed_mcdm_k1": dict(mode="fixed_mcdm", k=1),
        "policy_random_init": dict(mode="policy", policy=RoutingPolicy().to(next(model.parameters()).device)),
    }
    for name, kw in configs.items():
        router = DecisionMoERouter(model, artifact, **kw)
        preds, refs, types, ks = [], [], [], []
        t0 = time.time()
        for it in items:
            r = run_item(model, router, tokenizer, image_processor, it, ds.root)
            preds.append(r.answer); refs.append(it["answer"]); types.append(it["question_type"])
            ks += [rec.k for rec in r.records]
        router.close()
        print(f"{name:20s} acc={accuracy(preds, refs):.3f} halluc={hallucination_rate(preds, types):.3f} "
              f"meanK={sum(ks)/len(ks):.2f} {time.time()-t0:.0f}s VRAMpeak={torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)
        print("   örnek:", [(p, r) for p, r in list(zip(preds, refs))[:4]], flush=True)


if __name__ == "__main__":
    main()
