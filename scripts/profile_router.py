"""Router maliyet profili: TOPSIS modunda süre nereye gidiyor? (tracker CPU kopyaları, karar matrisi, TOPSIS, expert hesabı)

Kullanım: python scripts/profile_router.py [N_SORU]. Fonksiyonlar sarılıp cuda.synchronize ile süre toplanır; standard_topk ile karşılaştırılır.
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.routing.criteria as crit  # noqa: E402
import src.routing.router as rt  # noqa: E402
from src.backbone.loader import load_backbone, load_model_config  # noqa: E402
from src.data.dataset import DecisionMoEDataset  # noqa: E402
from src.routing.runner import run_item  # noqa: E402
from src.utils.experiment import build_router, load_variant  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
acc: dict[str, float] = defaultdict(float)


def timed(label, fn):
    def wrapper(*a, **k):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = fn(*a, **k)
        torch.cuda.synchronize(); acc[label] += time.perf_counter() - t0
        return out
    return wrapper


def main() -> None:
    tok, model, proc = load_backbone(load_model_config())
    artifact = torch.load(ROOT / "artifacts" / "prototypes.pt")
    ds = DecisionMoEDataset("val")
    items = [ds[i] for i in range(0, len(ds), max(1, len(ds) // N))][:N]

    def run(variant: str) -> float:
        router = build_router(model, artifact, load_variant(variant))
        for it in items[:3]:
            run_item(model, router, tok, proc, it, ds.root)     # ısınma
        acc.clear()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for it in items:
            run_item(model, router, tok, proc, it, ds.root)
        torch.cuda.synchronize(); total = time.perf_counter() - t0
        router.close()
        return total

    base = run("standard_topk")
    crit.CriteriaTracker.update_batch = timed("tracker.update_batch (CPU kopyası)", crit.CriteriaTracker.update_batch)
    crit.CriteriaTracker.decision_matrix_batch = timed("karar matrisi D", crit.CriteriaTracker.decision_matrix_batch)
    rt.compute_topsis_scores = timed("TOPSIS skoru", rt.compute_topsis_scores)
    total = run("fixed_mcdm_cosine")
    print(f"\nstandard_topk   : {1000 * base / N:7.1f} ms/soru")
    print(f"fixed_mcdm_cosine: {1000 * total / N:7.1f} ms/soru  (+{100 * (total / base - 1):.1f}%)")
    for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
        print(f"   {k:38s} {1000 * v / N:7.2f} ms/soru  ({100 * v / (total - base):5.1f}% fark içinde)")


if __name__ == "__main__":
    main()
