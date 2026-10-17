"""Duyarlılık testi: backbone'un doğruluğu expert seçimine duyarlı mı? (PPO'nun öğrenebileceği alan var mı?)

Aynı val alt kümesinde Top-K (K=1..4), rastgele expert, tek expert'e sabitleme ve Fixed-MCDM karşılaştırılır.
Çıktı: experiments/results/sensitivity_val.json + yazdırılan tablo (referans: topk_k2'ye göre eşleştirilmiş fark ve %95 GA).
"""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backbone.loader import load_backbone, load_model_config  # noqa: E402
from src.data.dataset import DecisionMoEDataset  # noqa: E402
from src.rl.policy import FixedActionPolicy  # noqa: E402
from src.routing.router import DecisionMoERouter  # noqa: E402
from src.routing.runner import run_item  # noqa: E402
from src.utils.metrics import is_correct, is_hallucination  # noqa: E402

N_ITEMS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
OUT = ROOT / "experiments" / "results" / "sensitivity_val.json"
N_BOOT = 2000


def configs() -> dict[str, dict]:
    c = {f"topk_k{k}": dict(mode="topk", k=k) for k in (1, 2, 3, 4)}
    c.update({f"random_k{k}": dict(mode="random", k=k) for k in (1, 2)})
    c["fixed_mcdm_k2"] = dict(mode="fixed_mcdm", k=2)
    for e in range(4):
        u = [1e-6] * 4
        u[e] = 1.0
        c[f"pin_expert{e}_k1"] = dict(mode="direct", policy=FixedActionPolicy(u, 1))
    return c


def paired_bootstrap(a: torch.Tensor, b: torch.Tensor, n: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    """mean(a - b) ve %95 GA (soru bazında yeniden örnekleme)."""
    g = torch.Generator().manual_seed(seed)
    d = (a - b).float()
    idx = torch.randint(0, d.numel(), (n, d.numel()), generator=g)
    means = d[idx].mean(dim=1)
    return d.mean().item(), means.quantile(0.025).item(), means.quantile(0.975).item()


def main() -> None:
    tokenizer, model, image_processor = load_backbone(load_model_config())
    artifact = torch.load(ROOT / "artifacts" / "prototypes.pt")
    ds = DecisionMoEDataset("val")
    step = max(1, len(ds) // N_ITEMS)
    items = [ds[i] for i in range(0, len(ds), step)][:N_ITEMS]
    print(f"{len(items)} val sorusu", flush=True)
    results: dict[str, dict] = {}
    for name, kw in configs().items():
        torch.manual_seed(0)
        router = DecisionMoERouter(model, artifact, **kw)
        ok, hal, t0 = [], [], time.time()
        for it in items:
            r = run_item(model, router, tokenizer, image_processor, it, ds.root)
            ok.append(float(is_correct(r.answer, it["answer"])))
            hal.append(float(is_hallucination(r.answer, it["question_type"])) if it["question_type"] == "T4" else float("nan"))
        router.close()
        t4 = [h for h in hal if h == h]
        results[name] = {"correct": ok, "accuracy": sum(ok) / len(ok),
                         "hallucination_rate": sum(t4) / len(t4) if t4 else 0.0, "seconds": time.time() - t0}
        print(f"{name:18s} acc={results[name]['accuracy']:.3f} halluc={results[name]['hallucination_rate']:.3f} "
              f"{results[name]['seconds']:.0f}s", flush=True)
        OUT.write_text(json.dumps({"n": len(items), "results": results}))

    ref = torch.tensor(results["topk_k2"]["correct"])
    print("\n=== topk_k2'ye göre eşleştirilmiş fark (puan), %95 GA ===")
    for name, r in results.items():
        if name == "topk_k2":
            continue
        m, lo, hi = paired_bootstrap(torch.tensor(r["correct"]), ref)
        sig = "ANLAMLI" if lo > 0 or hi < 0 else "-"
        print(f"{name:18s} {100*m:+6.1f}  [{100*lo:+5.1f}, {100*hi:+5.1f}]  {sig}")
    accs = [results[k]["accuracy"] for k in results if k.startswith("pin_")]
    print(f"\ntek-expert sabitleme doğruluk aralığı: {min(accs):.3f} – {max(accs):.3f}")
    print("TAMAMLANDI")


if __name__ == "__main__":
    main()
