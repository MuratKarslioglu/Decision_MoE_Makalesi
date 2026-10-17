"""Kontrollü gecikme benchmark'ı: tüm variant'lar aynı sorular, ısınma, tekrarlı ölçüm, variant'lar tekrarlar arasında dönüşümlü.

Kullanım: python scripts/benchmark_latency.py --variants standard_topk standard_topk_k1 ... --n 100 --reps 3
Her ölçüm cuda.synchronize ile çevrilir; soru başına uçtan uca süre ve adım sayısı kaydedilir.
Çıktı: experiments/results/latency_benchmark.json (variant başına ortalama ± std, ms/soru ve ms/adım)
"""
import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backbone.loader import load_backbone, load_model_config  # noqa: E402
from src.data.dataset import DecisionMoEDataset  # noqa: E402
from src.routing.runner import run_item  # noqa: E402
from src.utils.experiment import build_policy, build_router, load_variant, policy_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    tok, model, proc = load_backbone(load_model_config())
    device = next(model.parameters()).device
    artifact = torch.load(ROOT / "artifacts" / "prototypes.pt")
    ds = DecisionMoEDataset("val")
    stride = max(1, len(ds) // args.n)
    items = [ds[i] for i in range(0, len(ds), stride)][: args.n]

    setups = {}
    for name in args.variants:
        v = load_variant(name)
        pol = build_policy(v, device)
        if pol is not None:
            ckpt = policy_path(v["name"])
            if not ckpt.exists():
                print(f"{name}: checkpoint yok, atlandı", flush=True)
                continue
            pol.load_state_dict(torch.load(ckpt)["policy"])
            pol.eval()
        setups[name] = (v, pol)

    def run_once(name: str, subset: list[dict]) -> tuple[float, int]:
        v, pol = setups[name]
        router = build_router(model, artifact, v, pol, deterministic=True)
        steps = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for it in subset:
            steps += len(run_item(model, router, tok, proc, it, ds.root).records)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        router.close()
        return elapsed, steps

    for name in setups:                     # ısınma (derleme/önbellek etkilerini at)
        run_once(name, items[: args.warmup])
    samples: dict[str, list[tuple[float, int]]] = {n: [] for n in setups}
    for rep in range(args.reps):            # dönüşümlü: zamanla kayma tüm variant'lara eşit yansır
        for name in setups:
            samples[name].append(run_once(name, items))
            print(f"rep {rep + 1}/{args.reps} {name:24s} {1000 * samples[name][-1][0] / len(items):7.1f} ms/soru", flush=True)

    result = {"gpu": torch.cuda.get_device_name(), "n": len(items), "reps": args.reps, "variants": {}}
    for name, s in samples.items():
        per_q = [1000 * t / len(items) for t, _ in s]
        per_step = [1000 * t / n for t, n in s]
        result["variants"][name] = {"ms_per_question_mean": st.mean(per_q), "ms_per_question_std": st.pstdev(per_q),
                                    "ms_per_step_mean": st.mean(per_step), "ms_per_step_std": st.pstdev(per_step),
                                    "steps_per_question": s[0][1] / len(items)}
    base = result["variants"].get("standard_topk", {}).get("ms_per_question_mean")
    print(f"\n{'variant':24s} {'ms/soru':>16s} {'ms/adım':>14s} {'oran (top2=1)':>14s}")
    for name, r in result["variants"].items():
        ratio = r["ms_per_question_mean"] / base if base else float("nan")
        r["ratio_vs_standard_topk"] = ratio
        print(f"{name:24s} {r['ms_per_question_mean']:8.1f}±{r['ms_per_question_std']:5.1f} "
              f"{r['ms_per_step_mean']:7.1f}±{r['ms_per_step_std']:4.1f} {ratio:14.3f}")
    (ROOT / "experiments" / "results" / "latency_benchmark.json").write_text(json.dumps(result, indent=1))
    print("TAMAMLANDI")


if __name__ == "__main__":
    main()
