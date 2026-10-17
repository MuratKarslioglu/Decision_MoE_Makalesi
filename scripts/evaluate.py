"""Benchmark değerlendirmesi: bir variant'ı split üzerinde çalıştırır, sonucu experiments/results/ altına yazar.

Kullanım: python scripts/evaluate.py --variant decisionmoe_b --split test [--limit 200]
Policy variant'ları artifacts/policy_<variant>.pt checkpoint'ini yükler.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backbone.loader import load_backbone, load_model_config  # noqa: E402
from src.utils.experiment import build_policy, evaluate_variant, load_variant, policy_path, prototypes_path, seed_tag  # noqa: E402
from src.utils.provenance import collect_meta  # noqa: E402

RESULTS = ROOT / "experiments" / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42, help="hangi seed ile eğitilmiş checkpoint değerlendirilecek")
    ap.add_argument("--baseline-step-ms", type=float, default=None, help="latency oranı için τ_baseline (ms)")
    args = ap.parse_args()

    variant = load_variant(args.variant)
    tok, model, proc = load_backbone(load_model_config())
    device = next(model.parameters()).device
    artifact = torch.load(prototypes_path())
    policy = build_policy(variant, device)
    if policy is not None:
        ckpt = policy_path(variant["name"], args.seed)
        if not ckpt.exists():
            sys.exit(f"checkpoint yok: {ckpt}; önce scripts/train.py çalıştırın")
        policy.load_state_dict(torch.load(ckpt)["policy"])
        policy.eval()
    base = args.baseline_step_ms / 1000 if args.baseline_step_ms else None
    res = evaluate_variant(model, tok, proc, artifact, variant, args.split, policy, args.limit, base)
    res["meta"] = collect_meta(seed=args.seed, extra={"quantization": load_model_config()["quantization"]["mode"], "split": args.split})
    out = RESULTS / f"eval_{variant['name']}{seed_tag(args.seed)}_{args.split}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("expert_utilization", "items")}, indent=1))
    print("kaydedildi:", out)


if __name__ == "__main__":
    main()
