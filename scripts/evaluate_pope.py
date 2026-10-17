"""POPE değerlendirmesi: variant'ları üç POPE alt kümesinde (random/popular/adversarial) çalıştırır.

Kullanım: python scripts/evaluate_pope.py --variants standard_topk standard_topk_k1 decisionmoe_b --n-per-split 1000
Çıktı: experiments/results/pope_<variant>.json (doğruluk, kesinlik, duyarlılık, F1, yes oranı, K, FLOPs, soru bazlı tahmin)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backbone.loader import load_backbone, load_model_config  # noqa: E402
from src.data.pope import POPE_ROOT, POPE_SPLITS, load_pope, parse_yes_no, pope_metrics  # noqa: E402
from src.routing.runner import run_item  # noqa: E402
from src.utils.experiment import build_policy, build_router, load_variant, policy_path, seed_tag  # noqa: E402

RES = ROOT / "experiments" / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--n-per-split", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42, help="POPE soru örnekleme seed'i")
    ap.add_argument("--policy-seed", type=int, default=42, help="hangi seed ile eğitilmiş politika kullanılacak")
    args = ap.parse_args()

    tok, model, proc = load_backbone(load_model_config())
    device = next(model.parameters()).device
    artifact = torch.load(ROOT / "artifacts" / "prototypes.pt")
    for name in args.variants:
        out_path = RES / f"pope_{name}{seed_tag(args.policy_seed)}.json"
        if out_path.exists():
            print(f"{name}: sonuç zaten var", flush=True)
            continue
        variant = load_variant(name)
        policy = build_policy(variant, device)
        if policy is not None:
            ckpt = policy_path(variant["name"], args.policy_seed)
            if not ckpt.exists():
                print(f"{name}: checkpoint yok, atlandı", flush=True)
                continue
            policy.load_state_dict(torch.load(ckpt)["policy"])
            policy.eval()
        router = build_router(model, artifact, variant, policy, deterministic=True)
        result = {"variant": name, "n_per_split": args.n_per_split, "seed": args.seed, "policy_seed": args.policy_seed, "splits": {}, "items": []}
        ks, flops, t0 = [], [], time.time()
        for split in POPE_SPLITS:
            items = load_pope(split, args.n_per_split, args.seed)
            preds = []
            for it in items:
                r = run_item(model, router, tok, proc, it, POPE_ROOT, max_new_tokens=6)
                preds.append(r.answer)
                ks += [rec.k for rec in r.records]
                flops += [rec.flops_fraction for rec in r.records]
                result["items"].append({"id": it["id"], "split": split, "label": it["answer"],
                                        "pred": r.answer, "parsed": parse_yes_no(r.answer)})
            result["splits"][split] = pope_metrics(preds, [it["answer"] for it in items])
            m = result["splits"][split]
            print(f"{name:20s} {split:12s} acc={m['accuracy']:.3f} f1={m['f1']:.3f} yes={m['yes_ratio']:.2f}", flush=True)
        allm = pope_metrics([x["pred"] for x in result["items"]], [x["label"] for x in result["items"]])
        result.update(overall=allm, mean_k=sum(ks) / len(ks), flops_fraction=sum(flops) / len(flops), seconds=time.time() - t0)
        router.close()
        out_path.write_text(json.dumps(result))
        print(f"{name:20s} GENEL        acc={allm['accuracy']:.3f} f1={allm['f1']:.3f} K={result['mean_k']:.2f}", flush=True)
    print("TAMAMLANDI")


if __name__ == "__main__":
    main()
