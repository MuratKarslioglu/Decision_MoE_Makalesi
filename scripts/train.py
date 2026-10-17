"""Ana eğitim scripti: bir ablation variant'ının PPO policy'sini eğitir (yalnızca policy modu eğitilir).

Kullanım: python scripts/train.py --variant decisionmoe_b --updates 100 [--resume]
Backbone dondurulmuştur; eğitilen: policy ağı (projection dahil). Her güncelleme episodes_per_update soru işler.
"""
import argparse
import dataclasses
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backbone.loader import load_backbone, load_model_config  # noqa: E402
from src.data.dataset import DecisionMoEDataset  # noqa: E402
from src.rl.policy import load_policy_config  # noqa: E402
from src.rl.ppo import PPOTrainer, build_batch, episode_rewards  # noqa: E402
from src.rl.reward import load_reward_weights  # noqa: E402
from src.routing.runner import run_item  # noqa: E402
from src.utils.experiment import (  # noqa: E402
    build_policy, build_router, evaluate_variant, load_variant, policy_path, prototypes_path)
from src.utils.logger import RunLogger, routing_metrics  # noqa: E402
from src.utils.provenance import collect_meta  # noqa: E402
from src.utils.metrics import is_correct, is_hallucination  # noqa: E402


def calibrate_baseline_step_time(model, tok, proc, artifact, ds, n: int = 15) -> dict[str, float]:
    """τ_baseline: Standard Top-K (K=2) ile ortalama adım süresi, prefill ve decode ayrı (ilk soru ısınma)."""
    router = build_router(model, artifact, {"mode": "topk", "k": 2})
    pre, dec = [], []
    for i in range(n + 1):
        recs = run_item(model, router, tok, proc, ds[i], ds.root).records
        if i == 0:
            continue
        pre += [r.wall_time_s for r in recs if r.n_tokens > 1]
        dec += [r.wall_time_s for r in recs if r.n_tokens == 1]
    router.close()
    return {"prefill": sum(pre) / len(pre), "decode": sum(dec) / len(dec)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--updates", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    variant = load_variant(args.variant)
    if variant["mode"] not in ("policy", "direct"):
        sys.exit(f"{variant['name']} eğitilmez (mode={variant['mode']}); evaluate.py kullanın.")
    random.seed(args.seed); torch.manual_seed(args.seed)

    cfg_m, cfg_p = load_model_config(), load_policy_config()
    tok, model, proc = load_backbone(cfg_m)
    device = next(model.parameters()).device
    artifact = torch.load(prototypes_path())
    train = DecisionMoEDataset("train")
    weights = load_reward_weights()
    if variant.get("lambda_flops") is not None:           # λ2 taraması: yalnızca FLOPs katsayısı değişir
        weights = dataclasses.replace(weights, flops=float(variant["lambda_flops"]))
    print(f"reward λ: {weights}", flush=True)
    baseline_t = calibrate_baseline_step_time(model, tok, proc, artifact, train)
    print(f"τ_baseline: prefill {baseline_t['prefill']*1000:.0f} ms, decode {baseline_t['decode']*1000:.0f} ms", flush=True)

    policy = build_policy(variant, device)
    ckpt = policy_path(variant["name"], args.seed)
    start = 0
    if args.resume and ckpt.exists():
        saved = torch.load(ckpt)
        policy.load_state_dict(saved["policy"])
        start = saved["update"]
        print(f"devam: {start}. güncellemeden", flush=True)
    trainer = PPOTrainer(policy, cfg_p)
    logger = RunLogger(variant["name"], {"variant": variant, "ppo": cfg_p["ppo"], "reward": dataclasses.asdict(weights),
                                         "seed": args.seed, "baseline_step_s": baseline_t,
                                         "meta": collect_meta(seed=args.seed, extra={"quantization": cfg_m["quantization"]["mode"]})}, use_wandb=not args.no_wandb)
    router = build_router(model, artifact, variant, policy, collect_transitions=True)
    order = list(range(len(train)))
    random.shuffle(order)
    cursor, n_ep = 0, cfg_p["ppo"]["episodes_per_update"]

    for update in range(start + 1, args.updates + 1):
        t0 = time.time()
        episodes, accs, hals, ks, rews, flops, loads, kpre, kdec = [], [], [], [], [], [], [], [], []
        w_sum, util0 = None, router.utilization.clone()
        for _ in range(n_ep):
            it = train[order[cursor % len(order)]]; cursor += 1
            r = run_item(model, router, tok, proc, it, train.root)
            a, h = is_correct(r.answer, it["answer"]), is_hallucination(r.answer, it["question_type"])
            rw = episode_rewards(r.records, float(a), h, weights, baseline_t)
            episodes.append(([_detach(t) for t in r.transitions], rw))
            accs.append(float(a)); hals.append(float(h) if it["question_type"] == "T4" else float("nan"))
            ks += [x.k for x in r.records]
            tw = sum(x.n_tokens for x in r.records); flops.append(sum(x.flops_fraction * x.n_tokens for x in r.records) / tw)
            kpre += [x.k for x in r.records if x.n_tokens > 1]; kdec += [x.k for x in r.records if x.n_tokens == 1]
            loads += [x.load_std for x in r.records]; rews.append(rw.sum().item())
            for x in r.records:
                w_sum = x.weights.clone() if w_sum is None else w_sum + x.weights
        stats = trainer.update(build_batch(episodes, cfg_p["ppo"]["gamma"], cfg_p["ppo"]["gae_lambda"]))
        used = (router.utilization - util0).sum(0).float()
        metrics = {
            "train/reward": sum(rews) / len(rews), "train/accuracy": sum(accs) / len(accs),
            "train/flops_ratio": sum(flops) / len(flops),   # token ağırlıklı
            "routing/k_prefill": sum(kpre) / max(1, len(kpre)), "routing/k_decode": sum(kdec) / max(1, len(kdec)), "train/load_imbalance": sum(loads) / len(loads),
            "train/hallucination_rate": _nanmean(hals),
            **routing_metrics(w_sum / len(ks), sum(ks) / len(ks), used / used.sum().clamp(min=1)),
            **{f"ppo/{k}": v for k, v in stats.items()}, "time/update_s": time.time() - t0,
        }
        logger.log(metrics, step=update)
        print(f"[{update}/{args.updates}] R={metrics['train/reward']:.3f} acc={metrics['train/accuracy']:.2f} "
              f"K={metrics['routing/mean_K']:.2f}(pre {metrics['routing/k_prefill']:.1f}/dec {metrics['routing/k_decode']:.1f}) FLOPs={metrics['train/flops_ratio']:.2f} H={stats['entropy']:.2f} kl={stats['approx_kl']:.4f} "
              f"{metrics['time/update_s']:.0f}s", flush=True)
        if update % args.eval_every == 0 or update == args.updates:
            torch.save({"policy": policy.state_dict(), "update": update, "variant": variant, "seed": args.seed,
                        "meta": collect_meta(seed=args.seed, extra={"quantization": cfg_m["quantization"]["mode"]})}, ckpt)
            router.close()
            res = evaluate_variant(model, tok, proc, artifact, variant, "val", policy, args.eval_n, baseline_t)
            logger.log({"val/accuracy": res["accuracy"], "val/hallucination_rate": res["hallucination_rate"],
                        "val/mean_K": res["mean_k"], "val/flops_ratio": res["flops_fraction"]}, step=update)
            print(f"   VAL acc={res['accuracy']:.3f} halluc={res['hallucination_rate']:.3f} K={res['mean_k']:.2f}", flush=True)
            router = build_router(model, artifact, variant, policy, collect_transitions=True)
    router.close(); logger.close()
    print("TAMAMLANDI; checkpoint:", ckpt)


def _detach(t):
    t.hidden, t.other, t.weights = t.hidden.clone(), t.other.clone(), t.weights.clone()
    t.log_prob, t.value = t.log_prob.clone(), t.value.clone()
    return t


def _nanmean(xs: list[float]) -> float:
    v = [x for x in xs if x == x]
    return sum(v) / len(v) if v else 0.0


if __name__ == "__main__":
    main()
