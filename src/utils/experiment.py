"""Variant yükleme, router kurma ve değerlendirme: train.py / evaluate.py / baseline ortak kodu."""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

from src.data.dataset import DecisionMoEDataset
from src.rl.policy import RoutingPolicy, load_policy_config
from src.routing.router import DecisionMoERouter
from src.routing.runner import run_item
from src.utils.metrics import is_correct, is_hallucination

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"


def load_variant(name_or_path: str) -> dict[str, Any]:
    """experiments/ablation/variant_<ad>.yaml (ad veya tam yol)."""
    p = Path(name_or_path)
    if not p.exists():
        p = ROOT / "experiments" / "ablation" / f"variant_{name_or_path}.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_router(model, artifact: dict, variant: dict, policy: RoutingPolicy | None = None,
                 collect_transitions: bool = False, deterministic: bool = False) -> DecisionMoERouter:
    """Variant tanımından router kurar (use_task_utility=False -> kriter 1 ağırlığı maskelenir)."""
    mask = None if variant.get("use_task_utility", True) else torch.tensor([0.0, 1, 1, 1, 1, 1, 1])
    return DecisionMoERouter(model, artifact, mode=variant["mode"], k=variant.get("k", 2), policy=policy,
                             weight_mask=mask, collect_transitions=collect_transitions, deterministic=deterministic)


def build_policy(variant: dict, device: torch.device) -> RoutingPolicy | None:
    if variant["mode"] == "policy":
        return RoutingPolicy(fixed_k=variant.get("fixed_k")).to(device)
    if variant["mode"] == "direct":   # sadece PPO: 4 expert için öncelik vektörü, nötr başlangıç
        conc = load_policy_config()["policy"]["direct_init_concentration"]
        return RoutingPolicy(n_criteria=4, fixed_k=variant.get("fixed_k"), neutral_init_concentration=conc).to(device)
    return None


DEFAULT_SEED = 42


def seed_tag(seed: int | None) -> str:
    """Varsayılan seed (42) eski dosya adlarını korur; diğer seed'ler `_s<seed>` eki alır."""
    base = "" if seed in (None, DEFAULT_SEED) else f"_s{seed}"
    tag = os.environ.get("DM_TAG")                   # deney profili (örn. uzmanlaşmış expert'ler): dosya adlarına eklenir
    return base + (f"_{tag}" if tag else "")


def prototypes_path() -> Path:
    tag = os.environ.get("DM_TAG")
    return ARTIFACTS / (f"prototypes_{tag}.pt" if tag else "prototypes.pt")


def policy_path(variant_name: str, seed: int | None = None) -> Path:
    return ARTIFACTS / f"policy_{variant_name}{seed_tag(seed)}.pt"


def evaluate_variant(model, tokenizer, image_processor, artifact: dict, variant: dict, split: str,
                     policy: RoutingPolicy | None, limit: int | None, baseline_step_time_s: float | dict[str, float] | None = None,
                     max_new_tokens: int = 10) -> dict[str, Any]:
    """Bir variant'ı split üzerinde deterministik değerlendirir; makale tablolarının ham verisini döndürür."""
    ds = DecisionMoEDataset(split, limit=limit)
    router = build_router(model, artifact, variant, policy, deterministic=True)
    by_type: dict[str, list[float]] = defaultdict(list)
    correct, halluc, t4, ks, flops, step_times, weight_sum, n_steps = 0, 0, 0, [], [], [], None, 0
    tok_flops, tok_total, ks_prefill, ks_decode = 0.0, 0, [], []
    per_item: list[dict[str, Any]] = []
    progress_file = ROOT / "experiments" / "results" / f".progress_eval_{variant['name']}.json"
    t_start = time.time()
    for it in ds:
        if len(per_item) % 10 == 0:   # takip aracı (scripts/progress.py) için
            progress_file.write_text(json.dumps({"i": len(per_item), "n": len(ds), "t0": t_start, "t": time.time()}))
        u0 = router.utilization.clone()
        r = run_item(model, router, tokenizer, image_processor, it, ds.root, max_new_tokens)
        ok = is_correct(r.answer, it["answer"])
        per_item.append({"id": it["id"], "type": it["question_type"], "correct": int(ok), "pred": r.answer,
                         "hallucinated": int(is_hallucination(r.answer, "T4")) if it["question_type"] == "T4" else None,
                         "ks": [rec.k for rec in r.records], "ntok": [rec.n_tokens for rec in r.records],
                         "w": [round(x, 4) for x in (sum(rec.weights for rec in r.records) / len(r.records)).tolist()],
                         "util": (router.utilization - u0).sum(0).tolist()})
        correct += ok
        by_type[it["question_type"]].append(float(ok))
        if it["question_type"] == "T4":
            t4 += 1
            halluc += is_hallucination(r.answer, "T4")
        for rec in r.records:
            ks.append(rec.k); flops.append(rec.flops_fraction); step_times.append(rec.wall_time_s)
            tok_flops += rec.flops_fraction * rec.n_tokens; tok_total += rec.n_tokens
            (ks_prefill if rec.n_tokens > 1 else ks_decode).append(rec.k)
            weight_sum = rec.weights.clone() if weight_sum is None else weight_sum + rec.weights
            n_steps += 1
    progress_file.unlink(missing_ok=True)
    util = router.utilization_fraction()
    router.close()
    n = len(ds)
    return {
        "variant": variant["name"], "split": split, "n": n,
        "accuracy": correct / n,
        "accuracy_by_type": {t: sum(v) / len(v) for t, v in sorted(by_type.items())},
        "hallucination_rate": halluc / t4 if t4 else 0.0,
        "mean_k": sum(ks) / len(ks), "k_distribution": {k: ks.count(k) for k in range(1, 5)},
        "flops_fraction": tok_flops / tok_total,                       # token ağırlıklı (gerçek hesaplama maliyeti)
        "flops_fraction_step": sum(flops) / len(flops),                # adım başına ortalama (eski tanım; yanıltıcı olabilir)
        "mean_k_prefill": sum(ks_prefill) / max(1, len(ks_prefill)), "mean_k_decode": sum(ks_decode) / max(1, len(ks_decode)),
        "mean_step_time_s": sum(step_times) / len(step_times),
        "latency_ratio": (sum(step_times) / len(step_times)) / (sum(baseline_step_time_s.values()) / 2 if isinstance(baseline_step_time_s, dict) else baseline_step_time_s) if baseline_step_time_s else None,
        ("mean_expert_prior" if weight_sum.numel() == 4 else "mean_criteria_weights"): (weight_sum / n_steps).tolist(),
        "expert_utilization": util.tolist(),
        "items": per_item,
    }
