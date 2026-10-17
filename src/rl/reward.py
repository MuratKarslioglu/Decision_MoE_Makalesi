"""Reward fonksiyonu (Briefing §3.4): R = λ1·A − λ2·F − λ3·L − λ4·I − λ5·H.

λ katsayıları configs/ppo.yaml'dan okunur (araştırmacı tarafından sabit); w_t ise PPO tarafından üretilir.
"""
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "ppo.yaml"


@dataclass(frozen=True)
class RewardWeights:
    accuracy: float
    flops: float
    latency: float
    load_imbalance: float
    hallucination: float


def load_reward_weights(path: Path = CONFIG_PATH) -> RewardWeights:
    with open(path) as f:
        r = yaml.safe_load(f)["reward"]
    return RewardWeights(
        accuracy=r["lambda_accuracy"],
        flops=r["lambda_flops"],
        latency=r["lambda_latency"],
        load_imbalance=r["lambda_load_imbalance"],
        hallucination=r["lambda_hallucination"],
    )


def flops_ratio(
    selected_flops: torch.Tensor,   # (K,) seçilen expert'lerin FLOPs'u; K=0 geçersiz
    baseline_flops: float,          # referans (Standard Top-K) FLOPs
) -> torch.Tensor:                  # () F_t
    """F_t = Σ_{i∈selected} FLOPs(E_i) / FLOPs_baseline. K ≥ 1 zorunludur."""
    if selected_flops.numel() < 1:
        raise ValueError("K_t=0 geçersiz: en az 1 expert seçilmeli")
    return selected_flops.sum() / baseline_flops


def load_imbalance(expert_load: torch.Tensor) -> torch.Tensor:  # (N_experts=4,) -> ()
    """I_t = Std(l_t)."""
    return expert_load.std(unbiased=False)


def compute_reward(
    accuracy: torch.Tensor,        # () veya (B,) A_t ∈ [0,1]
    flops: torch.Tensor,           # F_t
    latency: torch.Tensor,         # L_t = τ_t / τ_baseline
    imbalance: torch.Tensor,       # I_t
    hallucination: torch.Tensor,   # H_t ∈ {0,1}
    k: torch.Tensor | int,         # K_t; < 1 ise ValueError
    weights: RewardWeights,
) -> torch.Tensor:                 # R_t, girişle aynı shape
    """Reward'u hesaplar; K_t < 1 reddedilir (reward hacking koruması)."""
    if bool((torch.as_tensor(k) < 1).any()):
        raise ValueError("K_t >= 1 olmalı")
    return (
        weights.accuracy * accuracy
        - weights.flops * flops
        - weights.latency * latency
        - weights.load_imbalance * imbalance
        - weights.hallucination * hallucination
    )
