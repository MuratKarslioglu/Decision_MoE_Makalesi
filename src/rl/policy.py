"""PPO policy ağı: durum -> (w_t ∈ Δ^6, K_t ∈ {1,2,3,4}) (Briefing §3.2, §3.3).

Giriş: h_t (2560) ham hidden state + diğer 14 özellik. h_t, eğitilebilir Linear(2560 -> 64) ile
indirgenir. Çıkış: w_t için Dirichlet, K_t için Categorical; ayrıca değer tahmini (critic).
Alternatif A (fixed_k verilirse): K sabit, yalnızca w_t öğrenilir.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.distributions import Categorical, Dirichlet

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "ppo.yaml"


def load_policy_config(path: Path = CONFIG_PATH) -> dict:
    """configs/ppo.yaml dosyasını okur."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def state_dim_from_config(cfg: dict) -> int:
    """PPO giriş boyutu: proj_dim + m + d + u + p + l + c."""
    s = cfg["state"]
    return s["proj_dim"] + sum(s[k] for k in (
        "modality_dim", "entropy_dim", "uncertainty_dim", "phase_dim", "load_dim", "budget_dim"))


@dataclass
class PolicyOutput:
    """Bir policy çağrısının sonucu.

    weights:  (B, 7) simplex üzerinde, satır toplamı 1
    k:        (B,) long ∈ {k_min..k_max}
    log_prob: (B,) w ve K'nın birleşik log olasılığı
    entropy:  (B,)
    value:    (B,) critic çıktısı
    """

    weights: torch.Tensor
    k: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


class RoutingPolicy(nn.Module):
    """Actor-critic routing kontrolcüsü."""

    def __init__(self, cfg: dict | None = None, n_criteria: int = 7, fixed_k: int | None = None,
                 neutral_init_concentration: float | None = None) -> None:
        """n_criteria: simplex aksiyonun boyutu (7 = kriter ağırlıkları, 4 = DIRECT kolunda expert önceliği).

        neutral_init_concentration: verilirse Dirichlet başlangıçta girdiden bağımsız uniform ortalamaya sahip olur
        (weight_head ağırlıkları 0, alpha = bu değer); DIRECT kolunda başlangıç davranışı Standard Top-K'ya yakın kalır.
        """
        super().__init__()
        cfg = cfg or load_policy_config()
        self.k_min, self.k_max = cfg["action"]["k_min"], cfg["action"]["k_max"]
        self.fixed_k = fixed_k
        self.hidden_dim = cfg["state"]["hidden_dim"]
        self.simplex_eps = cfg["policy"]["simplex_eps"]
        self.min_concentration = cfg["policy"]["dirichlet_min_concentration"]
        self.n_other = state_dim_from_config(cfg) - cfg["state"]["proj_dim"]
        self.state_dim = state_dim_from_config(cfg)

        self.projection = nn.Linear(self.hidden_dim, cfg["state"]["proj_dim"])
        layers: list[nn.Module] = []
        in_dim = self.state_dim
        for h in cfg["policy"]["hidden_sizes"]:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.weight_head = nn.Linear(in_dim, n_criteria)
        if neutral_init_concentration is not None:
            if neutral_init_concentration <= self.min_concentration:
                raise ValueError("neutral_init_concentration, dirichlet_min_concentration'dan büyük olmalı")
            target = neutral_init_concentration - self.min_concentration          # softplus(bias) = target
            nn.init.zeros_(self.weight_head.weight)
            nn.init.constant_(self.weight_head.bias, math.log(math.expm1(target)))
        self.k_head = nn.Linear(in_dim, self.k_max - self.k_min + 1)
        self.value_head = nn.Linear(in_dim, 1)

    def encode(self, hidden: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        """hidden (B, 2560), other (B, 14) -> s̃_t (B, state_dim)."""
        return torch.cat([self.projection(hidden), other], dim=-1)

    def _distributions(self, state: torch.Tensor) -> tuple[Dirichlet, Categorical, torch.Tensor]:
        z = self.trunk(state)
        alpha = nn.functional.softplus(self.weight_head(z)) + self.min_concentration
        return Dirichlet(alpha), Categorical(logits=self.k_head(z)), self.value_head(z).squeeze(-1)

    def _clip_simplex(self, w: torch.Tensor) -> torch.Tensor:
        w = w.clamp(min=self.simplex_eps)
        return w / w.sum(dim=-1, keepdim=True)

    def forward(
        self,
        hidden: torch.Tensor,        # (B, 2560)
        other: torch.Tensor,         # (B, 14) = [m, d, u, p, l, c]
        action: tuple[torch.Tensor, torch.Tensor] | None = None,   # (w (B,7), k (B,)) -> log_prob hesapla
        deterministic: bool = False,
    ) -> PolicyOutput:
        """Aksiyon örnekler (veya verilen aksiyonun log_prob'unu hesaplar)."""
        state = self.encode(hidden, other)
        w_dist, k_dist, value = self._distributions(state)
        if action is None:
            w = w_dist.mean if deterministic else w_dist.sample()
            k_idx = k_dist.probs.argmax(-1) if deterministic else k_dist.sample()
            w = self._clip_simplex(w)
            k = k_idx + self.k_min
        else:
            w, k = action
            w = self._clip_simplex(w)
            k_idx = k - self.k_min
        if self.fixed_k is not None:
            k = torch.full_like(k_idx, self.fixed_k)
            log_prob = w_dist.log_prob(w)
            entropy = w_dist.entropy()
        else:
            log_prob = w_dist.log_prob(w) + k_dist.log_prob(k_idx)
            entropy = w_dist.entropy() + k_dist.entropy()
        return PolicyOutput(weights=w, k=k, log_prob=log_prob, entropy=entropy, value=value)


class FixedActionPolicy(nn.Module):
    """Her adımda aynı (w veya u, K) aksiyonunu döndüren sahte policy: deney kontrolleri ve testler için.

    Örn. FixedActionPolicy([0,0,1,0], k=1) DIRECT modunda tüm token'ları expert 2'ye sabitler.
    """

    def __init__(self, action: list[float], k: int) -> None:
        super().__init__()
        self.action = torch.as_tensor(action, dtype=torch.float32)
        self.k = k
        self.weight_head = nn.Linear(1, len(action))          # router'ın boyut doğrulaması için

    def forward(self, hidden: torch.Tensor, other: torch.Tensor, action=None, deterministic: bool = False) -> PolicyOutput:
        b = hidden.shape[0]
        z = torch.zeros(b)
        w = self.action.to(hidden.device).expand(b, -1)
        return PolicyOutput(weights=w, k=torch.full((b,), self.k, device=hidden.device), log_prob=z, entropy=z, value=z)
