"""DeepSeek-VL2 için Dynamic-K politikası (actor-critic) + PPO (GAE, clipped). Bileşenler MoE-LLaVA hattından bağımsızdır.

Durum: son token hidden (1280 -> 64) + 5 özellik [prefill?, adım/10, gate entropisi, top-1 prob, top-6 kütle]. Aksiyon: K ∈ {1..6}.
Ödül: terminal doğruluk − λ·Σ_t (K_t/6)·(n_t/Σn)  (FLOPs TOKEN AĞIRLIKLI; adım-ortalaması reward hacking'e açıktır, bkz. makale §5.3).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

K_MAX = 6


class DynKPolicy(nn.Module):
    def __init__(self, hidden: int = 1280, proj: int = 64, n_feats: int = 5, width: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden, proj)
        self.trunk = nn.Sequential(nn.Linear(proj + n_feats, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
        self.k_head = nn.Linear(width, K_MAX)
        self.v_head = nn.Linear(width, 1)
        nn.init.zeros_(self.k_head.weight)
        with torch.no_grad():
            self.k_head.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))       # başlangıç: Top-6'ya yakın, keşif açık

    def dist(self, hidden: torch.Tensor, feats: torch.Tensor):
        z = self.trunk(torch.cat([torch.tanh(self.proj(hidden)), feats], dim=-1))
        return Categorical(logits=self.k_head(z)), self.v_head(z).squeeze(-1)

    def forward(self, hidden, feats, deterministic: bool = False) -> dict:
        d, v = self.dist(hidden, feats)
        a = d.probs.argmax(-1) if deterministic else d.sample()
        return {"k": a + 1, "logp": d.log_prob(a), "value": v}


def episode_rewards(steps: list[dict], accuracy: float, lam_flops: float) -> torch.Tensor:
    tot = sum(s["n_tokens"] for s in steps)
    r = torch.tensor([-lam_flops * (s["k"] / K_MAX) * s["n_tokens"] / tot for s in steps])
    r[-1] += accuracy
    return r


def flops_fraction(steps: list[dict]) -> float:
    tot = sum(s["n_tokens"] for s in steps)
    return sum((s["k"] / K_MAX) * s["n_tokens"] for s in steps) / tot


def gae(rewards: torch.Tensor, values: torch.Tensor, lam: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
    adv, last = torch.zeros_like(rewards), 0.0
    for t in reversed(range(rewards.numel())):
        nv = values[t + 1] if t + 1 < rewards.numel() else 0.0
        last = rewards[t] + nv - values[t] + lam * last
        adv[t] = last
    return adv, adv + values


def ppo_update(policy: DynKPolicy, opt, episodes: list[tuple[list[dict], torch.Tensor]], clip: float = 0.2, epochs: int = 4,
               mb: int = 64, vf: float = 0.5, ent: float = 0.01) -> dict:
    H, F_, K, LP, ADV, RET = [], [], [], [], [], []
    for steps, rew in episodes:
        v = torch.stack([s["value"] for s in steps]).float()
        a, r = gae(rew, v)
        H += [s["hidden"] for s in steps]; F_ += [s["feats"] for s in steps]
        K += [s["k"] - 1 for s in steps]; LP += [s["logp"] for s in steps]; ADV.append(a); RET.append(r)
    H, F_, K, LP = torch.stack(H), torch.stack(F_), torch.tensor(K), torch.stack(LP).float()
    ADV, RET = torch.cat(ADV), torch.cat(RET)
    ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
    dev = next(policy.parameters()).device
    st = {"pl": 0.0, "vl": 0.0, "ent": 0.0, "n": 0}
    for _ in range(epochs):
        perm = torch.randperm(len(K))
        for i in range(0, len(K), mb):
            ix = perm[i:i + mb]
            d, v = policy.dist(H[ix].to(dev), F_[ix].to(dev))
            ratio = (d.log_prob(K[ix].to(dev)) - LP[ix].to(dev)).exp()
            a = ADV[ix].to(dev)
            pl = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
            vl = (v - RET[ix].to(dev)).pow(2).mean()
            e = d.entropy().mean()
            loss = pl + vf * vl - ent * e
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5); opt.step()
            st["pl"] += pl.item(); st["vl"] += vl.item(); st["ent"] += e.item(); st["n"] += 1
    return {k: v / max(1, st["n"]) for k, v in st.items() if k != "n"}
