"""PPO eğitim döngüsü bileşenleri: bölüm ödülleri, GAE ve clipped-PPO güncellemesi (Briefing §3.3, §3.4).

Bir soru = bir bölüm. Adım-yerel maliyetler (F, L, I) her adıma, doğruluk ve hallucination terimleri
bölümün son adımına yazılır: r_t = -λ2 F_t - λ3 L_t - λ4 I_t  (+ λ1 A - λ5 H  yalnızca t = T).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.rl.policy import RoutingPolicy, load_policy_config
from src.rl.reward import RewardWeights, compute_reward
from src.routing.router import StepRecord, Transition


def _baseline_for(rec: StepRecord, baseline: float | dict[str, float]) -> float:
    """Prefill adımı (çok token) ile decode adımı (tek token) farklı referans süre kullanır."""
    if isinstance(baseline, dict):
        return baseline["prefill" if rec.n_tokens > 1 else "decode"]
    return baseline


def episode_rewards(
    records: list[StepRecord],
    accuracy: float,                  # A ∈ [0,1], bölüm düzeyi
    hallucinated: bool,               # H ∈ {0,1}, bölüm düzeyi
    weights: RewardWeights,
    baseline_step_time_s: float | dict[str, float],   # τ_baseline: float veya {'prefill': s, 'decode': s}
) -> torch.Tensor:                    # (T,) adım ödülleri
    """Bölümün adım ödüllerini hesaplar. compute_reward K_t >= 1'i zorunlu kılar.

    FLOPs terimi token ağırlıklıdır: adımın payı n_tokens/Σn_tokens ile çarpılır; böylece bölüm boyunca toplamı,
    bölümün token ağırlıklı ortalama expert oranına (K/E) eşittir. (Adım başına eşit sayarsak politika ~600 token'lık
    prefill adımında K=4 seçip decode'da K=1 seçerek maliyeti olduğundan düşük gösterir.)"""
    out = []
    last = len(records) - 1
    total_tokens = sum(r.n_tokens for r in records)        # F_t token ağırlıklı: prefill adımı ~600 token işler
    for t, rec in enumerate(records):
        terminal = t == last
        out.append(compute_reward(
            accuracy=torch.tensor(accuracy if terminal else 0.0),
            flops=torch.tensor(rec.flops_fraction * rec.n_tokens / total_tokens),
            latency=torch.tensor(rec.wall_time_s / max(_baseline_for(rec, baseline_step_time_s), 1e-9)),
            imbalance=torch.tensor(rec.load_std),
            hallucination=torch.tensor(float(hallucinated) if terminal else 0.0),
            k=rec.k, weights=weights))
    return torch.stack(out)


def compute_gae(
    rewards: torch.Tensor,    # (T,)
    values: torch.Tensor,     # (T,) V(s_t)
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:   # advantages (T,), returns (T,)
    """Tek bölüm için GAE-λ; bölüm sonunda V = 0 (terminal)."""
    adv = torch.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(rewards.numel())):
        next_v = values[t + 1] if t + 1 < rewards.numel() else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values


@dataclass
class Batch:
    """Birleştirilmiş rollout.

    hidden (N, 2560), other (N, 14), weights (N, 7), k (N,), old_log_prob (N,), advantages (N,), returns (N,)
    """

    hidden: torch.Tensor
    other: torch.Tensor
    weights: torch.Tensor
    k: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def __len__(self) -> int:
        return self.k.numel()


def build_batch(episodes: list[tuple[list[Transition], torch.Tensor]], gamma: float, lam: float) -> Batch:
    """episodes: [(transitions, step_rewards)] -> Batch (avantajlar batch üzerinde normalize edilir)."""
    parts: dict[str, list[torch.Tensor]] = {k: [] for k in ("hidden", "other", "weights", "k", "lp", "adv", "ret")}
    for trans, rewards in episodes:
        if len(trans) != rewards.numel():
            raise ValueError(f"adım sayısı uyuşmuyor: {len(trans)} != {rewards.numel()}")
        trans = [Transition(t.hidden.cpu(), t.other.cpu(), t.weights.cpu(), t.k, t.log_prob.cpu(), t.value.cpu()) for t in trans]
        values = torch.stack([t.value for t in trans]).float()
        adv, ret = compute_gae(rewards.float(), values, gamma, lam)
        parts["hidden"] += [t.hidden for t in trans]
        parts["other"] += [t.other for t in trans]
        parts["weights"] += [t.weights for t in trans]
        parts["k"] += [torch.tensor(t.k) for t in trans]
        parts["lp"] += [t.log_prob for t in trans]
        parts["adv"].append(adv)
        parts["ret"].append(ret)
    adv = torch.cat(parts["adv"])
    if adv.numel() > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return Batch(torch.stack(parts["hidden"]), torch.stack(parts["other"]), torch.stack(parts["weights"]),
                 torch.stack(parts["k"]), torch.stack(parts["lp"]), adv, torch.cat(parts["ret"]))


class PPOTrainer:
    """Clipped-PPO güncellemesi (policy + critic birlikte)."""

    def __init__(self, policy: RoutingPolicy, cfg: dict | None = None) -> None:
        self.policy = policy
        self.cfg = (cfg or load_policy_config())["ppo"]
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.cfg["lr"])

    def update(self, batch: Batch) -> dict[str, float]:
        """Rollout üzerinde `epochs` tur minibatch güncellemesi. Ortalama kayıpları döndürür."""
        c = self.cfg
        device = next(self.policy.parameters()).device
        n = len(batch)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
        n_updates = 0
        for _ in range(c["epochs"]):
            perm = torch.randperm(n)
            for start in range(0, n, c["minibatch_size"]):
                idx = perm[start : start + c["minibatch_size"]]
                b = {name: getattr(batch, name)[idx].to(device) for name in
                     ("hidden", "other", "weights", "k", "old_log_prob", "advantages", "returns")}
                out = self.policy(b["hidden"], b["other"], action=(b["weights"], b["k"]))
                log_ratio = out.log_prob - b["old_log_prob"]
                ratio = log_ratio.exp()
                unclipped = ratio * b["advantages"]
                clipped = ratio.clamp(1 - c["clip_eps"], 1 + c["clip_eps"]) * b["advantages"]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = (out.value - b["returns"]).pow(2).mean()
                entropy = out.entropy.mean()
                loss = policy_loss + c["vf_coef"] * value_loss - c["ent_coef"] * entropy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), c["max_grad_norm"])
                self.optimizer.step()
                with torch.no_grad():
                    stats["policy_loss"] += policy_loss.item()
                    stats["value_loss"] += value_loss.item()
                    stats["entropy"] += entropy.item()
                    stats["approx_kl"] += ((ratio - 1) - log_ratio).mean().item()
                    stats["clip_frac"] += ((ratio - 1).abs() > c["clip_eps"]).float().mean().item()
                n_updates += 1
        return {k: v / max(1, n_updates) for k, v in stats.items()}
