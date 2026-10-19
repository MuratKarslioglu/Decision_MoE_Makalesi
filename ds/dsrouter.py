"""DeepSeek-VL2 MoE katmanları için expert seçim yaması (gate çıktısını değiştirir; shared expert'ler etkilenmez).

Modlar: topk (orijinal), random, bottom (en düşük skorlu K), skip_top (sıra 2..K+1), mcdm (64 expert üzerinde TOPSIS: gate olasılığı, alan önceliği, uzmanlık, yük),
topp (token ve katman başına uyarlanır K: en küçük j ki ilk-6 gate kütlesinin ≥ tau'su; K_max=6),
pool / pool_random (prefill'de katman başına m expertlik havuz: toplam gate kütlesine göre / rastgele; token başına Top-K yalnız havuzda; decode aynı havuzu kullanır). K sabit; orijinal gate'in ağırlık normalizasyonu korunur.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.routing.topsis import compute_topsis_scores


class DSRouter:
    def __init__(self, model, mode: str = "topk", k: int | None = None, seed: int = 0, policy=None, deterministic: bool = False,
                 banned: torch.Tensor | None = None, mcdm: dict | None = None, pool_m: int | None = None, tau: float | None = None, k_dec: int | None = None) -> None:
        self.mode, self.k = mode, k
        self.policy, self.deterministic = policy, deterministic      # mode="dyn": adım başına K politikadan gelir
        self.banned = banned                                          # (L, E) bool: seçilemeyen expertler (ablasyon)
        self.mcdm = mcdm        # mode="mcdm": {"prior": (L,D,E), "spec": (L,E), "load": (L,E), "w": (4,)}; self.domain soru başına atanır
        self.domain = 0
        self.tau = tau
        self.k_dec = k_dec                    # topk: decode adımlarında farklı K (prefill'de k)
        self.ksum, self.kcnt = 0, 0           # topp: token-ağırlıklı ortalama K için toplamlar
        self.pool_m = pool_m                  # pool modları: havuz boyutu
        self.pools: dict[int, torch.Tensor] = {}
        self.used = None                      # (L, E) bool: bu soruda etkinleşen expertler (IO göstergesi)
        self.steps: list[dict] = []
        self._cur_k = k or 6
        self.gen = torch.Generator(device="cuda").manual_seed(seed)
        self.gates = [m.gate for m in model.modules() if m.__class__.__name__ == "DeepseekV2MoE"]
        self.utilization = torch.zeros(len(self.gates), self.gates[0].n_routed_experts, dtype=torch.long, device="cuda")
        for i, g in enumerate(self.gates):
            g.forward = lambda hs, _g=g, _i=i: self._forward(_i, _g, hs)

    def _decide(self, x: torch.Tensor, scores: torch.Tensor) -> None:
        """Yeni üretim adımı (ilk MoE katmanı): durum -> K. Prefill tek adımdır (n_tokens büyük), her decode token bir adım."""
        last = scores[-1]
        ent = -(last * (last + 1e-12).log()).sum() / torch.log(torch.tensor(float(last.numel())))
        top = last.topk(6).values
        feats = torch.stack([torch.tensor(float(x.shape[0] > 1), device=x.device), torch.tensor(len(self.steps) / 10.0, device=x.device),
                             ent, top[0], top.sum()])
        out = self.policy(x[-1].float(), feats, deterministic=self.deterministic)
        self._cur_k = int(out["k"])
        self.steps.append({"hidden": x[-1].float().cpu(), "feats": feats.cpu(), "k": self._cur_k, "logp": out["logp"].detach().cpu(),
                           "value": out["value"].detach().cpu(), "n_tokens": int(x.shape[0])})

    def begin_episode(self) -> None:
        self.steps = []
        self.pools = {}
        self.used = torch.zeros_like(self.utilization, dtype=torch.bool)

    def close(self) -> None:
        for g in self.gates:
            g.__dict__.pop("forward", None)

    def _forward(self, li: int, g, hidden_states):
        h = hidden_states.shape[-1]
        x = hidden_states.reshape(-1, h)
        logits = F.linear(x.float(), g.weight.float(), None)
        scores = logits.softmax(dim=-1, dtype=torch.float32) if g.scoring_func == "softmax" else logits.sigmoid()
        if self.mode == "dyn":
            if li == 0:
                self._decide(x, scores)
            k = self._cur_k
        else:
            k = self.k or g.top_k
            if self.k_dec and x.shape[0] == 1:
                k = self.k_dec
        if self.mode in ("topk", "dyn"):
            sel = scores if self.banned is None else scores.masked_fill(self.banned[li].to(scores.device), -1.0)
            idx = sel.topk(k, dim=-1, sorted=False).indices
        elif self.mode == "topp":
            top = scores.topk(6, dim=-1)
            cum = top.values.cumsum(-1) / top.values.sum(-1, keepdim=True)
            kt = (cum < self.tau).sum(-1) + 1                          # (T,) token başına K
            kt = kt.clamp(max=6)
            self.ksum += int(kt.sum()); self.kcnt += kt.numel()
            idx = top.indices
            keep = torch.arange(6, device=scores.device)[None, :] < kt[:, None]
        elif self.mode in ("pool", "pool_random"):
            if x.shape[0] > 1 or li not in self.pools:            # prefill: havuzu bu katmanda belirle
                if self.mode == "pool":
                    key = scores.sum(0)
                else:
                    key = torch.rand(scores.shape[1], device=scores.device, generator=self.gen)
                pm = torch.zeros(scores.shape[1], dtype=torch.bool, device=scores.device)
                pm[key.topk(self.pool_m).indices] = True
                self.pools[li] = pm
            idx = scores.masked_fill(~self.pools[li], -1.0).topk(k, dim=-1, sorted=False).indices
        elif self.mode == "mcdm":
            m = self.mcdm
            dev = scores.device
            cols = torch.stack([scores, m["prior"][li, self.domain].to(dev).expand_as(scores), m["spec"][li].to(dev).expand_as(scores),
                                m["load"][li].to(dev).expand_as(scores)], dim=-1)           # (T, E, 4)
            cols = cols + 1e-6
            ben = torch.tensor([True, True, True, False], device=dev)
            idx = compute_topsis_scores(cols, m["w"].to(dev).expand(scores.shape[0], -1), ben).topk(k, dim=-1).indices
        elif self.mode == "random":
            idx = torch.rand(scores.shape, device=scores.device, generator=self.gen).topk(k, dim=-1).indices
        elif self.mode == "bottom":
            idx = (-scores).topk(k, dim=-1).indices
        elif self.mode == "skip_top":
            idx = scores.topk(k + 1, dim=-1).indices[:, 1:].contiguous()
        else:
            raise ValueError(self.mode)
        w = scores.gather(1, idx)
        if self.mode == "topp":
            w = w * keep
        if k > 1 and g.norm_topk_prob:
            w = w / (w.sum(-1, keepdim=True) + 1e-20) * g.routed_scaling_factor
        else:
            w = w * g.routed_scaling_factor
        if self.mode != "topp":
            self.ksum += k * x.shape[0]; self.kcnt += x.shape[0]
        if self.used is not None:
            self.used[li, idx.flatten().unique()] = True
        self.utilization[li] += torch.bincount(idx.flatten(), minlength=self.utilization.shape[1])
        return idx, w, None
