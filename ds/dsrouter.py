"""DeepSeek-VL2 MoE katmanları için expert seçim yaması (gate çıktısını değiştirir; shared expert'ler etkilenmez).

Modlar: topk (orijinal), random, bottom (en düşük skorlu K), skip_top (sıra 2..K+1). K sabit; orijinal gate'in ağırlık normalizasyonu korunur.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class DSRouter:
    def __init__(self, model, mode: str = "topk", k: int | None = None, seed: int = 0) -> None:
        self.mode, self.k = mode, k
        self.gen = torch.Generator(device="cuda").manual_seed(seed)
        self.gates = [m.gate for m in model.modules() if m.__class__.__name__ == "DeepseekV2MoE"]
        self.utilization = torch.zeros(len(self.gates), self.gates[0].n_routed_experts, dtype=torch.long, device="cuda")
        for i, g in enumerate(self.gates):
            g.forward = lambda hs, _g=g, _i=i: self._forward(_i, _g, hs)

    def close(self) -> None:
        for g in self.gates:
            g.__dict__.pop("forward", None)

    def _forward(self, li: int, g, hidden_states):
        h = hidden_states.shape[-1]
        x = hidden_states.reshape(-1, h)
        logits = F.linear(x.float(), g.weight.float(), None)
        scores = logits.softmax(dim=-1, dtype=torch.float32) if g.scoring_func == "softmax" else logits.sigmoid()
        k = self.k or g.top_k
        if self.mode == "topk":
            idx = scores.topk(k, dim=-1, sorted=False).indices
        elif self.mode == "random":
            idx = torch.rand(scores.shape, device=scores.device, generator=self.gen).topk(k, dim=-1).indices
        elif self.mode == "bottom":
            idx = (-scores).topk(k, dim=-1).indices
        elif self.mode == "skip_top":
            idx = scores.topk(k + 1, dim=-1).indices[:, 1:]
        else:
            raise ValueError(self.mode)
        w = scores.gather(1, idx)
        if k > 1 and g.norm_topk_prob:
            w = w / (w.sum(-1, keepdim=True) + 1e-20) * g.routed_scaling_factor
        else:
            w = w * g.routed_scaling_factor
        self.utilization[li] += torch.bincount(idx.flatten(), minlength=self.utilization.shape[1])
        return idx, w, None
