"""Expert prototype vektörleri (Briefing §3.1, kriter 1: Task Utility için cosine proxy).

Tanım: MoE katmanı l ve expert e için prototype, o experti top-1 seçen token'ların MoE katmanına
giriş hidden state'lerinin ortalamasıdır. Seçim, katmanın kendi gate'i (wg) ile yapılır.
Prototype'lar bir kez hesaplanır ve frozen kalır.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def find_moe_layers(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    """Modeldeki deepspeed MoE katmanlarını (ad, modül) olarak, derinlik sırasıyla döndürür."""
    from deepspeed.moe.layer import MoE

    return [(n, m) for n, m in model.named_modules() if isinstance(m, MoE)]


class PrototypeAccumulator:
    """MoE katmanlarına forward-pre hook bağlar; expert başına hidden state toplamı ve sayısını tutar.

    sums:   (L, E, H) float32 — top-1 expert'e göre toplanmış giriş hidden state'leri
    counts: (L, E)    long    — her expert'e düşen token sayısı
    """

    def __init__(self, model: torch.nn.Module, n_experts: int, hidden_dim: int) -> None:
        self.layers = find_moe_layers(model)
        device = next(model.parameters()).device
        n_layers = len(self.layers)
        self.sums = torch.zeros(n_layers, n_experts, hidden_dim, dtype=torch.float32, device=device)
        self.counts = torch.zeros(n_layers, n_experts, dtype=torch.long, device=device)
        # (L, E, 2): [text, image] token sayıları; image_mask her forward'dan önce set edilir
        self.modality_counts = torch.zeros(n_layers, n_experts, 2, dtype=torch.long, device=device)
        self.image_mask: torch.Tensor | None = None   # (T,) bool, görüntü token konumları
        self._handles = [m.register_forward_pre_hook(self._make_hook(i)) for i, (_, m) in enumerate(self.layers)]

    def _make_hook(self, layer_idx: int):
        def hook(module: torch.nn.Module, args: tuple) -> None:
            x = args[0].detach().reshape(-1, args[0].shape[-1]).float()   # (T, H)
            wg = module.deepspeed_moe.gate.wg
            top1 = wg(x.to(wg.weight.dtype)).argmax(dim=-1)               # (T,)
            self.sums[layer_idx].index_add_(0, top1, x)
            self.counts[layer_idx] += torch.bincount(top1, minlength=self.counts.shape[1])
            if self.image_mask is not None:
                if self.image_mask.numel() != top1.numel():
                    raise ValueError(f"image_mask uzunluğu {self.image_mask.numel()} != token sayısı {top1.numel()}")
                mod = self.image_mask.to(top1.device).long()
                flat = top1 * 2 + mod
                self.modality_counts[layer_idx] += torch.bincount(flat, minlength=self.counts.shape[1] * 2).view(-1, 2)
        return hook

    def prototypes(self) -> torch.Tensor:
        """(L, E, H) ortalama vektörler; token düşmeyen expert'ler sıfır vektör kalır."""
        return self.sums / self.counts.clamp(min=1).unsqueeze(-1)

    def modality_rates(self) -> torch.Tensor:
        """(L, E, 3) [text, image, mixed]: modality'nin token'larının expert'e düşen oranı.

        Her modality için expert üzerinden toplam 1 (Briefing kriter 2). mixed = text ve image ortalaması.
        """
        mc = self.modality_counts.float()
        rates = mc / mc.sum(dim=1, keepdim=True).clamp(min=1.0)
        return torch.cat([rates, rates.mean(dim=-1, keepdim=True)], dim=-1)

    def close(self) -> None:
        for h in self._handles:
            h.remove()


def cosine_to_prototypes(hidden: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Task Utility kriteri için cosine benzerliği.

    hidden:     (T, H) tek MoE katmanının girişi
    prototypes: (E, H) o katmanın expert prototype'ları
    Returns:    (T, E) cosine ∈ [-1, 1]
    """
    return F.normalize(hidden.float(), dim=-1) @ F.normalize(prototypes.float(), dim=-1).T
