"""Expert MLP'lerine (fc1/fc2) elle LoRA ekler (nf4 Linear4bit üzerinde; peft'e bağımlı değil).

Expert d'nin LoRA'sı yalnızca alan d verisiyle eğitilir (scripts/train_experts.py); çıkarımda tüm LoRA'lar aktiftir.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from src.routing.router import find_moe_layers


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Module, rank: int = 16, alpha: float = 32.0) -> None:
        super().__init__()
        self.base = base
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(base.in_features, rank, device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(rank, base.out_features, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        return y + ((x.to(self.A.dtype) @ self.A) @ self.B * self.scale).to(y.dtype)


def add_expert_lora(model: nn.Module, rank: int = 16, alpha: float = 32.0) -> list[list[nn.Module]]:
    """Tüm MoE katmanlarının tüm expertlerine LoRA ekler. Dönüş: [katman][expert] -> expert modülü."""
    out = []
    for _, moe in find_moe_layers(model):
        row = []
        for ex in moe.deepspeed_moe.experts.deepspeed_experts:
            ex.fc1 = LoRALinear(ex.fc1, rank, alpha)
            ex.fc2 = LoRALinear(ex.fc2, rank, alpha)
            row.append(ex)
        out.append(row)
    return out


def lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu() for n, p in model.named_parameters() if n.endswith((".A", ".B"))}


def load_expert_lora(model: nn.Module, path: Path, rank: int = 16, alpha: float = 32.0) -> None:
    add_expert_lora(model, rank, alpha)
    sd = torch.load(path)
    params = dict(model.named_parameters())
    missing = [k for k in sd if k not in params]
    if missing:
        raise KeyError(f"LoRA anahtarı modelde yok: {missing[:3]}")
    with torch.no_grad():
        for k, v in sd.items():
            params[k].copy_(v)
