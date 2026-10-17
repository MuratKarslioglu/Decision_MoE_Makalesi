"""TOPSIS katmanı (saf PyTorch): 4 expert'i 7 kriter ve PPO'dan gelen w_t ağırlıklarıyla ranklar.

Briefing §3.1 adımları: L2 normalizasyon -> ağırlıklandırma -> ideal/anti-ideal -> mesafeler -> skor -> Top-K.
Tüm fonksiyonlar tek örnek (N, M) veya batch (B, N, M) girişini destekler.
"""
from pathlib import Path

import torch
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "topsis.yaml"


def load_topsis_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def benefit_mask_from_config(config: dict) -> torch.Tensor:
    """(N_criteria=7,) bool — True=benefit, False=cost."""
    return torch.tensor([c["type"] == "benefit" for c in config["criteria"]], dtype=torch.bool)


def compute_topsis_scores(
    decision_matrix: torch.Tensor,   # (N_experts=4, N_criteria=7) veya (B, 4, 7)
    weights: torch.Tensor,            # (N_criteria=7,) veya (B, 7) — PPO'dan gelir, sum=1
    benefit_mask: torch.Tensor,       # (N_criteria=7,) bool — True=benefit
    eps: float = 1e-8,
) -> torch.Tensor:                    # (N_experts=4,) veya (B, 4) TOPSIS skoru ∈ [0,1]
    """TOPSIS skorlarını hesaplar. Yüksek skor = ideal çözüme yakın."""
    # Adım 1: sütun bazlı L2 normalizasyon (expert ekseni boyunca)
    norm = decision_matrix.pow(2).sum(dim=-2, keepdim=True).sqrt().clamp_min(eps)
    r = decision_matrix / norm
    # Adım 2: ağırlıklandırma
    v = r * weights.unsqueeze(-2)
    # Adım 3: ideal (A+) ve anti-ideal (A-) çözümler
    v_max = v.max(dim=-2).values
    v_min = v.min(dim=-2).values
    a_pos = torch.where(benefit_mask, v_max, v_min)
    a_neg = torch.where(benefit_mask, v_min, v_max)
    # Adım 4: Öklid mesafeleri
    d_pos = (v - a_pos.unsqueeze(-2)).pow(2).sum(dim=-1).sqrt()
    d_neg = (v - a_neg.unsqueeze(-2)).pow(2).sum(dim=-1).sqrt()
    # Adım 5: skor
    denom = d_pos + d_neg
    # Tüm expert'ler özdeşse (denom=0) skor tarafsız 0.5 olur
    return torch.where(denom < eps, torch.full_like(denom, 0.5), d_neg / denom.clamp_min(eps))


def select_top_k(
    scores: torch.Tensor,   # (N_experts=4,) veya (B, 4)
    k: int,                 # K_t ∈ {k_min..k_max}
    k_min: int = 1,
    k_max: int = 4,
) -> torch.Tensor:          # (k,) veya (B, k) seçilen expert indeksleri (skora göre azalan)
    """Adım 6: en yüksek TOPSIS skorlu K expert'i seçer. K < k_min veya K > k_max hata verir."""
    if not k_min <= k <= k_max:
        raise ValueError(f"K={k} geçersiz; {k_min}..{k_max} aralığında olmalı")
    return scores.topk(k, dim=-1).indices
