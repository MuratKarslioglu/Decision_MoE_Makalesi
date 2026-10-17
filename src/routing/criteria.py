"""7 kriterin online hesabı: karar matrisi D ∈ R^{4x7} (Briefing §3.1).

Sütun sırası configs/topsis.yaml ile aynıdır:
  0 task_utility (B), 1 modality_match (B), 2 specialization (B), 3 reliability (B),
  4 latency (C),      5 flops (C),          6 load_balance (C)

Tanımlar (briefing'in belirsiz bıraktığı yerler):
  specialization = 1 - H(P(phase | expert)) / log(n_phases); aktivasyon geçmişi reasoning-phase
                   histogramıdır, Laplace yumuşatmalı. Hiç geçmiş yokken 0 (uniform).
  reliability    = 1 - ECE_i, offline tablo (E,); hesaplanana kadar nötr 1.0.
  flops          = (expert param payı) x (son T token'daki aktivasyon yoğunluğu).
  load_balance   = son T token'daki expert aktivasyon oranı (cost: yüksek = kötü).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "topsis.yaml"
CRITERIA_NAMES = (
    "task_utility", "modality_match", "specialization", "reliability",
    "latency", "flops", "load_balance",
)


def load_criteria_config(path: Path = CONFIG_PATH) -> dict:
    """configs/topsis.yaml dosyasını okur."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class StaticCriteriaTables:
    """Offline hesaplanan, bir MoE katmanına ait sabit tablolar.

    modality_rates: (E, M=3) expert'in modality başına training-time aktivasyon oranı
    reliability:    (E,)     1 - ECE_i ∈ [0,1]
    latency_ms:     (E,)     expert forward süresi (ms)
    expert_params:  (E,)     expert parametre sayısı
    """

    modality_rates: torch.Tensor
    reliability: torch.Tensor
    latency_ms: torch.Tensor
    expert_params: torch.Tensor

    @classmethod
    def neutral(cls, n_experts: int, n_modalities: int) -> "StaticCriteriaTables":
        """Ölçüm yokken tüm expertleri eşit sayan nötr tablolar."""
        return cls(
            modality_rates=torch.full((n_experts, n_modalities), 1.0 / n_experts),
            reliability=torch.ones(n_experts),
            latency_ms=torch.ones(n_experts),
            expert_params=torch.ones(n_experts),
        )


@dataclass
class CriteriaTracker:
    """Tek bir MoE katmanı için kriterleri online hesaplar.

    update() her routing adımından sonra seçilen expertleri ve phase'i işler;
    decision_matrix() bir sonraki adımın D matrisini üretir.
    """

    tables: StaticCriteriaTables
    n_experts: int = 4
    window: int = 100
    n_phases: int = 4
    device: torch.device | str = "cpu"
    _buf: torch.Tensor = field(init=False, repr=False)                          # (window, E) halka tampon (multi-hot)
    _phase_counts: torch.Tensor = field(init=False, repr=False)                 # (E, P)

    def __post_init__(self) -> None:
        """Tüm durum ve statik tablolar `device` üzerinde tutulur (her adımda CPU<->GPU kopyası olmaz)."""
        self.device = torch.device(self.device)
        t = self.tables
        self.tables = StaticCriteriaTables(*(x.to(self.device) for x in (
            t.modality_rates, t.reliability, t.latency_ms, t.expert_params)))
        self._buf = torch.zeros(self.window, self.n_experts, device=self.device)
        self._n, self._ptr = 0, 0
        self._phase_counts = torch.zeros(self.n_experts, self.n_phases, device=self.device)

    def reset(self) -> None:
        """Yeni örnek başlangıcında geçmişi sıfırlar."""
        self._buf.zero_()
        self._n, self._ptr = 0, 0
        self._phase_counts.zero_()

    def _push(self, rows: torch.Tensor) -> None:
        """rows (m, E) multi-hot satırlarını halka tampona ekler; yalnızca son `window` satır tutulur."""
        rows = rows[-self.window:]
        m = rows.shape[0]
        idx = (self._ptr + torch.arange(m, device=self.device)) % self.window
        self._buf[idx] = rows
        self._ptr = (self._ptr + m) % self.window
        self._n = min(self.window, self._n + m)

    def update(self, selected: torch.Tensor, phase: int) -> None:
        """selected: (K,) long, seçilen expert indeksleri; phase: 0..n_phases-1."""
        sel = selected.to(self.device)
        onehot = torch.zeros(1, self.n_experts, device=self.device)
        onehot[0, sel] = 1.0
        self._push(onehot)
        self._phase_counts[sel, phase] += 1.0

    def load(self) -> torch.Tensor:
        """l_t: son T token'da her expert'in aktivasyon oranı, (E,) ∈ [0,1]."""
        if self._n == 0:
            return torch.zeros(self.n_experts, device=self.device)
        return self._buf[: self._n].mean(dim=0)

    def specialization(self) -> torch.Tensor:
        """(E,) ∈ [0,1]: 1 - normalize edilmiş phase entropisi (Laplace yumuşatmalı)."""
        p = F.normalize(self._phase_counts + 1.0, p=1, dim=-1)
        entropy = -(p * p.log()).sum(dim=-1) / math.log(self.n_phases)
        return 1.0 - entropy

    def update_batch(self, selected: torch.Tensor, phases: torch.Tensor) -> None:
        """Bir chunk'taki tüm token'ları işler. selected: (S, K) long, phases: (S,) long.

        Load penceresine yalnızca son `window` token girer; phase histogramı tüm token'larla güncellenir.
        """
        sel, ph = selected.detach().to(self.device), phases.detach().to(self.device)
        onehot = torch.zeros(sel.shape[0], self.n_experts, device=self.device).scatter_(1, sel, 1.0)   # (S, E)
        self._push(onehot)
        phase_onehot = F.one_hot(ph, num_classes=self.n_phases).float()                   # (S, P)
        self._phase_counts += onehot.T @ phase_onehot

    def decision_matrix_batch(
        self,
        hidden: torch.Tensor,        # (S, H) MoE katmanı girişi
        prototypes: torch.Tensor,    # (E, H)
        modality: torch.Tensor,      # (S,) long: 0=text, 1=image, 2=mixed
    ) -> torch.Tensor:               # (S, E=4, 7)
        """Her token için karar matrisi. Load/specialization chunk başındaki geçmişten alınır."""
        device = hidden.device
        s = hidden.shape[0]
        cos = F.normalize(hidden.float(), dim=-1) @ F.normalize(prototypes.float().to(device), dim=-1).T   # (S, E)
        task_utility = (cos + 1.0) / 2.0        # [-1,1] -> [0,1]: L2 normalizasyonu için negatif yok
        modality_match = self.tables.modality_rates.to(device)[:, modality].T                              # (S, E)
        load = self.load().to(device)
        params_share = self.tables.expert_params / self.tables.expert_params.sum()
        per_expert = torch.stack((
            self.specialization().to(device),
            self.tables.reliability.to(device),
            self.tables.latency_ms.to(device),
            params_share.to(device) * load,
            load,
        ), dim=1)                                                                                          # (E, 5)
        spec_rel, lat_flops_load = per_expert[:, :2], per_expert[:, 2:]
        shared = torch.cat([spec_rel, lat_flops_load], dim=1).unsqueeze(0).expand(s, -1, -1)              # (S, E, 5)
        return torch.cat([task_utility.unsqueeze(-1), modality_match.unsqueeze(-1), shared], dim=-1)

    def decision_matrix(
        self,
        hidden: torch.Tensor,        # (H,) MoE katmanı girişi
        prototypes: torch.Tensor,    # (E, H)
        modality: int,               # 0=text, 1=image, 2=mixed
    ) -> torch.Tensor:               # (E=4, 7)
        """Tek token için karar matrisi D."""
        mod = torch.tensor([modality], device=hidden.device)
        return self.decision_matrix_batch(hidden.unsqueeze(0), prototypes, mod)[0]


def load_static_tables(artifact: dict, layer_idx: int, n_experts: int = 4) -> tuple[StaticCriteriaTables, torch.Tensor]:
    """artifacts/prototypes.pt içeriğinden tek bir MoE katmanı için tablo ve prototype'ları çıkarır.

    reliability / latency / expert_params ölçülene kadar nötr (eşit) kalır.
    Returns: (StaticCriteriaTables, prototypes (E, H))
    """
    tables = StaticCriteriaTables.neutral(n_experts, artifact["modality_rates"].shape[-1])
    tables.modality_rates = artifact["modality_rates"][layer_idx].clone()
    return tables, artifact["prototypes"][layer_idx].clone()
