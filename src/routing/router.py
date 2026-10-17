"""DecisionMoE router: backbone'un MoE gate'ini TOPSIS + PPO seçimiyle değiştirir (Tasarım A).

Tek kontrolcü: her adımda (prefill chunk'ı = 1 adım, sonra her üretilen token = 1 adım) policy bir
(w_t, K_t) üretir ve bu çift o adımdaki TÜM MoE katmanlarına uygulanır. Karar matrisi D ise katmana
özeldir (katmanın prototype'ları, modality oranları ve yük geçmişi). Expert çıktıları, backbone'un kendi
gate olasılıklarının seçilen expertler üzerinde yeniden normalize edilmiş hâliyle birleştirilir.

Modlar:
  topk        Standart Top-K baseline (gate olasılığına göre, K sabit; kapasite sınırı yok)
  fixed_mcdm  TOPSIS, sabit w (varsayılan: eşit) ve sabit K
  policy      TOPSIS, w_t ve K_t policy'den (PPO)
  direct      TOPSIS yok. Policy her adımda 4 expert için öncelik vektörü u_t ∈ Δ³ ve K_t üretir;
              seçim/karışım skoru p'_e ∝ p_gate,e · u_e (u uniform ise Standard Top-K ile aynı)

Not (yaklaşıklık): bir forward çağrısındaki tüm token'lar için load/specialization geçmişi çağrı
BAŞINDAKİ değerden okunur; çağrı sonunda geçmiş güncellenir. Batch boyutu 1 varsayılır.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn.functional as F

from src.backbone.prototype import find_moe_layers
from src.routing.criteria import CriteriaTracker, load_criteria_config, load_static_tables
from src.routing.phase import GENERATION, phase_one_hot
from src.routing.topsis import benefit_mask_from_config, compute_topsis_scores, load_topsis_config

TEXT, IMAGE, MIXED = 0, 1, 2


class RoutingMode(str, Enum):
    TOPK = "topk"
    FIXED_MCDM = "fixed_mcdm"
    POLICY = "policy"
    DIRECT = "direct"
    RANDOM = "random"      # duyarlılık testi için: gate'ten bağımsız rastgele K expert


@dataclass
class StepRecord:
    """Tek routing adımının özeti (reward ve analiz için)."""

    k: int
    weights: torch.Tensor            # (7,)
    flops_fraction: float            # K / n_experts
    load_std: float                  # katmanlar üzerinde ortalama Std(l_t)
    n_tokens: int
    wall_time_s: float = 0.0


@dataclass
class Transition:
    """PPO için bir adımın durumu ve aksiyonu."""

    hidden: torch.Tensor             # (H,)
    other: torch.Tensor              # (14,)
    weights: torch.Tensor            # (7,)
    k: int
    log_prob: torch.Tensor
    value: torch.Tensor


@dataclass
class SequenceContext:
    """Prefill chunk'ı için token başına modality ve phase (decode adımlarında tek token, text/generation)."""

    modality: torch.Tensor           # (T,) long
    phases: torch.Tensor             # (T,) long

    @property
    def length(self) -> int:
        return int(self.modality.numel())


class DecisionMoERouter:
    """Modeldeki tüm MoE katmanlarını sarar. close() ile orijinal davranış geri yüklenir."""

    def __init__(
        self,
        model: torch.nn.Module,
        artifact: dict[str, Any],
        mode: RoutingMode | str = RoutingMode.TOPK,
        k: int = 2,
        policy: torch.nn.Module | None = None,
        fixed_weights: torch.Tensor | None = None,
        weight_mask: torch.Tensor | None = None,
        baseline_k: int = 2,
        collect_transitions: bool = False,
        deterministic: bool = False,
    ) -> None:
        self.mode = RoutingMode(mode)
        self.topsis_cfg = load_topsis_config()
        self.crit_cfg = load_criteria_config()
        self.n_experts: int = self.topsis_cfg["n_experts"]
        self.n_criteria: int = self.topsis_cfg["n_criteria"]
        self.k_min, self.k_max = self.topsis_cfg["k_min"], self.topsis_cfg["k_max"]
        if not self.k_min <= k <= self.k_max:
            raise ValueError(f"K={k} {self.k_min}..{self.k_max} dışında")
        self.k, self.baseline_k = k, baseline_k
        self.policy, self.deterministic, self.collect_transitions = policy, deterministic, collect_transitions
        if self.mode in (RoutingMode.POLICY, RoutingMode.DIRECT):
            if policy is None:
                raise ValueError(f"{self.mode.value} modu için policy gerekir")
            width = policy.weight_head.out_features
            expected = self.n_experts if self.mode is RoutingMode.DIRECT else self.n_criteria
            if width != expected:
                raise ValueError(f"{self.mode.value} modu {expected} boyutlu aksiyon bekler, policy {width} üretiyor")
        self.benefit_mask = benefit_mask_from_config(self.topsis_cfg)
        self.fixed_weights = (fixed_weights if fixed_weights is not None
                              else torch.full((self.n_criteria,), 1.0 / self.n_criteria))
        # Ablation: maskelenen kriterlerin ağırlığı 0 yapılır (örn. task_utility yok)
        self.weight_mask = weight_mask

        self.model = model
        self.layers = find_moe_layers(model)
        self.device = next(model.parameters()).device
        self.trackers: list[CriteriaTracker] = []
        self.prototypes: list[torch.Tensor] = []
        for i in range(len(self.layers)):
            tables, protos = load_static_tables(artifact, i, self.n_experts)
            self.trackers.append(CriteriaTracker(
                tables, n_experts=self.n_experts, window=self.crit_cfg["load_window"],
                n_phases=self.crit_cfg["n_phases"], device=self.device))
            self.prototypes.append(protos.to(self.device))

        self.context: SequenceContext | None = None
        self.records: list[StepRecord] = []
        self.transitions: list[Transition] = []
        self.utilization = torch.zeros(len(self.layers), self.n_experts, dtype=torch.long, device=self.device)   # toplam seçim sayısı
        self.k_counts = torch.zeros(self.k_max + 1, dtype=torch.long)
        self._cur_w: torch.Tensor = self.fixed_weights.to(self.device)
        self._cur_k: int = k
        self._step_t0 = 0.0
        self._budget_total, self._budget_spent = 1.0, 0.0
        self._load_std_acc: list[float] = []
        self._pending: StepRecord | None = None
        self._patched: list[torch.nn.Module] = []
        self._patch()

    # ---------------------------------------------------------------- yaşam döngüsü
    def _patch(self) -> None:
        for pos, (_, moe) in enumerate(self.layers):
            layer = moe.deepspeed_moe
            layer.forward = lambda hidden, used_token=None, _pos=pos, _layer=layer: self._layer_forward(_pos, _layer, hidden)
            self._patched.append(layer)

    def close(self) -> None:
        """Orijinal DeepSpeed forward'ını geri yükler."""
        self._finish_step()
        for layer in self._patched:
            if "forward" in layer.__dict__:
                del layer.__dict__["forward"]
        self._patched.clear()

    def begin_episode(self, context: SequenceContext, max_steps: int) -> None:
        """Yeni bir soru: geçmişi sıfırlar, FLOPs bütçesini kurar."""
        self._finish_step()
        for t in self.trackers:
            t.reset()
        self.context = context
        self.records, self.transitions = [], []
        self._budget_total = max(1e-6, max_steps * self.baseline_k / self.n_experts)
        self._budget_spent = 0.0
        self._new_episode = True

    def end_episode(self) -> list[StepRecord]:
        self._finish_step()
        return self.records

    # ---------------------------------------------------------------- adım yönetimi
    def _finish_step(self) -> None:
        if self._pending is not None:
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            rec = self._pending
            rec.wall_time_s = time.perf_counter() - self._step_t0
            rec.load_std = float(torch.stack(self._load_std_acc).mean()) if self._load_std_acc else 0.0
            self.records.append(rec)
            self._pending = None
            self._load_std_acc = []

    def _begin_step(self, x_first: torch.Tensor, n_tokens: int, probs_last: torch.Tensor) -> None:
        """İlk MoE katmanına girişte çağrılır: bu adımın (w_t, K_t) kararını verir."""
        self._finish_step()
        self._step_t0 = time.perf_counter()
        prefill = self.context is not None and n_tokens == self.context.length and n_tokens > 1
        if self.mode in (RoutingMode.POLICY, RoutingMode.DIRECT):
            hidden = x_first[-1].float()
            phase = int(self.context.phases[-1]) if prefill else GENERATION
            modality = MIXED if prefill else TEXT
            entropy = -(probs_last * probs_last.clamp_min(1e-9).log()).sum()
            other = torch.cat([
                F.one_hot(torch.tensor(modality), 3).float().to(self.device),
                entropy.view(1), (1.0 - probs_last.max()).view(1),
                phase_one_hot(phase).to(self.device),
                self.trackers[0].load().to(self.device),
                torch.tensor([max(0.0, 1.0 - self._budget_spent / self._budget_total)], device=self.device),
            ])
            with torch.no_grad() if not self.collect_transitions else torch.enable_grad():
                out = self.policy(hidden.unsqueeze(0), other.unsqueeze(0), deterministic=self.deterministic)
            self._cur_w, self._cur_k = out.weights[0].detach(), int(out.k[0])
            if self.collect_transitions:
                self.transitions.append(Transition(
                    hidden.detach(), other.detach(), out.weights[0].detach(), self._cur_k,
                    out.log_prob[0].detach(), out.value[0].detach()))
        else:
            self._cur_w, self._cur_k = self.fixed_weights.to(self.device), self.k
        w = self._cur_w
        if self.weight_mask is not None and self.mode is not RoutingMode.DIRECT:
            w = w * self.weight_mask.to(w.device)
            w = w / w.sum().clamp_min(1e-8)
        self._eff_w = w
        frac = self._cur_k / self.n_experts
        self._budget_spent += frac
        self.k_counts[self._cur_k] += 1
        self._pending = StepRecord(self._cur_k, self._cur_w.detach().cpu(), frac, 0.0, n_tokens)

    # ---------------------------------------------------------------- katman
    def _token_context(self, s: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(modality (S,), phases (S,)) — prefill'de bağlamdan, decode'da text/generation."""
        if self.context is not None and s == self.context.length:
            return self.context.modality.to(self.device), self.context.phases.to(self.device)
        return (torch.full((s,), TEXT, dtype=torch.long, device=self.device),
                torch.full((s,), GENERATION, dtype=torch.long, device=self.device))

    def _layer_forward(self, pos: int, layer: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[0] != 1:
            raise NotImplementedError("DecisionMoERouter batch boyutu 1 varsayar")
        shape = hidden.shape
        x = hidden.reshape(-1, shape[-1])                                  # (S, H)
        s = x.shape[0]
        wg = layer.gate.wg
        probs = F.softmax(wg(x.to(wg.weight.dtype)).float(), dim=-1)       # (S, E) backbone gate olasılıkları
        if pos == 0:
            self._begin_step(x, s, probs[-1])
        k = self._cur_k
        modality, phases = self._token_context(s)

        mix_probs = probs
        if self.mode is RoutingMode.TOPK:
            idx = probs.topk(k, dim=-1).indices
        elif self.mode is RoutingMode.RANDOM:
            idx = torch.rand_like(probs).topk(k, dim=-1).indices
        elif self.mode is RoutingMode.DIRECT:
            mix_probs = probs * (self.n_experts * self._eff_w).unsqueeze(0)                  # u uniform -> değişmez
            mix_probs = mix_probs / mix_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            idx = mix_probs.topk(k, dim=-1).indices
        else:
            tracker = self.trackers[pos]
            d = tracker.decision_matrix_batch(x, self.prototypes[pos], modality)             # (S, E, 7)
            w = self._eff_w.unsqueeze(0).expand(s, -1)
            scores = compute_topsis_scores(d, w, self.benefit_mask.to(x.device))
            idx = scores.topk(k, dim=-1).indices                                              # (S, K)
        sel = mix_probs.gather(1, idx)
        comb = sel / sel.sum(dim=-1, keepdim=True).clamp_min(1e-9)                           # (S, K)

        out = torch.zeros_like(x)
        experts = layer.experts.deepspeed_experts
        for e in range(self.n_experts):
            hit = (idx == e)                                                                 # (S, K)
            rows = hit.any(dim=-1).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            y = experts[e](x[rows])
            y = y[0] if isinstance(y, tuple) else y
            out.index_add_(0, rows, y * (comb[rows] * hit[rows]).sum(-1, keepdim=True).to(y.dtype))

        tracker = self.trackers[pos]
        tracker.update_batch(idx, phases)
        self.utilization[pos] += torch.bincount(idx.flatten(), minlength=self.n_experts)
        self._load_std_acc.append(tracker.load().std())          # tensör: adım sonunda tek senkronizasyon
        layer.l_aux = torch.zeros((), device=x.device)
        layer.exp_counts = torch.bincount(idx.flatten(), minlength=self.n_experts)
        return out.reshape(shape)

    # ---------------------------------------------------------------- raporlama
    def utilization_fraction(self) -> torch.Tensor:
        """(L, E) her katmanda seçimlerin expertlere payı (T3 tablosu için)."""
        return self.utilization.float() / self.utilization.sum(dim=1, keepdim=True).clamp(min=1)
