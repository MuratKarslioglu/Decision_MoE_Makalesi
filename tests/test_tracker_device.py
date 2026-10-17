"""Cihazda tutulan CriteriaTracker, eski (deque tabanlı, CPU) referans uygulamayla birebir aynı durumu üretmeli."""
import math
from collections import deque

import pytest
import torch
import torch.nn.functional as F

from src.routing.criteria import CriteriaTracker, StaticCriteriaTables

E, P = 4, 4


class ReferenceTracker:
    """Orijinal (GPU'ya taşıma öncesi) uygulamanın birebir kopyası."""

    def __init__(self, window):
        self.window = window
        self.recent = deque(maxlen=window)
        self.pc = torch.zeros(E, P)

    def update(self, selected, phase):
        onehot = torch.zeros(E)
        onehot[selected.cpu()] = 1.0
        self.recent.append(onehot)
        self.pc[selected.cpu(), phase] += 1.0

    def update_batch(self, selected, phases):
        sel, ph = selected.cpu(), phases.cpu()
        onehot = torch.zeros(sel.shape[0], E).scatter_(1, sel, 1.0)
        self.recent.extend(onehot[-self.window:].unbind(0))
        self.pc += onehot.T @ F.one_hot(ph, num_classes=P).float()

    def load(self):
        return torch.zeros(E) if not self.recent else torch.stack(tuple(self.recent)).mean(0)

    def specialization(self):
        p = F.normalize(self.pc + 1.0, p=1, dim=-1)
        return 1.0 - (-(p * p.log()).sum(-1) / math.log(P))


def devices():
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("window", [4, 100])
def test_state_matches_reference_over_random_sequence(device, window):
    g = torch.Generator().manual_seed(0)
    ref = ReferenceTracker(window)
    new = CriteriaTracker(StaticCriteriaTables.neutral(E, 3), n_experts=E, window=window, device=device)
    for step in range(60):
        if step % 3 == 0:     # chunk (prefill benzeri; pencereden büyük olabilir)
            s = int(torch.randint(1, 150, (1,), generator=g))
            sel = torch.stack([torch.randperm(E, generator=g)[:2] for _ in range(s)])
            ph = torch.randint(0, P, (s,), generator=g)
            ref.update_batch(sel, ph); new.update_batch(sel.to(device), ph.to(device))
        else:
            sel = torch.randperm(E, generator=g)[: int(torch.randint(1, 5, (1,), generator=g))]
            ph = int(torch.randint(0, P, (1,), generator=g))
            ref.update(sel, ph); new.update(sel.to(device), ph)
        assert torch.allclose(new.load().cpu(), ref.load(), atol=1e-6), step
        assert torch.allclose(new.specialization().cpu(), ref.specialization(), atol=1e-6), step


@pytest.mark.parametrize("device", devices())
def test_reset_and_tables_live_on_device(device):
    t = CriteriaTracker(StaticCriteriaTables.neutral(E, 3), n_experts=E, window=5, device=device)
    assert t.tables.modality_rates.device.type == torch.device(device).type
    t.update(torch.tensor([0, 1]), 0)
    t.reset()
    assert t.load().abs().sum() == 0 and t.specialization().abs().max() < 1e-6
    t.update(torch.tensor([2]), 1)
    assert torch.allclose(t.load().cpu(), torch.tensor([0.0, 0.0, 1.0, 0.0]))


@pytest.mark.parametrize("device", devices())
def test_decision_matrix_identical_across_devices(device):
    torch.manual_seed(1)
    sel = torch.tensor([[0, 1], [2, 3], [0, 2]]); ph = torch.tensor([0, 1, 2])
    h, protos, mod = torch.randn(5, 32), torch.randn(E, 32), torch.tensor([0, 1, 2, 1, 0])
    a = CriteriaTracker(StaticCriteriaTables.neutral(E, 3), n_experts=E, device="cpu")
    b = CriteriaTracker(StaticCriteriaTables.neutral(E, 3), n_experts=E, device=device)
    a.update_batch(sel, ph); b.update_batch(sel.to(device), ph.to(device))
    da = a.decision_matrix_batch(h, protos, mod)
    db = b.decision_matrix_batch(h.to(device), protos.to(device), mod.to(device))
    assert torch.allclose(da, db.cpu(), atol=1e-5)
