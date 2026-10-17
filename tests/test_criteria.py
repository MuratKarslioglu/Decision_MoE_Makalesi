"""criteria.py testleri: tip/aralık, sütun sırası, online güncelleme tutarlılığı."""
import pytest
import torch

from src.routing.criteria import CRITERIA_NAMES, CriteriaTracker, StaticCriteriaTables, load_criteria_config

E, H = 4, 32


def make_tracker(window: int = 100) -> CriteriaTracker:
    return CriteriaTracker(StaticCriteriaTables.neutral(E, 3), n_experts=E, window=window)


def test_decision_matrix_shape_type_range():
    t = make_tracker()
    t.update(torch.tensor([0, 1]), 0)
    D = t.decision_matrix(torch.randn(H), torch.randn(E, H), modality=1)
    assert isinstance(D, torch.Tensor) and D.shape == (E, 7)
    assert torch.isfinite(D).all()
    assert (D >= 0).all()                    # TOPSIS L2 normalizasyonu için negatif yok
    assert (D[:, [0, 1, 2, 3, 6]] <= 1.0).all()


def test_column_order_matches_topsis_yaml():
    names = [c["name"] for c in load_criteria_config()["criteria"]]
    assert tuple(names) == CRITERIA_NAMES


def test_task_utility_prefers_matching_prototype():
    t = make_tracker()
    protos = torch.randn(E, H)
    D = t.decision_matrix(protos[2].clone(), protos, modality=0)
    assert D[:, 0].argmax().item() == 2
    assert D[2, 0].item() == pytest.approx(1.0, abs=1e-5)


def test_load_is_window_utilization():
    t = make_tracker(window=4)
    for sel in ([0], [0], [0, 1], [2]):
        t.update(torch.tensor(sel), 0)
    assert torch.allclose(t.load(), torch.tensor([0.75, 0.25, 0.25, 0.0]))
    t.update(torch.tensor([3]), 0)           # pencere kayar: ilk [0] düşer
    assert torch.allclose(t.load(), torch.tensor([0.5, 0.25, 0.25, 0.25]))


def test_specialization_rises_with_single_phase_use():
    t = make_tracker()
    assert torch.allclose(t.specialization(), torch.zeros(E), atol=1e-6)     # geçmiş yok: uniform
    for _ in range(50):
        t.update(torch.tensor([0]), 2)       # expert 0 hep reasoning
        t.update(torch.tensor([1]), 0)
        t.update(torch.tensor([1]), 3)       # expert 1 iki phase'e dağılır
    s = t.specialization()
    assert s[0] > s[1] > s[2]
    assert ((s >= 0) & (s <= 1)).all()


def test_flops_scales_with_load_and_params():
    tables = StaticCriteriaTables.neutral(E, 3)
    tables.expert_params = torch.tensor([1.0, 1.0, 2.0, 0.0])
    t = CriteriaTracker(tables, n_experts=E)
    for _ in range(10):
        t.update(torch.tensor([0, 2]), 0)
    D = t.decision_matrix(torch.randn(H), torch.randn(E, H), 0)
    assert D[2, 5] > D[0, 5] > D[1, 5] == D[3, 5] == 0.0


def test_reset_clears_history():
    t = make_tracker()
    t.update(torch.tensor([1]), 1)
    t.reset()
    assert t.load().sum() == 0 and t.specialization().abs().max() < 1e-6


def test_modality_match_uses_table_column():
    tables = StaticCriteriaTables.neutral(E, 3)
    tables.modality_rates[:, 1] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    t = CriteriaTracker(tables, n_experts=E)
    D = t.decision_matrix(torch.randn(H), torch.randn(E, H), modality=1)
    assert torch.allclose(D[:, 1], torch.tensor([0.1, 0.2, 0.3, 0.4]))


def test_batch_matches_single_and_update_batch_matches_update():
    torch.manual_seed(0)
    tables = StaticCriteriaTables.neutral(E, 3)
    a, b = CriteriaTracker(tables, n_experts=E, window=6), CriteriaTracker(tables, n_experts=E, window=6)
    sel = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3], [0, 1], [2, 3], [0, 3], [1, 2]])
    ph = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    a.update_batch(sel, ph)
    for s_, p_ in zip(sel[-6:], ph[-6:]):          # b: yalnızca pencere içi token'lar için tek tek
        b.update(s_, int(p_))
    assert torch.allclose(a.load(), b.load())
    hid, proto, mod = torch.randn(5, H), torch.randn(E, H), torch.tensor([0, 1, 2, 1, 0])
    batch = a.decision_matrix_batch(hid, proto, mod)
    assert batch.shape == (5, E, 7)
    for i in range(5):
        assert torch.allclose(batch[i], a.decision_matrix(hid[i], proto, int(mod[i])), atol=1e-6)
