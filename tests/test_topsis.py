"""TOPSIS testleri (Briefing §8): L2 norm, skor aralığı, doğru expert, benefit/cost yönü, batch."""
import pytest
import torch

from src.routing.topsis import (
    benefit_mask_from_config,
    compute_topsis_scores,
    load_topsis_config,
    select_top_k,
)

MASK = benefit_mask_from_config(load_topsis_config())
W = torch.full((7,), 1 / 7)


def test_config_mask():
    assert MASK.tolist() == [True, True, True, True, False, False, False]


def test_l2_norm_columns_unit():
    d = torch.rand(4, 7) + 0.1
    r = d / d.pow(2).sum(dim=0, keepdim=True).sqrt()
    assert torch.allclose(r.pow(2).sum(dim=0), torch.ones(7), atol=1e-6)


def test_scores_in_unit_interval():
    s = compute_topsis_scores(torch.rand(4, 7) + 0.1, W, MASK)
    assert s.shape == (4,)
    assert ((s >= 0) & (s <= 1)).all()


def test_best_expert_selected():
    d = torch.full((4, 7), 0.5)
    d[2, :4] = 0.9   # benefit yüksek
    d[2, 4:] = 0.1   # cost düşük
    s = compute_topsis_scores(d, W, MASK)
    assert s.argmax().item() == 2
    assert select_top_k(s, 1).item() == 2


def test_benefit_cost_direction():
    base = torch.full((4, 7), 0.5)
    up_benefit = base.clone(); up_benefit[0, 0] = 0.9   # benefit artışı skoru artırmalı
    up_cost = base.clone(); up_cost[0, 4] = 0.9         # cost artışı skoru azaltmalı
    s0 = compute_topsis_scores(base, W, MASK)[0]
    assert compute_topsis_scores(up_benefit, W, MASK)[0] > s0
    assert compute_topsis_scores(up_cost, W, MASK)[0] < s0


def test_batch_matches_single():
    d = torch.rand(3, 4, 7) + 0.1
    w = torch.softmax(torch.rand(3, 7), dim=-1)
    batch = compute_topsis_scores(d, w, MASK)
    for i in range(3):
        assert torch.allclose(batch[i], compute_topsis_scores(d[i], w[i], MASK), atol=1e-6)


def test_identical_experts_no_nan():
    s = compute_topsis_scores(torch.full((4, 7), 0.5), W, MASK)
    assert torch.isfinite(s).all()
    assert torch.allclose(s, torch.full((4,), 0.5))


@pytest.mark.parametrize("k", [0, 5])
def test_invalid_k_raises(k):
    with pytest.raises(ValueError):
        select_top_k(torch.rand(4), k)


def test_top_k_shape():
    assert select_top_k(torch.rand(2, 4), 3).shape == (2, 3)
