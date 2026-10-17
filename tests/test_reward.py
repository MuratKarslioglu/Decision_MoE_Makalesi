"""Reward testleri (Briefing §8): K=0 imkansız, λ etkileri, bilinen input -> bilinen reward."""
import pytest
import torch

from src.rl.reward import compute_reward, flops_ratio, load_imbalance, load_reward_weights

W = load_reward_weights()
Z = torch.tensor(0.0)


def test_lambda_values_from_config():
    assert (W.accuracy, W.flops, W.latency, W.load_imbalance, W.hallucination) == (1.0, 0.1, 0.05, 0.1, 0.2)


def test_known_reward():
    r = compute_reward(torch.tensor(1.0), torch.tensor(0.5), torch.tensor(1.0), torch.tensor(0.2), torch.tensor(1.0), 2, W)
    assert r.item() == pytest.approx(1.0 - 0.05 - 0.05 - 0.02 - 0.2)


def test_perfect_correct_cheap_is_positive():
    assert compute_reward(torch.tensor(1.0), Z, Z, Z, Z, 1, W) > 0


def test_each_penalty_lowers_reward():
    base = compute_reward(torch.tensor(1.0), Z, Z, Z, Z, 2, W)
    one = torch.tensor(1.0)
    for args in [(one, Z, Z, Z), (Z, one, Z, Z), (Z, Z, one, Z), (Z, Z, Z, one)]:
        f, l, i, h = args
        assert compute_reward(torch.tensor(1.0), f, l, i, h, 2, W) < base


@pytest.mark.parametrize("k", [0, -1])
def test_k_zero_impossible(k):
    with pytest.raises(ValueError):
        compute_reward(Z, Z, Z, Z, Z, k, W)


def test_flops_ratio_empty_selection_rejected():
    with pytest.raises(ValueError):
        flops_ratio(torch.empty(0), 1.0)


def test_flops_ratio_value():
    assert flops_ratio(torch.tensor([1.0, 1.0]), 4.0).item() == pytest.approx(0.5)


def test_load_imbalance_uniform_is_zero():
    assert load_imbalance(torch.full((4,), 0.25)).item() == pytest.approx(0.0)
    assert load_imbalance(torch.tensor([1.0, 0, 0, 0])).item() > 0


def test_weights_dataclass_frozen():
    with pytest.raises(Exception):
        W.accuracy = 2.0  # type: ignore[misc]
