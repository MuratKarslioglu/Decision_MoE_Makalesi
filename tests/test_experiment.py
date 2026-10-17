"""Variant yükleme ve policy/router fabrikası testleri (backbone gerekmez)."""
import pytest
import torch

from src.utils.experiment import build_policy, load_variant
from src.utils.logger import routing_metrics

ALL = ["standard_topk", "fixed_mcdm_no_utility", "fixed_mcdm_cosine", "ppo_fixedk_no_utility",
       "ppo_fixedk_cosine", "decisionmoe_a", "decisionmoe_b", "ppo_direct_fixedk", "ppo_direct_dynk"]


@pytest.mark.parametrize("name", ALL)
def test_all_variants_load_with_valid_mode(name):
    v = load_variant(name)
    assert v["name"] == name and v["mode"] in {"topk", "fixed_mcdm", "policy", "direct"}


def test_policy_widths_per_mode():
    cpu = torch.device("cpu")
    assert build_policy(load_variant("standard_topk"), cpu) is None
    assert build_policy(load_variant("fixed_mcdm_cosine"), cpu) is None
    assert build_policy(load_variant("decisionmoe_b"), cpu).weight_head.out_features == 7
    assert build_policy(load_variant("ppo_direct_dynk"), cpu).weight_head.out_features == 4


def test_direct_fixed_and_dynamic_k():
    assert build_policy(load_variant("ppo_direct_fixedk"), torch.device("cpu")).fixed_k == 2
    assert build_policy(load_variant("ppo_direct_dynk"), torch.device("cpu")).fixed_k is None


def test_routing_metrics_keys_for_both_action_widths():
    load = torch.full((4,), 0.25)
    m7 = routing_metrics(torch.full((7,), 1 / 7), 2.0, load)
    m4 = routing_metrics(torch.full((4,), 0.25), 2.0, load)
    assert "routing/w_task_utility" in m7 and "routing/w_load_balance" in m7
    assert "routing/u_expert_0" in m4 and "routing/u_expert_3" in m4 and "routing/w_task_utility" not in m4
    assert m4["routing/expert_3_load"] == 0.25


@pytest.mark.parametrize("name,lam", [("decisionmoe_b_l03", 0.3), ("decisionmoe_b_l30", 3.0)])
def test_lambda_sweep_variants(name, lam):
    v = load_variant(name)
    assert v["mode"] == "policy" and v["fixed_k"] is None and v["lambda_flops"] == lam
