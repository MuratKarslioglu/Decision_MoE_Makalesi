"""router.py testleri: küçük bir DeepSpeed MoE ile (backbone gerektirmez; yalnızca venv_moe'da çalışır)."""
import os

import pytest
import torch
import torch.nn as nn

pytest.importorskip("deepspeed")
import torch.distributed as dist  # noqa: E402
from deepspeed.moe.layer import MoE  # noqa: E402

from src.rl.policy import RoutingPolicy  # noqa: E402
from src.routing.phase import GENERATION, PERCEPTION, QUESTION  # noqa: E402
from src.routing.router import DecisionMoERouter, RoutingMode, SequenceContext  # noqa: E402

H, E, L, FF = 2560, 4, 2, 16


class TinyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(H, FF), nn.Linear(FF, H)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.moes = nn.ModuleList([
            MoE(H, expert=TinyExpert(), num_experts=E, ep_size=1, k=2, capacity_factor=4.0,
                eval_capacity_factor=4.0, min_capacity=0, use_residual=False)
            for _ in range(L)])
        for m in self.moes:
            m.deepspeed_moe.gate.top2_2nd_expert_sampling = False   # deterministik karşılaştırma

    def forward(self, x):
        for m in self.moes:
            x = x + m(x)[0]
        return x


@pytest.fixture(scope="module", autouse=True)
def _dist():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29533")
        dist.init_process_group("gloo", rank=0, world_size=1)


@pytest.fixture()
def model() -> TinyModel:
    torch.manual_seed(0)
    return TinyModel().eval()


@pytest.fixture()
def artifact() -> dict:
    torch.manual_seed(1)
    rates = torch.rand(L, E, 3)
    return {"prototypes": torch.randn(L, E, H), "modality_rates": rates / rates.sum(1, keepdim=True)}


def ctx(s: int) -> SequenceContext:
    return SequenceContext(modality=torch.tensor([1] * (s - 2) + [0, 0]),
                           phases=torch.tensor([PERCEPTION] * (s - 2) + [QUESTION, GENERATION]))


def test_topk_matches_original_deepspeed_top2(model):
    x = torch.randn(1, 9, H)
    with torch.no_grad():
        ref = model(x)
        router = DecisionMoERouter(model, {"prototypes": torch.zeros(L, E, H), "modality_rates": torch.ones(L, E, 3) / E},
                                   mode="topk", k=2)
        out = model(x)
        router.close()
        restored = model(x)
    assert torch.allclose(out, ref, atol=1e-4)
    assert torch.allclose(restored, ref, atol=1e-4)           # close() orijinal davranışı geri getirir


def test_fixed_mcdm_selects_k_experts_and_records(model, artifact):
    router = DecisionMoERouter(model, artifact, mode=RoutingMode.FIXED_MCDM, k=3)
    router.begin_episode(ctx(9), max_steps=4)
    with torch.no_grad():
        model(torch.randn(1, 9, H))
        model(torch.randn(1, 1, H))
    recs = router.end_episode()
    assert len(recs) == 2 and all(r.k == 3 and r.flops_fraction == 0.75 for r in recs)
    assert router.utilization.sum().item() == (9 + 1) * 3 * L        # token x K x katman
    assert torch.allclose(router.utilization_fraction().sum(1), torch.ones(L))
    router.close()


def test_k1_and_k4_produce_finite_outputs(model, artifact):
    for k in (1, 4):
        router = DecisionMoERouter(model, artifact, mode="fixed_mcdm", k=k)
        with torch.no_grad():
            assert torch.isfinite(model(torch.randn(1, 5, H))).all()
        router.close()


def test_k4_equals_dense_mixture_independent_of_selection(model, artifact):
    """K=4'te tüm expertler seçilir: sonuç TOPSIS skorlarından bağımsız olmalı."""
    x = torch.randn(1, 6, H)
    outs = []
    for w in (torch.full((7,), 1 / 7), torch.tensor([0.9, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0])):
        router = DecisionMoERouter(model, artifact, mode="fixed_mcdm", k=4, fixed_weights=w)
        with torch.no_grad():
            outs.append(model(x))
        router.close()
    assert torch.allclose(outs[0], outs[1], atol=1e-5)


def test_policy_mode_applies_policy_action_and_collects_transitions(model, artifact):
    torch.manual_seed(2)
    policy = RoutingPolicy()
    router = DecisionMoERouter(model, artifact, mode="policy", policy=policy, collect_transitions=True)
    router.begin_episode(ctx(9), max_steps=3)
    with torch.no_grad():
        model(torch.randn(1, 9, H))
        model(torch.randn(1, 1, H))
        model(torch.randn(1, 1, H))
    recs = router.end_episode()
    assert len(recs) == 3 and len(router.transitions) == 3
    for r, t in zip(recs, router.transitions):
        assert 1 <= r.k <= 4 and r.k == t.k
        assert torch.allclose(r.weights.sum(), torch.tensor(1.0), atol=1e-5)
        assert t.other.shape == (14,) and t.hidden.shape == (H,)
    router.close()


def test_weight_mask_zeroes_masked_criterion(model, artifact):
    mask = torch.tensor([0.0, 1, 1, 1, 1, 1, 1])
    router = DecisionMoERouter(model, artifact, mode="fixed_mcdm", k=2, weight_mask=mask)
    with torch.no_grad():
        model(torch.randn(1, 4, H))
    assert router._eff_w[0].item() == 0.0 and abs(router._eff_w.sum().item() - 1.0) < 1e-6
    router.close()


def test_batch_size_gt1_rejected(model, artifact):
    router = DecisionMoERouter(model, artifact, mode="topk", k=2)
    with pytest.raises(NotImplementedError):
        model(torch.randn(2, 4, H))
    router.close()


# ---------------------------------------------------------------- DIRECT (sadece PPO) modu
class FixedPriorPolicy(nn.Module):
    """Sabit u_t ve K döndüren sahte policy (router'ı izole test etmek için)."""

    def __init__(self, u, k):
        super().__init__()
        self.u, self.k = torch.as_tensor(u, dtype=torch.float32), k
        self.weight_head = nn.Linear(1, len(u))                 # router'ın boyut doğrulaması için
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, hidden, other, action=None, deterministic=False):
        from src.rl.policy import PolicyOutput
        b = hidden.shape[0]
        z = torch.zeros(b)
        return PolicyOutput(weights=self.u.expand(b, -1), k=torch.full((b,), self.k), log_prob=z, entropy=z, value=z)


def test_direct_with_uniform_prior_equals_topk(model, artifact):
    x = torch.randn(1, 9, H)
    with torch.no_grad():
        r1 = DecisionMoERouter(model, artifact, mode="topk", k=2)
        ref = model(x)
        r1.close()
        r2 = DecisionMoERouter(model, artifact, mode="direct", policy=FixedPriorPolicy([0.25] * 4, 2))
        out = model(x)
        r2.close()
    assert torch.allclose(out, ref, atol=1e-4)


def test_direct_concentrated_prior_with_k1_routes_all_tokens_to_that_expert(model, artifact):
    router = DecisionMoERouter(model, artifact, mode="direct", policy=FixedPriorPolicy([1e-6, 1e-6, 1.0, 1e-6], 1))
    with torch.no_grad():
        model(torch.randn(1, 7, H))
    frac = router.utilization_fraction()
    assert torch.allclose(frac[:, 2], torch.ones(L)) and frac[:, [0, 1, 3]].sum() == 0
    router.close()


def test_direct_records_four_dim_weights_and_collects_transitions(model, artifact):
    from src.rl.policy import RoutingPolicy
    torch.manual_seed(3)
    policy = RoutingPolicy(n_criteria=4, neutral_init_concentration=10.0)
    router = DecisionMoERouter(model, artifact, mode="direct", policy=policy, collect_transitions=True)
    router.begin_episode(ctx(9), max_steps=3)
    with torch.no_grad():
        model(torch.randn(1, 9, H)); model(torch.randn(1, 1, H))
    recs = router.end_episode()
    assert len(recs) == 2 and len(router.transitions) == 2
    for r in recs:
        assert r.weights.shape == (4,) and abs(r.weights.sum().item() - 1.0) < 1e-5
    router.close()


def test_direct_rejects_wrong_policy_width(model, artifact):
    from src.rl.policy import RoutingPolicy
    with pytest.raises(ValueError):
        DecisionMoERouter(model, artifact, mode="direct", policy=RoutingPolicy())          # 7 boyutlu: yanlış
    with pytest.raises(ValueError):
        DecisionMoERouter(model, artifact, mode="policy", policy=RoutingPolicy(n_criteria=4))


def test_random_mode_selects_k_experts_and_is_not_deterministic_across_tokens(model, artifact):
    torch.manual_seed(5)
    router = DecisionMoERouter(model, artifact, mode="random", k=2)
    with torch.no_grad():
        out = model(torch.randn(1, 40, H))
    assert torch.isfinite(out).all() and router.utilization.sum().item() == 40 * 2 * L
    assert (router.utilization_fraction() > 0.1).all()           # dört expert de kullanılır
    router.close()


def test_fixed_action_policy_pins_expert(model, artifact):
    from src.rl.policy import FixedActionPolicy
    router = DecisionMoERouter(model, artifact, mode="direct", policy=FixedActionPolicy([1e-6, 1.0, 1e-6, 1e-6], 1))
    with torch.no_grad():
        model(torch.randn(1, 6, H))
    assert torch.allclose(router.utilization_fraction()[:, 1], torch.ones(L))
    router.close()
