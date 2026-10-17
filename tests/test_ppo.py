"""policy.py testleri: simplex, K aralığı, shape, log_prob tutarlılığı."""
import pytest
import torch

from src.rl.policy import RoutingPolicy, load_policy_config, state_dim_from_config

B = 16


@pytest.fixture(scope="module")
def policy() -> RoutingPolicy:
    torch.manual_seed(0)
    return RoutingPolicy()


def batch(policy: RoutingPolicy):
    return torch.randn(B, policy.hidden_dim), torch.randn(B, policy.n_other)


def test_state_dim_is_sum_of_components():
    assert state_dim_from_config(load_policy_config()) == 64 + 3 + 1 + 1 + 4 + 4 + 1 == 78


def test_weights_on_simplex(policy):
    out = policy(*batch(policy))
    assert out.weights.shape == (B, 7)
    assert torch.allclose(out.weights.sum(-1), torch.ones(B), atol=1e-6)
    assert (out.weights > 0).all()


def test_k_in_valid_range_and_never_zero(policy):
    for _ in range(20):
        k = policy(*batch(policy)).k
        assert ((k >= 1) & (k <= 4)).all()


def test_output_shapes(policy):
    out = policy(*batch(policy))
    for t in (out.k, out.log_prob, out.entropy, out.value):
        assert t.shape == (B,)
    assert torch.isfinite(out.log_prob).all()


def test_log_prob_of_taken_action_matches(policy):
    h, o = batch(policy)
    out = policy(h, o)
    again = policy(h, o, action=(out.weights, out.k))
    assert torch.allclose(out.log_prob, again.log_prob, atol=1e-4)


def test_fixed_k_alternative_a():
    p = RoutingPolicy(fixed_k=2)
    assert (p(torch.randn(B, p.hidden_dim), torch.randn(B, p.n_other)).k == 2).all()


def test_deterministic_is_repeatable(policy):
    h, o = batch(policy)
    a, b = policy(h, o, deterministic=True), policy(h, o, deterministic=True)
    assert torch.equal(a.k, b.k) and torch.allclose(a.weights, b.weights)


def test_gradients_flow_through_projection(policy):
    h, o = batch(policy)
    out = policy(h, o)
    (out.log_prob.sum() + out.value.sum()).backward()
    assert policy.projection.weight.grad is not None


# ---------------------------------------------------------------- ppo.py
from src.rl.ppo import PPOTrainer, build_batch, compute_gae, episode_rewards  # noqa: E402
from src.rl.reward import load_reward_weights  # noqa: E402
from src.routing.router import StepRecord, Transition  # noqa: E402


def test_gae_known_values():
    # gamma=1, lam=1 -> advantage = toplam gelecek ödül - V
    adv, ret = compute_gae(torch.tensor([1.0, 0.0, 2.0]), torch.tensor([0.5, 0.5, 0.5]), gamma=1.0, lam=1.0)
    assert torch.allclose(ret, torch.tensor([3.0, 2.0, 2.0]))
    assert torch.allclose(adv, ret - torch.tensor([0.5, 0.5, 0.5]))


def test_gae_lambda_zero_is_td_error():
    adv, _ = compute_gae(torch.tensor([1.0, 2.0]), torch.tensor([0.5, 1.0]), gamma=0.9, lam=0.0)
    assert torch.allclose(adv, torch.tensor([1.0 + 0.9 * 1.0 - 0.5, 2.0 - 1.0]))


def make_records(ks, load_std=0.0, wall=1.0):
    return [StepRecord(k, torch.full((7,), 1 / 7), k / 4, load_std, 1, wall) for k in ks]


def test_episode_rewards_terminal_terms_only_on_last_step():
    w = load_reward_weights()
    r = episode_rewards(make_records([2, 2, 2]), accuracy=1.0, hallucinated=True, weights=w, baseline_step_time_s=1.0)
    step_cost = w.flops * 0.5 / 3 + w.latency * 1.0       # eşit token'lı 3 adım: FLOPs payı 1/3'er
    assert r[0].item() == pytest.approx(-step_cost) and r[1].item() == pytest.approx(-step_cost)
    assert r[2].item() == pytest.approx(w.accuracy - w.hallucination - step_cost)


def test_more_experts_cost_more_reward():
    w = load_reward_weights()
    cheap = episode_rewards(make_records([1]), 1.0, False, w, 1.0)
    dense = episode_rewards(make_records([4]), 1.0, False, w, 1.0)
    assert cheap.item() > dense.item()


def test_k0_reward_is_rejected():
    with pytest.raises(ValueError):
        episode_rewards(make_records([0]), 1.0, False, load_reward_weights(), 1.0)


def test_ppo_learns_toy_bandit():
    """Ödül yalnızca K=3'te +1: eğitimden sonra ortalama K=3'e yaklaşmalı ve K hiç 0 olmamalı."""
    torch.manual_seed(0)
    policy = RoutingPolicy()
    cfg = load_policy_config()
    cfg["ppo"].update(epochs=4, minibatch_size=64, lr=3e-3)
    trainer = PPOTrainer(policy, cfg)
    hidden, other = torch.zeros(1, policy.hidden_dim), torch.zeros(1, policy.n_other)
    def mean_prob_k3():
        with torch.no_grad():
            return policy._distributions(policy.encode(hidden, other))[1].probs[0, 2].item()
    before = mean_prob_k3()
    for _ in range(40):
        episodes = []
        for _ in range(32):
            with torch.no_grad():
                o = policy(hidden, other)
            tr = [Transition(hidden[0], other[0], o.weights[0], int(o.k[0]), o.log_prob[0], o.value[0])]
            episodes.append((tr, torch.tensor([1.0 if int(o.k[0]) == 3 else 0.0])))
        trainer.update(build_batch(episodes, gamma=1.0, lam=0.95))
    assert mean_prob_k3() > max(0.8, before + 0.3)


def test_latency_baseline_separate_for_prefill_and_decode():
    w = load_reward_weights()
    recs = [StepRecord(2, torch.full((7,), 1 / 7), 0.5, 0.0, 500, 0.5), StepRecord(2, torch.full((7,), 1 / 7), 0.5, 0.0, 1, 0.05)]
    r = episode_rewards(recs, 0.0, False, w, {"prefill": 0.5, "decode": 0.05})
    total = 500 + 1
    expected_pre = -(w.flops * 0.5 * 500 / total + w.latency * 1.0)   # FLOPs terimi token ağırlıklı
    expected_dec = -(w.flops * 0.5 * 1 / total + w.latency * 1.0)
    assert r[0].item() == pytest.approx(expected_pre) and r[1].item() == pytest.approx(expected_dec)


# ---------------------------------------------------------------- DIRECT (sadece PPO) kolu: 4 boyutlu aksiyon
def test_direct_policy_outputs_expert_simplex():
    p = RoutingPolicy(n_criteria=4, neutral_init_concentration=10.0)
    out = p(torch.randn(B, p.hidden_dim), torch.randn(B, p.n_other))
    assert out.weights.shape == (B, 4)
    assert torch.allclose(out.weights.sum(-1), torch.ones(B), atol=1e-6)
    assert (out.weights > 0).all() and ((out.k >= 1) & (out.k <= 4)).all()


def test_direct_policy_neutral_init_is_uniform_in_expectation():
    p = RoutingPolicy(n_criteria=4, neutral_init_concentration=10.0)
    h, o = torch.randn(B, p.hidden_dim), torch.randn(B, p.n_other)
    mean = p(h, o, deterministic=True).weights
    assert torch.allclose(mean, torch.full((B, 4), 0.25), atol=1e-4)      # girdiden bağımsız uniform


def test_neutral_init_default_is_off_for_mcdm_policy():
    p = RoutingPolicy()                                                   # 7 kriterli kol etkilenmemeli
    assert p.weight_head.weight.abs().sum() > 0


def test_ppo_learns_to_prefer_one_expert_prior():
    """Ödül yalnızca u'nun expert 2 payı >= 0.4 olduğunda: PPO u_2'yi yükseltmeli."""
    torch.manual_seed(0)
    policy = RoutingPolicy(n_criteria=4, neutral_init_concentration=10.0, fixed_k=2)
    cfg = load_policy_config()
    cfg["ppo"].update(epochs=4, minibatch_size=64, lr=3e-3)
    trainer = PPOTrainer(policy, cfg)
    hidden, other = torch.zeros(1, policy.hidden_dim), torch.zeros(1, policy.n_other)
    def mean_u2():
        with torch.no_grad():
            return policy(hidden, other, deterministic=True).weights[0, 2].item()
    before = mean_u2()
    for _ in range(60):
        episodes = []
        for _ in range(32):
            with torch.no_grad():
                o = policy(hidden, other)
            tr = [Transition(hidden[0], other[0], o.weights[0], int(o.k[0]), o.log_prob[0], o.value[0])]
            episodes.append((tr, torch.tensor([1.0 if o.weights[0, 2] >= 0.4 else 0.0])))
        trainer.update(build_batch(episodes, gamma=1.0, lam=0.95))
    assert before == pytest.approx(0.25, abs=1e-4) and mean_u2() > 0.35


def test_flops_penalty_is_token_weighted_prefill_dominates():
    """~600 token'lık prefill'de K=4 seçmek, decode'da K=1 seçmekten çok daha pahalı olmalı (reward hacking koruması)."""
    w = load_reward_weights()
    def rec(k, ntok):
        return StepRecord(k, torch.full((7,), 1 / 7), k / 4, 0.0, ntok, 0.0)
    decode = [rec(1, 1) for _ in range(10)]
    prefill_k4 = episode_rewards([rec(4, 600)] + decode, 0.0, False, w, {"prefill": 1.0, "decode": 1.0}).sum()
    prefill_k1 = episode_rewards([rec(1, 600)] + decode, 0.0, False, w, {"prefill": 1.0, "decode": 1.0}).sum()
    assert prefill_k1.item() - prefill_k4.item() > 0.9 * w.flops * 0.75       # ~0.75 * λ2 kazanç
    # decode adımlarında K=4 ise fark küçük kalır
    decode_k4 = [rec(4, 1) for _ in range(10)]
    cheap_prefill_dense_decode = episode_rewards([rec(1, 600)] + decode_k4, 0.0, False, w, {"prefill": 1.0, "decode": 1.0}).sum()
    assert prefill_k1.item() - cheap_prefill_dense_decode.item() < 0.05 * w.flops


def test_episode_flops_terms_sum_to_token_weighted_fraction():
    w = load_reward_weights()
    recs = [StepRecord(4, torch.full((7,), 1 / 7), 1.0, 0.0, 600, 0.0)] + [StepRecord(1, torch.full((7,), 1 / 7), 0.25, 0.0, 1, 0.0) for _ in range(10)]
    zero_lat = type(w)(accuracy=0.0, flops=1.0, latency=0.0, load_imbalance=0.0, hallucination=0.0)
    total = -episode_rewards(recs, 0.0, False, zero_lat, 1.0).sum().item()
    assert total == pytest.approx((600 * 1.0 + 10 * 0.25) / 610, abs=1e-5)
