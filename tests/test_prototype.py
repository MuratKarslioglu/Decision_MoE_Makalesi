"""Prototype hesabı ve cosine kriteri testleri (model gerektirmez)."""
import pytest
import torch

pytest.importorskip("deepspeed")
from src.backbone.prototype import PrototypeAccumulator, cosine_to_prototypes  # noqa: E402


def test_cosine_shape_and_range():
    h, p = torch.randn(10, 16), torch.randn(4, 16)
    c = cosine_to_prototypes(h, p)
    assert c.shape == (10, 4)
    assert c.abs().max() <= 1.0 + 1e-5


def test_cosine_identity_is_one():
    p = torch.randn(4, 16)
    assert torch.allclose(cosine_to_prototypes(p, p).diagonal(), torch.ones(4), atol=1e-5)


def test_zero_prototype_is_nan_free():
    p = torch.zeros(4, 16)
    assert not torch.isnan(cosine_to_prototypes(torch.randn(5, 16), p)).any()


def test_prototypes_are_mean_per_expert():
    acc = object.__new__(PrototypeAccumulator)
    acc.sums = torch.tensor([[[2.0, 4.0], [0.0, 0.0]]])
    acc.counts = torch.tensor([[2, 0]])
    proto = acc.prototypes()
    assert torch.allclose(proto[0, 0], torch.tensor([1.0, 2.0]))
    assert torch.equal(proto[0, 1], torch.zeros(2))   # token düşmeyen expert: sıfır vektör
