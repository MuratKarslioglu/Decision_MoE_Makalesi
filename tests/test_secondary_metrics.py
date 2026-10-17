"""secondary_metrics testleri (model gerektirmeyen kısım)."""
import pytest

from src.utils.secondary_metrics import token_f1


def test_f1_exact_and_disjoint():
    assert token_f1("Three", "3") == 1.0
    assert token_f1("cat", "dog") == 0.0


def test_f1_partial_known_value():
    # pred 2 kelime, ref 3 kelime, 2 ortak -> P=1, R=2/3 -> F1=0.8
    assert token_f1("rigid footprint", "the rigid footprint left") == pytest.approx(2 * 1 * (2 / 3) / (1 + 2 / 3), abs=1e-6)


def test_f1_empty_cases():
    assert token_f1("", "") == 1.0
    assert token_f1("", "x") == 0.0
