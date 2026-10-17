"""metrics.py testleri: bilinen cevaplar için bilinen değerler."""
import pytest

from src.utils.metrics import accuracy, hallucination_rate, is_correct, is_hallucination, normalize_answer


def test_normalize():
    assert normalize_answer("The Three!") == "3"
    assert normalize_answer("  A  Cat. ") == "cat"


@pytest.mark.parametrize("pred,ref,expected", [
    ("Three", "3", True), ("There are three footprints.", "Three", True), ("Right", "right", True),
    ("The left one", "right", False), ("", "x", False), ("cat", "", False), ("category", "cat", False),
])
def test_is_correct(pred, ref, expected):
    assert is_correct(pred, ref) is expected


def test_accuracy_known_value():
    assert accuracy(["3", "no", "red"], ["3", "no", "blue"]) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        accuracy(["a"], ["a", "b"])


def test_hallucination_only_for_t4():
    assert is_hallucination("Yes, there is a cat.", "T4")
    assert not is_hallucination("No.", "T4")
    assert not is_hallucination("No, there is no cat", "T4")
    assert not is_hallucination("Yes", "T1")


def test_hallucination_rate():
    preds, types = ["yes", "no", "yes", "no"], ["T4", "T4", "T1", "T4"]
    assert hallucination_rate(preds, types) == pytest.approx(1 / 3)
    assert hallucination_rate(["yes"], ["T1"]) == 0.0
