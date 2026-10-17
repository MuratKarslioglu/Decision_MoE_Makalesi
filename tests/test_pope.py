"""POPE ayrıştırma ve metrik testleri (veri indirmeden)."""
import pytest

from src.data.pope import parse_yes_no, pope_metrics


@pytest.mark.parametrize("text,expected", [("Yes.", "yes"), ("No, there is none", "no"), ("There is not a cat", "no"),
                                           ("maybe", "other"), ("", "other"), ("Yes, it is", "yes")])
def test_parse_yes_no(text, expected):
    assert parse_yes_no(text) == expected


def test_metrics_known_values():
    preds = ["yes", "yes", "no", "no"]
    labels = ["yes", "no", "yes", "no"]
    m = pope_metrics(preds, labels)
    assert m["accuracy"] == 0.5 and m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5 and m["yes_ratio"] == 0.5


def test_unparsed_counts_as_wrong_not_yes():
    m = pope_metrics(["???", "yes"], ["no", "yes"])
    assert m["accuracy"] == 0.5 and m["unparsed"] == 0.5 and m["yes_ratio"] == 0.5


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        pope_metrics(["yes"], ["yes", "no"])
