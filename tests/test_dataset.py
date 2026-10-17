"""Dataset testleri (Briefing §8): JSON format doğrulaması, split sızıntısı yok, kalite kuralları."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qc_rules  # noqa: E402
from filter_questions import validate_t4  # noqa: E402
from validate_dataset import validate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "decisionmoe_dataset"
TERMS = {"dog": "dog", "person": "person", "man": "person", "tree": "tree"}


@pytest.mark.skipif(not (ROOT / "splits" / "test.json").exists(), reason="dataset henüz üretilmedi")
def test_real_dataset_valid():
    assert validate(ROOT) == []


@pytest.mark.parametrize("qtype,answer,ok", [("T1", "yes", False), ("T2", "Yes.", False), ("T3", "none", False),
                                             ("T1", "Beige", True), ("T4", "no", True)])
def test_forbidden_answers(qtype, answer, ok):
    assert qc_rules.answer_allowed(qtype, answer) is ok


def test_length_rule():
    assert not qc_rules.word_count_ok("What is this?")
    assert qc_rules.word_count_ok("How many people are in this image?")
    assert not qc_rules.word_count_ok("word " * 41)


def test_prompt_example_leak():
    assert qc_rules.leaks_prompt_example("T5", "Does the sign show a price above $10 dollars?")
    assert not qc_rules.leaks_prompt_example("T5", "What word is written on the blue banner?")


def test_t4_negation():
    assert qc_rules.t4_is_negation("None") and qc_rules.t4_is_negation("No, there is not")
    assert not qc_rules.t4_is_negation("Yes")


def test_t4_coco_validation():
    assert validate_t4("Is there a dog in this image?", {"person"}, TERMS) == "ok"
    assert validate_t4("Is there a dog in this image?", {"dog"}, TERMS) == "present"
    assert validate_t4("Is there a dog on the tree?", {"tree"}, TERMS) == "ok"   # bağlam nesnesi sayılmaz
    assert validate_t4("Is there a unicorn here?", {"dog"}, TERMS) == "unverified"
