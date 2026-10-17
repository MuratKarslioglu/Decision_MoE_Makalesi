"""phase.py testleri (sentetik token id'leri)."""
import torch

from src.routing.phase import GENERATION, PERCEPTION, QUESTION, REASONING, detect_phases, phase_one_hot

IMG, N_IMG, QMARK, MARKER = -200, 5, 30, [90, 91]


def run(ids):
    return detect_phases(torch.tensor(ids), IMG, N_IMG, [QMARK], MARKER)


def test_length_and_regions():
    # [USER, :, <img>, soru..., ?, ASSISTANT, :]
    ids = [1, 2, IMG, 10, 11, QMARK, 90, 91]
    ph = run(ids)
    assert ph.numel() == len(ids) - 1 + N_IMG
    assert (ph[2:7] == PERCEPTION).all()
    assert (ph[:2] == REASONING).all()
    assert (ph[7:10] == QUESTION).all()
    assert (ph[10:] == GENERATION).all()


def test_no_question_mark_is_reasoning():
    ph = run([1, IMG, 10, 11, 90, 91])
    assert (ph[1 + N_IMG : -2] == REASONING).all()


def test_generated_tokens_after_marker_are_generation():
    ph = run([1, IMG, 10, QMARK, 90, 91, 50, 51, 52])
    assert (ph[-5:] == GENERATION).all()


def test_missing_marker_has_no_generation():
    assert (run([1, IMG, 10, QMARK]) != GENERATION).all()


def test_one_hot():
    assert torch.equal(phase_one_hot(2), torch.tensor([0.0, 0.0, 1.0, 0.0]))
