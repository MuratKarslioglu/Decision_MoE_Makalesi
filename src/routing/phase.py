"""Reasoning phase tespiti (Briefing §3.2, heuristic).

perception: görüntü token bölgesi | question: '?' içeren soru segmenti |
generation: cevap işaretçisinden (örn. "ASSISTANT:") sonrası | reasoning: geri kalan (varsayılan).
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

PERCEPTION, QUESTION, REASONING, GENERATION = 0, 1, 2, 3


def _find_last_subsequence(ids: torch.Tensor, pattern: Sequence[int]) -> int:
    """pattern'in ids içindeki son başlangıç indeksi; yoksa -1."""
    n, m = ids.numel(), len(pattern)
    pat = torch.tensor(pattern, dtype=ids.dtype, device=ids.device)
    for start in range(n - m, -1, -1):
        if torch.equal(ids[start : start + m], pat):
            return start
    return -1


def detect_phases(
    input_ids: torch.Tensor,            # (T,) long, görüntü yer tutucusu image_token_index
    image_token_index: int,
    n_image_tokens: int,
    question_mark_ids: Sequence[int],   # '?' içeren token id'leri
    answer_marker_ids: Sequence[int],   # cevap başlangıç işaretçisinin id dizisi
) -> torch.Tensor:                      # (T - 1 + n_image_tokens,) long ∈ {0,1,2,3}
    """Genişletilmiş (görüntü token'ları açılmış) dizi için her konumun phase'ini döndürür."""
    pos = (input_ids == image_token_index).nonzero()
    if pos.numel() == 0:
        raise ValueError("input_ids görüntü yer tutucusu içermiyor")
    pos = pos[0].item()
    expanded = torch.cat([
        input_ids[:pos],
        torch.full((n_image_tokens,), image_token_index, dtype=input_ids.dtype, device=input_ids.device),
        input_ids[pos + 1 :],
    ])
    phases = torch.full(expanded.shape, REASONING, dtype=torch.long, device=input_ids.device)
    img_start, img_end = pos, pos + n_image_tokens
    phases[img_start:img_end] = PERCEPTION

    marker = _find_last_subsequence(expanded, answer_marker_ids)
    answer_start = marker if marker >= 0 else expanded.numel()
    phases[answer_start:] = GENERATION

    segment = expanded[img_end:answer_start]
    if segment.numel() and torch.isin(segment, torch.tensor(list(question_mark_ids), device=segment.device)).any():
        phases[img_end:answer_start] = QUESTION
    return phases


def phase_one_hot(phase: int) -> torch.Tensor:
    """p_t ∈ R^4 one-hot."""
    return torch.nn.functional.one_hot(torch.tensor(phase), num_classes=4).float()
