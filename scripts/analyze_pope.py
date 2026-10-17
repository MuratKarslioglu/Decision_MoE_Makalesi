"""POPE sonuçları için eşleştirilmiş bootstrap farkları (standard_topk ve standard_topk_k1'e göre) → pope_summary.json.

Çalıştırma: python scripts/analyze_pope.py (venv). Girdi: experiments/results/pope_*.json (soru bazlı `items` içerir).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[1] / "experiments" / "results"
N_BOOT, SEED = 10000, 0


def correct(items: list[dict]) -> np.ndarray:
    return np.array([float(it["parsed"] == it["label"]) for it in items])


def paired_ci(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    d = (a - b)[idx].mean(1)
    return float((a - b).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    runs = {p.stem.removeprefix("pope_"): json.loads(p.read_text()) for p in sorted(RES.glob("pope_*.json")) if p.stem != "pope_summary"}
    base = {k: correct(runs[k]["items"]) for k in ("standard_topk", "standard_topk_k1")}
    out = {}
    for name, r in runs.items():
        c = correct(r["items"])
        row = {"accuracy": float(c.mean()), "f1": r["overall"]["f1"], "mean_k": r["mean_k"], "flops_fraction": r["flops_fraction"]}
        for bn, bc in base.items():
            if name != bn and len(bc) == len(c):
                d, lo, hi = paired_ci(c, bc)
                row[f"diff_vs_{bn}"] = {"diff": d, "ci": [lo, hi]}
        out[name] = row
        diffs = " ".join(f"vs {bn}: {row[f'diff_vs_{bn}']['diff'] * 100:+.2f} [{row[f'diff_vs_{bn}']['ci'][0] * 100:+.2f}, "
                         f"{row[f'diff_vs_{bn}']['ci'][1] * 100:+.2f}]" for bn in base if f"diff_vs_{bn}" in row)
        print(f"{name:22s} acc={row['accuracy']:.3f} K={row['mean_k']:.2f} {diffs}")
    (RES / "pope_summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
