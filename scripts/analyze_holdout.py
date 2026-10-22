"""ScienceQA test2 (bağımsız dilim): politikaların Top-K+statik hull'a göre kazancı. Kullanım: venv/bin/python scripts/analyze_holdout.py"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
res = json.loads((ROOT / "experiments/results/ds_eval_sqa_test2.json").read_text())
rng = np.random.default_rng(0)
n = len(res["topk_k6"]["correct"])
idx = rng.integers(0, n, (10000, n))
pts = sorted((v["flops"], v["accuracy"], k) for k, v in res.items() if k.startswith(("topk", "st")))
hull = []
for q in pts:
    while len(hull) >= 2 and (hull[-1][1] - hull[-2][1]) * (q[0] - hull[-2][0]) <= (q[1] - hull[-2][1]) * (hull[-1][0] - hull[-2][0]):
        hull.pop()
    hull.append(q)
out = {}
for k, v in res.items():
    if not k.startswith("dyn"):
        continue
    f = v["flops"]
    for a, b in zip(hull, hull[1:]):
        if a[0] <= f <= b[0]:
            w = (f - a[0]) / (b[0] - a[0])
            base = (1 - w) * np.array(res[a[2]]["correct"], float) + w * np.array(res[b[2]]["correct"], float)
    d = (np.array(v["correct"], float) - base)[idx].mean(1) * 100
    t3 = (np.array(v["correct"], float) - np.array(res["topk_k3"]["correct"], float))[idx].mean(1) * 100
    out[k] = dict(acc=v["accuracy"], flops=f, gain=float(d.mean()), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                  vs_top3=float(t3.mean()), lo3=float(np.percentile(t3, 2.5)), hi3=float(np.percentile(t3, 97.5)))
    print(f"{k:12s} acc={v['accuracy']:.3f} fl={f:.3f} vs hull {out[k]['gain']:+.2f} [{out[k]['lo']:+.2f},{out[k]['hi']:+.2f}]  vs Top-3 {out[k]['vs_top3']:+.2f} [{out[k]['lo3']:+.2f},{out[k]['hi3']:+.2f}]")
(ROOT / "experiments/results/ds_sqa_test2_summary.json").write_text(json.dumps(out, indent=1))
