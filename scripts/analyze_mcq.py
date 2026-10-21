"""Akıl yürütme rejimi (AI2D, ScienceQA-IMG): eşleşen-FLOPs Top-K karışımına göre kazanç (eşleştirilmiş bootstrap).
Çıktı: experiments/results/ds_mcq_summary.json, paper/tables/t8_mcq.tex, paper/figures/f9_mcq_pareto.{png,pdf}. Kullanım: venv/bin/python scripts/analyze_mcq.py"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILES = {"ai2d": "ds_eval_ai2d_test.json", "sqa": "ds_eval_sqa_test.json", "aok": "ds_eval_aok_test.json"}
rng = np.random.default_rng(0)
summary, tables = {}, []
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))


def baseline(res, f):
    ks = sorted((res[f"topk_k{k}"]["flops"], k) for k in range(1, 7))
    fl = np.array([a for a, _ in ks])
    j = int(np.clip(np.searchsorted(fl, f) - 1, 0, len(ks) - 2))
    w = (f - fl[j]) / (fl[j + 1] - fl[j])
    a, b = (np.array(res[f"topk_k{ks[j + t][1]}"]["correct"], float) for t in (0, 1))
    return (1 - w) * a + w * b


def hull_baseline(res, f):
    """Top-K ve sabit asimetrik (st*) noktalarının üst dışbükey zarfı üzerinde, f maliyetinde iki komşunun karışımı."""
    pts = sorted(((v["flops"], v["accuracy"], n) for n, v in res.items() if n.startswith(("topk", "st"))))
    hull = []
    for q in pts:
        while len(hull) >= 2 and (hull[-1][1] - hull[-2][1]) * (q[0] - hull[-2][0]) <= (q[1] - hull[-2][1]) * (hull[-1][0] - hull[-2][0]):
            hull.pop()
        hull.append(q)
    for a, b in zip(hull, hull[1:]):
        if a[0] <= f <= b[0]:
            w = (f - a[0]) / (b[0] - a[0])
            return (1 - w) * np.array(res[a[2]]["correct"], float) + w * np.array(res[b[2]]["correct"], float)
    return baseline(res, f)


for ax, (task, fn) in zip(axes, FILES.items(), strict=True):
    path = ROOT / "experiments/results" / fn
    if not path.exists():
        continue
    res = json.loads(path.read_text())
    if not all(f"topk_k{k}" in res for k in range(1, 7)):
        continue
    n = len(res["topk_k6"]["correct"])
    idx = rng.integers(0, n, (10000, n))
    rows = {}
    for name, v in res.items():
        if name.startswith("topk") or not (res["topk_k1"]["flops"] <= v["flops"] <= 1.0):
            continue
        d = (np.array(v["correct"], float) - baseline(res, v["flops"]))[idx].mean(1) * 100
        h = (np.array(v["correct"], float) - hull_baseline(res, v["flops"]))[idx].mean(1) * 100
        rows[name] = dict(acc=v["accuracy"], flops=v["flops"], d=float(d.mean()), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                          dh=float(h.mean()), loh=float(np.percentile(h, 2.5)), hih=float(np.percentile(h, 97.5)))
    summary[task] = {"n": n, "topk": {k: dict(acc=res[f"topk_k{k}"]["accuracy"], flops=res[f"topk_k{k}"]["flops"]) for k in range(1, 7)}, "policies": rows}
    tk = [res[f"topk_k{k}"] for k in range(1, 7)]
    ax.plot([c["flops"] for c in tk], [c["accuracy"] for c in tk], "o-", label="Top-K")
    for pref, mk, col, lab in (("dyn", "s", "C3", "PPO"), ("topp", "^", "C2", "top-p heuristic"), ("st", "x", "C1", "static (K_p,K_d)")):
        pts = [r for k, r in rows.items() if k.startswith(pref)]
        ax.scatter([r["flops"] for r in pts], [r["acc"] for r in pts], marker=mk, c=col, label=lab, zorder=3)
    ax.set_title(f"{task} (n={n})"); ax.set_xlabel("token-weighted expert FLOPs"); ax.grid(alpha=.3)
    tables.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{task}}}}}\\\\")
    for k, r in rows.items():
        tables.append(f"{k.replace('_', '-')} & {r['flops']:.3f} & {r['acc']:.3f} & {r['d']:+.1f} [{r['lo']:+.1f},{r['hi']:+.1f}] & {r['dh']:+.1f} [{r['loh']:+.1f},{r['hih']:+.1f}]\\\\")
    print(task)
    for k, r in rows.items():
        print(f"  {k:11s} acc={r['acc']:.3f} flops={r['flops']:.3f} gain={r['d']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}]  vs hull(+static) {r['dh']:+.2f} [{r['loh']:+.2f},{r['hih']:+.2f}]")
axes[0].set_ylabel("test accuracy"); axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(ROOT / "paper/figures/f9_mcq_pareto.png", dpi=200); fig.savefig(ROOT / "paper/figures/f9_mcq_pareto.pdf")
(ROOT / "experiments/results/ds_mcq_summary.json").write_text(json.dumps(summary, indent=1))
(ROOT / "paper/tables/t8_mcq.tex").write_text("\\begin{table}[t]\\centering\\small\n\\begin{tabular}{lcccc}\\toprule\nPolicy & FLOPs & Acc. & Gain vs Top-K mix (pp) & Gain vs Top-K+static hull (pp)\\\\\\midrule\n" + "\n".join(tables) + "\n\\bottomrule\\end{tabular}\n\\caption{AI2D and ScienceQA-IMG test sets (597 questions each): gain in accuracy points over the matched-FLOPs mixture of adjacent Top-K points, with paired-bootstrap 95\\% CIs. PPO policies named a$\\cdot$ were trained on AI2D, q$\\cdot$ on ScienceQA.}\\label{tab:mcq}\n\\end{table}")
