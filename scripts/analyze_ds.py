"""DeepSeek Pareto analizi: doğruluk vs token-ağırlıklı FLOPs, eşleştirilmiş bootstrap GA. Çıktı: experiments/results/ds_pareto_summary.json,
paper/tables/t7_ds_pareto.tex, paper/figures/f8_ds_pareto.png. Kullanım: venv/bin/python scripts/analyze_ds.py"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
res = json.loads((ROOT / "experiments/results/ds_eval_test.json").read_text())
rng = np.random.default_rng(0)
N = len(res["topk_k6"]["correct"])
IDX = rng.integers(0, N, (10000, N))


def diff(a, b):
    d = (np.array(res[a]["correct"]) - np.array(res[b]["correct"]))[IDX].mean(1) * 100
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


PAIRS = [("dyn_l005", "topk_k4"), ("dyn_l015", "topk_k2"), ("dyn_l040", "topk_k1"), ("dyn_l015", "topk_k3"),
         ("dyn_l005", "topk_k3"), ("topk_k3", "topk_k6"), ("topk_k4", "topk_k6"), ("topk_k2", "topk_k6"),
         ("topk_k1", "topk_k6"), ("dyn_l005", "topk_k6"), ("dyn_l015", "topk_k6")]
out = {"n": N, "configs": {k: {"acc": v["accuracy"], "flops": v["flops"], "k_prefill": v["k_prefill"], "k_decode": v["k_decode"]}
                          for k, v in res.items()},
       "paired_pp": {f"{a}-{b}": dict(zip(("d", "lo", "hi"), diff(a, b), strict=True)) for a, b in PAIRS}}
(ROOT / "experiments/results/ds_pareto_summary.json").write_text(json.dumps(out, indent=1))
for k, v in out["paired_pp"].items():
    print(f"{k:22s} {v['d']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}]")

rows = ["\\begin{table}[t]\\centering\\small", "\\begin{tabular}{lccc}\\toprule", "Policy & FLOPs & Acc. & $\\Delta$ vs Top-6 (pp)\\\\\\midrule"]
for k in ("topk_k6", "topk_k4", "topk_k3", "topk_k2", "topk_k1", "dyn_l005", "dyn_l015", "dyn_l040"):
    c = out["configs"][k]
    d = out["paired_pp"].get(f"{k}-topk_k6")
    ds = f"{d['d']:+.1f} [{d['lo']:+.1f},{d['hi']:+.1f}]" if d else "--"
    rows.append(f"{k.replace('_', '-')} & {c['flops']:.3f} & {c['acc']:.3f} & {ds}\\\\")
rows += ["\\bottomrule\\end{tabular}", "\\caption{DeepSeek-VL2-Tiny on the synthetic test set (1,321 questions): accuracy and token-weighted expert FLOPs; $\\Delta$ is the paired difference to Top-6 in points with 95\\% CI.}", "\\label{tab:ds}", "\\end{table}"]
(ROOT / "paper/tables/t7_ds_pareto.tex").write_text("\n".join(rows))

fig, ax = plt.subplots(figsize=(4.5, 3.4))
tk = [out["configs"][f"topk_k{k}"] for k in (1, 2, 3, 4, 6)]
ax.plot([c["flops"] for c in tk], [c["acc"] for c in tk], "o-", label="Top-K (K=1,2,3,4,6)")
dy = [out["configs"][k] for k in ("dyn_l040", "dyn_l015", "dyn_l005")]
ax.plot([c["flops"] for c in dy], [c["acc"] for c in dy], "s--", c="C3", label="Dynamic-K (PPO)")
ax.set_xlabel("token-weighted expert FLOPs"); ax.set_ylabel("test accuracy"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(ROOT / "paper/figures/f8_ds_pareto.png", dpi=200); fig.savefig(ROOT / "paper/figures/f8_ds_pareto.pdf")
