"""Uzmanlaşma deneyi analizi: domain_sensitivity_*.json (400-1074 val) ve eval_*_spec_test.json (test) -> özet JSON + t6 tablosu.

Çıktı: experiments/results/specialization_summary.json, paper/tables/t6_specialization.tex
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.domains import DOMAIN_NAMES, domain_of  # noqa: E402

RES = ROOT / "experiments" / "results"
rng = np.random.default_rng(0)


def ci(z: np.ndarray) -> list[float]:
    b = [z[rng.integers(0, len(z), len(z))].mean() for _ in range(5000)]
    return [float(z.mean()), *map(float, np.quantile(b, [0.025, 0.975]))]


def main() -> None:
    out: dict = {"val_domain": {}, "test_spec": {}}
    for tag, label in (("base", "orijinal (n=400)"), ("lora300", "LoRA"), ("contrast300", "LoRA+kontrast"),
                       ("contrast2_300", "LoRA+güçlü kontrast")):
        f = RES / f"domain_sensitivity_{tag}.json"
        if not f.exists():
            continue
        D = json.load(open(f))
        c, d = D["correct"], np.array(D["domains"])
        orc = np.array([c[f"pin{x}"][i] for i, x in enumerate(d)])
        wrong = np.array([np.mean([c[f"pin{e}"][i] for e in range(4) if e != x]) for i, x in enumerate(d)])
        top2 = np.array(c["topk_k2"])
        out["val_domain"][tag] = {"label": label, "n": len(d), "top2": float(top2.mean()), "oracle": float(orc.mean()),
                                  "oracle_minus_top2": ci(orc - top2), "right_minus_wrong": ci(orc - wrong)}
    test = {x["id"]: x for x in json.load(open(ROOT / "decisionmoe_dataset" / "splits" / "test.json"))}
    ref = None
    for v in ("standard_topk", "fixed_mcdm_cosine", "ppo_direct_fixedk"):
        f = RES / f"eval_{v}_spec_test.json"
        if not f.exists():
            continue
        its = json.load(open(f))["items"]
        cc = np.array([i["correct"] for i in its], dtype=float)
        dd = np.array([domain_of(test[i["id"]]) for i in its])
        ref = cc if v == "standard_topk" else ref
        out["test_spec"][v] = {"accuracy": float(cc.mean()), "by_domain": {DOMAIN_NAMES[k]: float(cc[dd == k].mean()) for k in range(4)},
                               "diff_vs_top2": None if v == "standard_topk" else ci(cc - ref)}
    (RES / "specialization_summary.json").write_text(json.dumps(out, indent=1))
    rows = []
    for r in out["val_domain"].values():
        o, w = r["oracle_minus_top2"], r["right_minus_wrong"]
        rows.append(f"{r['label']} & {r['n']} & {r['top2']:.3f} & {r['oracle']:.3f} & "
                    f"{100*o[0]:+.1f} [{100*o[1]:+.1f}, {100*o[2]:+.1f}] & {100*w[0]:+.1f} [{100*w[1]:+.1f}, {100*w[2]:+.1f}] \\\\")
    tex = ("\\begin{tabular}{lrrrcc}\n\\toprule\nModel & $n$ & Top-2 & Oracle & Oracle$-$Top-2 (pp) & Right$-$wrong expert (pp) \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (ROOT / "paper" / "tables" / "t6_specialization.tex").write_text(tex)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
