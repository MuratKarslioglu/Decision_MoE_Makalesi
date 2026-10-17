"""Makale tabloları (t1–t5, LaTeX) ve figürleri (f2–f7, PDF+PNG) mevcut sonuç JSON'larından üretir.

GPU gerekmez; yalnızca `experiments/results/` ve `artifacts/` okunur. Önce `scripts/analyze_results.py` çalıştırılmış olmalı.
Çıktı: paper/tables/*.tex, paper/figures/*.{pdf,png}. Çalıştırma: venv/bin/python scripts/generate_paper_materials.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper.style import COLORS, apply_style, variant_color  # noqa: E402

RES = ROOT / "experiments" / "results"
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
CRITERIA = ["task utility", "modality match", "specialization", "reliability", "latency", "FLOPs", "load balance"]
TYPES = ["T1", "T2", "T3", "T4", "T5"]
TYPE_NAMES = {"T1": "Perception", "T2": "Spatial", "T3": "Multi-hop", "T4": "Halluc. trap", "T5": "OCR+reason"}
ORDER = ["standard_topk", "standard_topk_k1", "fixed_mcdm_no_utility", "fixed_mcdm_cosine", "ppo_fixedk_no_utility",
         "ppo_fixedk_cosine", "ppo_direct_fixedk", "decisionmoe_b"]
LABEL = {
    "standard_topk": "Top-2 (baseline)", "standard_topk_k1": "Top-1", "fixed_mcdm_no_utility": "Fixed-MCDM (no util.)",
    "fixed_mcdm_cosine": "Fixed-MCDM", "ppo_fixedk_no_utility": "PPO-K2 (no util.)", "ppo_fixedk_cosine": "PPO-K2",
    "ppo_direct_fixedk": "PPO-direct K2", "decisionmoe_b": "DecisionMoE-B (Dyn-K)",
}
SEEDS = ["decisionmoe_b", "decisionmoe_b_s1", "decisionmoe_b_s2"]  # eval dosyası adları
SEED_KEY = {"decisionmoe_b": "decisionmoe_b", "decisionmoe_b_s1": "decisionmoe_b@s1", "decisionmoe_b_s2": "decisionmoe_b@s2"}  # analiz anahtarları


def load(name: str):
    return json.loads((RES / name).read_text())


def savefig(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    print("yazıldı: figures/", name)


def write_tex(name: str, caption: str, label: str, header: list[str], rows: list[list[str]], align: str) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    lines = ["\\begin{table}[t]", "\\centering", "\\small", f"\\begin{{tabular}}{{{align}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}", ""]
    (TAB / f"{name}.tex").write_text("\n".join(lines))
    print("yazıldı: tables/", name)


def esc(s: str) -> str:
    return s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def ci_str(d: float, ci: list[float]) -> str:
    return f"{100 * d:+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"


def eval_items(name: str) -> list[dict]:
    return load(f"eval_{name}_test.json")["items"]


# ---------------------------------------------------------------- tablolar
def table_main(abl, eff, lat) -> None:
    rows = []
    for v in ORDER:
        a, e = abl[v], eff[v]
        d2 = a["vs"].get("standard_topk")
        d1 = a["vs"].get("standard_topk_k1")
        hc = a["hallucination_corrected"]["no_broken"]["rate"]
        r = lat.get(v, {}).get("ratio_vs_standard_topk")
        rows.append([esc(LABEL[v]), f"{100 * a['accuracy']:.1f}",
                     ci_str(d2["acc_diff"], d2["acc_ci"]) if d2 else "--",
                     ci_str(d1["acc_diff"], d1["acc_ci"]) if d1 else "--",
                     f"{100 * a['hallucination_rate']:.1f} / {100 * hc:.1f}",
                     f"{e['mean_k']:.2f}", f"{e['flops_fraction']:.2f}", f"{r:.2f}" if r else "--"])
    write_tex("t1_main", "Test results (1321 questions, seed 42). $\\Delta$: paired accuracy difference in points vs. Top-2 / Top-1 "
              "with 95\\% bootstrap CI. Halluc.: raw / after removing broken traps (\\%). FLOPs: token-weighted fraction of the dense "
              "top-4 cost. Latency: controlled benchmark ratio vs. Top-2.", "tab:main",
              ["Method", "Acc.", "$\\Delta$ vs Top-2", "$\\Delta$ vs Top-1", "Halluc.", "$\\bar K$", "FLOPs", "Lat."], rows, "lccccccc")


def table_types(abl) -> None:
    rows = [[esc(LABEL[v])] + [f"{100 * abl[v]['accuracy_by_type'][t]:.1f}" for t in TYPES] for v in ORDER]
    write_tex("t2_by_type", "Accuracy (\\%) by question type (strict exact match; T3 references are long sentences, see text).",
              "tab:bytype", ["Method"] + [f"{t} {TYPE_NAMES[t]}" for t in TYPES], rows, "lccccc")


def table_seeds(abl, eff) -> None:
    rows = []
    for s, nm in zip(SEEDS, ["42", "1", "2"]):
        rows.append([nm, f"{100 * abl[SEED_KEY[s]]['accuracy']:.1f}", f"{eff[SEED_KEY[s]]['mean_k']:.2f}", f"{eff[SEED_KEY[s]]['flops_fraction']:.3f}"])
    acc = np.array([abl[SEED_KEY[s]]["accuracy"] for s in SEEDS]) * 100
    k = np.array([eff[SEED_KEY[s]]["mean_k"] for s in SEEDS])
    fl = np.array([eff[SEED_KEY[s]]["flops_fraction"] for s in SEEDS])
    rows.append(["mean $\\pm$ std", f"{acc.mean():.1f} $\\pm$ {acc.std(ddof=1):.1f}", f"{k.mean():.2f} $\\pm$ {k.std(ddof=1):.2f}",
                 f"{fl.mean():.3f} $\\pm$ {fl.std(ddof=1):.3f}"])
    write_tex("t3_seeds", "DecisionMoE-B over three seeds (test). Prefill uses $K{=}1$ in every seed; seed-to-seed variation in "
              "$\\bar K$ comes from decode steps, which carry little weight in token-weighted FLOPs.", "tab:seeds",
              ["Seed", "Acc.", "$\\bar K$", "FLOPs"], rows, "lccc")


def table_pope() -> None:
    summ = load("pope_summary.json")
    names = ["standard_topk", "standard_topk_k1", "fixed_mcdm_cosine", "ppo_direct_fixedk", "decisionmoe_b", "decisionmoe_b_s1", "decisionmoe_b_s2"]
    lab = dict(LABEL, decisionmoe_b_s1="DecisionMoE-B (seed 1)", decisionmoe_b_s2="DecisionMoE-B (seed 2)")
    lab["decisionmoe_b"] = "DecisionMoE-B (seed 42)"
    rows = []
    for v in names:
        p = load(f"pope_{v}.json")
        sp = p["splits"]
        d = summ[v].get("diff_vs_standard_topk")
        rows.append([esc(lab[v])] + [f"{100 * sp[x]['accuracy']:.1f}" for x in ("random", "popular", "adversarial")]
                    + [f"{100 * p['overall']['accuracy']:.1f}", f"{100 * p['overall']['f1']:.1f}", f"{p['mean_k']:.2f}",
                       ci_str(d["diff"], d["ci"]) if d else "--"])
    write_tex("t4_pope", "POPE (3$\\times$1000 questions): accuracy per split, overall accuracy and F1 (\\%), mean $K$, and paired "
              "accuracy difference vs. Top-2 (points, 95\\% bootstrap CI). Learned Dynamic-K policies are slightly but consistently "
              "below the fixed baselines.", "tab:pope",
              ["Method", "Rand.", "Pop.", "Adv.", "Acc.", "F1", "$\\bar K$", "$\\Delta$ vs Top-2"], rows, "lccccccc")


def table_sensitivity() -> None:
    s = load("sensitivity_val.json")["results"]
    names = {"topk_k1": "Top-1", "topk_k2": "Top-2", "topk_k3": "Top-3", "topk_k4": "Top-4", "random_k1": "Random K=1",
             "random_k2": "Random K=2", "fixed_mcdm_k2": "Fixed-MCDM K=2", "pin_expert0_k1": "Expert 0 only",
             "pin_expert1_k1": "Expert 1 only", "pin_expert2_k1": "Expert 2 only", "pin_expert3_k1": "Expert 3 only"}
    rows = [[names[k], f"{100 * v['accuracy']:.1f}", f"{100 * v['hallucination_rate']:.1f}"] for k, v in s.items() if k in names]
    write_tex("t5_sensitivity", "Routing sensitivity on 400 validation questions: accuracy and hallucination rate (\\%) under "
              "different expert selections. All settings fall within a 2.5-point band.", "tab:sens",
              ["Routing", "Acc.", "Halluc."], rows, "lcc")


# ---------------------------------------------------------------- figürler
def read_log(path: Path) -> list[dict]:
    return [r for r in (json.loads(line) for line in path.read_text().splitlines()) if "train/reward" in r]


def fig_reward_hacking(plt) -> None:
    """f2: adım bazlı FLOPs ödülü (reward hacking) vs token ağırlıklı."""
    old = read_log(RES / "archive_run2_stepflops" / "decisionmoe_b_20261014_192315.jsonl")
    new = read_log(RES / "routing_logs" / "decisionmoe_b_20261015_131740.jsonl")
    ev_old = load("archive_run2_stepflops/eval_decisionmoe_b_test.json")
    ev_new = load("eval_decisionmoe_b_test.json")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].plot([r["step"] for r in old], [r["train/flops_ratio"] for r in old], color=COLORS["ablation"], label="step-averaged reward")
    ax[0].plot([r["step"] for r in new], [r["train/flops_ratio"] for r in new], color=COLORS["decisionmoe_b"], label="token-weighted reward")
    ax[0].set_xlabel("PPO update")
    ax[0].set_ylabel("training FLOPs fraction")
    ax[0].set_title("Reported training cost")
    ax[0].legend()
    labels = ["step-avg.\nreward", "token-weighted\nreward"]
    # run2 kayıtlarında ntok yok: aynı sorunun istem uzunluğu (prefill) yeni koşudan alınır, decode adımları 1 token
    ntok = {it["id"]: it["ntok"] for it in eval_items("decisionmoe_b")}
    num = den = 0.0
    for it in ev_old["items"]:
        nt = ntok[it["id"]][:len(it["ks"])]
        num += sum(k * t for k, t in zip(it["ks"], nt))
        den += sum(nt)
    old_true = num / den / 4
    step_f = [ev_old["flops_fraction"], ev_new["flops_fraction_step"]]
    tok_f = [old_true, ev_new["flops_fraction"]]
    print(f"run2 token-ağırlıklı FLOPs (yeniden hesap): {old_true:.3f}")
    x = np.arange(2)
    ax[1].set_ylim(0, 1.4)
    ax[1].bar(x - 0.18, step_f, 0.36, color=COLORS["neutral"], label="per-step mean (misleading)")
    ax[1].bar(x + 0.18, tok_f, 0.36, color=COLORS["decisionmoe_b"], label="token-weighted (true)")
    ax[1].set_xticks(x, labels)
    ax[1].set_ylabel("test FLOPs fraction")
    ax[1].set_title("Evaluation of the learned policy")
    ax[1].legend(loc="upper center", ncol=1, fontsize=8)
    fig.tight_layout()
    savefig(fig, "f2_reward_hacking")


def fig_k_by_type(plt) -> None:
    """f3: soru tipine göre ortalama K (prefill / decode), 3 seed."""
    res = {"prefill": [], "decode": []}
    for s in SEEDS:
        items = eval_items(s)
        for phase, sl in (("prefill", slice(0, 1)), ("decode", slice(1, None))):
            row = []
            for t in TYPES:
                ks = [k for it in items if it["type"] == t for k in it["ks"][sl]]
                row.append(np.mean(ks))
            res[phase].append(row)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(TYPES))
    for i, (phase, col) in enumerate((("prefill", COLORS["decisionmoe_b"]), ("decode", COLORS["baseline"]))):
        a = np.array(res[phase])
        ax.bar(x + (i - 0.5) * 0.36, a.mean(0), 0.36, yerr=a.std(0, ddof=1), color=col, label=phase, capsize=2)
    ax.set_xticks(x, [f"{t}\n{TYPE_NAMES[t]}" for t in TYPES])
    ax.set_ylabel("mean K (3 seeds)")
    ax.set_ylim(0, 4)
    ax.legend()
    fig.tight_layout()
    savefig(fig, "f3_k_by_type")


def fig_expert_by_type(plt) -> None:
    """f4: soru tipi x expert aktivasyonu (Top-2 vs DecisionMoE-B)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    for ax, v in zip(axes, ["standard_topk", "decisionmoe_b"]):
        items = eval_items(v)
        m = np.array([np.sum([it["util"] for it in items if it["type"] == t], axis=0) for t in TYPES], dtype=float)
        m = m / m.sum(1, keepdims=True)
        im = ax.imshow(m, vmin=0, vmax=0.5, cmap="Blues")
        ax.set_xticks(range(4), [f"E{j}" for j in range(4)])
        ax.set_yticks(range(5), TYPES)
        ax.set_title(LABEL[v])
        for i in range(5):
            for j in range(4):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8, label="share of activations")
    savefig(fig, "f4_expert_by_type")


def fig_weights(plt) -> None:
    """f5: PPO'nun öğrendiği ortalama kriter ağırlıkları (3 seed) ve sabit-MCDM referansı."""
    w = np.array([load(f"eval_{s}_test.json")["mean_criteria_weights"] for s in SEEDS])
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(range(7), w.mean(0), yerr=w.std(0, ddof=1), color=COLORS["decisionmoe_b"], capsize=2)
    ax.axhline(1 / 7, color=COLORS["neutral"], ls="--", lw=1, label="uniform (1/7)")
    ax.set_xticks(range(7), CRITERIA, rotation=30, ha="right")
    ax.set_ylabel("mean weight (3 seeds)")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "f5_criteria_weights")


def fig_ablation(plt, abl) -> None:
    """f6: variant doğrulukları + top-2'ye göre fark (GA)."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    y = np.arange(len(ORDER))[::-1]
    for yi, v in zip(y, ORDER):
        a = abl[v]
        ax.barh(yi, 100 * a["accuracy"], color=variant_color(v), height=0.6)
        ax.text(100 * a["accuracy"] + 0.2, yi, f"{100 * a['accuracy']:.1f}", va="center", fontsize=9)
    ax.set_yticks(y, [LABEL[v] for v in ORDER])
    ax.set_xlim(35, 43)
    ax.set_xlabel("test accuracy (%)  (axis truncated; all differences within 95% CI)")
    fig.tight_layout()
    savefig(fig, "f6_ablation")


def fig_pareto(plt, abl, eff, lat) -> None:
    """f7: doğruluk–FLOPs ve doğruluk–gecikme düzlemleri."""
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for v in ORDER:
        a = 100 * abl[v]["accuracy"]
        ax[0].scatter(eff[v]["flops_fraction"], a, color=variant_color(v), s=45)
        if eff[v]["flops_fraction"] < 0.4 or v in ("standard_topk", "fixed_mcdm_cosine", "ppo_direct_fixedk"):
            ax[0].annotate(LABEL[v], (eff[v]["flops_fraction"], a), fontsize=7, xytext=(-4, 4), ha="right" if eff[v]["flops_fraction"] > 0.4 else "left", textcoords="offset points")
        r = lat.get(v, {}).get("ratio_vs_standard_topk")
        if r:
            ax[1].scatter(r, a, color=variant_color(v), s=45)
            ax[1].annotate(LABEL[v], (r, a), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[0].set_xlabel("token-weighted FLOPs fraction")
    ax[0].set_ylabel("test accuracy (%)")
    ax[1].set_xlabel("latency ratio vs Top-2")
    for a_ in ax:
        a_.set_ylim(39.0, 40.5)
    fig.tight_layout()
    savefig(fig, "f7_pareto")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style()
    abl, eff = load("ablation_scores.json"), load("efficiency_metrics.json")
    lat = load("latency_benchmark.json")["variants"]
    table_main(abl, eff, lat)
    table_types(abl)
    table_seeds(abl, eff)
    table_pope()
    table_sensitivity()
    fig_reward_hacking(plt)
    fig_k_by_type(plt)
    fig_expert_by_type(plt)
    fig_weights(plt)
    fig_ablation(plt, abl)
    fig_pareto(plt, abl, eff, lat)


if __name__ == "__main__":
    main()
