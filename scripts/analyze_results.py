"""Test değerlendirmelerini birleştirir: experiments/results/{ablation_scores,efficiency_metrics,expert_utilization,criteria_weights}.json
üretir ve eşleştirilmiş bootstrap güven aralıklarıyla özet tabloyu yazdırır (referans: standard_topk ve standard_topk_k1).
"""
import glob
import json
from pathlib import Path

import torch

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"
REFS = ("standard_topk", "standard_topk_k1")
N_BOOT = 5000


def paired_ci(a: list[float], b: list[float], seed: int = 0) -> tuple[float, float, float]:
    """mean(a-b) ve %95 GA; soru bazında eşleştirilmiş bootstrap."""
    d = torch.tensor(a, dtype=torch.float32) - torch.tensor(b, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    means = d[torch.randint(0, d.numel(), (N_BOOT, d.numel()), generator=g)].mean(1)
    return d.mean().item(), means.quantile(0.025).item(), means.quantile(0.975).item()


def corrected_hallucination(evals: dict) -> dict:
    """T4 denetimine (experiments/results/t4_audit.json) göre düzeltilmiş hallucination oranı.

    'broken' = aranan nesne görüntüde gerçekten var (tuzak bozuk); 'amb' = belirsiz. Yalnızca bir variant'ın
    hallucination saydığı sorular denetlendi; sayılmayanlar (model No dedi) zaten hallucination değil.
    """
    f = RESULTS / "t4_audit.json"
    if not f.exists():
        return {}
    v = json.load(open(f))["verdicts"]
    broken = {i for i, x in v.items() if x["verdict"] == "broken"}
    amb = {i for i, x in v.items() if x["verdict"] == "amb"}
    out = {}
    for name, d in evals.items():
        t4 = {i["id"]: i for i in d.get("items", []) if i["type"] == "T4"}
        row = {}
        for tag, excl in (("raw", set()), ("no_broken", broken), ("no_broken_amb", broken | amb)):
            keep = [k for k in t4 if k not in excl]
            row[tag] = {"rate": sum(t4[k]["hallucinated"] for k in keep) / max(1, len(keep)), "n": len(keep)}
        out[name] = row
    return out


def main() -> None:
    evals = {}
    for f in sorted(glob.glob(str(RESULTS / "eval_*_test.json"))):
        d = json.load(open(f))
        seed = (d.get("meta") or {}).get("seed")
        key = d["variant"] if seed in (None, 42) else f"{d['variant']}@s{seed}"     # seed 42 / kayıtsız = ana koşu
        evals[key] = d
    base_time = evals["standard_topk"]["mean_step_time_s"]
    scores, eff, util, crit = {}, {}, {}, {}
    for name, d in evals.items():
        items = {it["id"]: it for it in d.get("items", [])}
        row = {"accuracy": d["accuracy"], "accuracy_by_type": d["accuracy_by_type"],
               "hallucination_rate": d["hallucination_rate"], "n": d["n"], "vs": {}}
        for ref in REFS:
            if ref == name or ref not in evals or not items or not evals[ref].get("items"):
                continue
            ref_items = {it["id"]: it for it in evals[ref]["items"]}
            ids = sorted(items)
            m, lo, hi = paired_ci([items[i]["correct"] for i in ids], [ref_items[i]["correct"] for i in ids])
            t4 = [i for i in ids if items[i]["hallucinated"] is not None]
            hm, hlo, hhi = paired_ci([items[i]["hallucinated"] for i in t4], [ref_items[i]["hallucinated"] for i in t4])
            row["vs"][ref] = {"acc_diff": m, "acc_ci": [lo, hi], "halluc_diff": hm, "halluc_ci": [hlo, hhi]}
        scores[name] = row
        eff[name] = {"mean_k": d["mean_k"], "flops_fraction": d["flops_fraction"], "k_distribution": d["k_distribution"],
                     "mean_step_time_s": d["mean_step_time_s"], "latency_ratio": d["mean_step_time_s"] / base_time}
        util[name] = d["expert_utilization"]
        crit[name] = d.get("mean_criteria_weights") or d.get("mean_expert_prior")
    corr = corrected_hallucination(evals)
    for name, row in scores.items():
        if name in corr:
            row["hallucination_corrected"] = corr[name]
    for fname, obj in (("ablation_scores", scores), ("efficiency_metrics", eff),
                       ("expert_utilization", util), ("criteria_weights", crit)):
        (RESULTS / f"{fname}.json").write_text(json.dumps(obj, indent=1))

    print(f"{'variant':24s} {'acc':>6s} {'Δacc vs top2 [95% GA]':>26s} {'Δacc vs top1 [95% GA]':>26s} {'hal':>6s} {'K':>5s} {'FLOPs':>6s} {'lat':>5s}")
    for name, r in scores.items():
        def fmt(ref, row=r):
            v = row["vs"].get(ref)
            return f"{100*v['acc_diff']:+5.1f} [{100*v['acc_ci'][0]:+5.1f},{100*v['acc_ci'][1]:+5.1f}]" if v else "-"
        e = eff[name]
        print(f"{name:24s} {r['accuracy']:6.3f} {fmt('standard_topk'):>26s} {fmt('standard_topk_k1'):>26s} "
              f"{r['hallucination_rate']:6.3f} {e['mean_k']:5.2f} {e['flops_fraction']:6.3f} {e['latency_ratio']:5.2f}")
    if corr:
        print(f"\n{'variant':24s} {'halluc ham':>11s} {'bozuk tuzaklar çıkarıldı':>26s} {'bozuk+belirsiz çıkarıldı':>26s}")
        for name, c in corr.items():
            print(f"{name:24s} {c['raw']['rate']:11.3f} {c['no_broken']['rate']:19.3f} (n={c['no_broken']['n']}) {c['no_broken_amb']['rate']:19.3f} (n={c['no_broken_amb']['n']})")
    seeds = {}
    for key in scores:
        base = key.split("@s")[0]
        seeds.setdefault(base, []).append(key)
    multi = {b: ks for b, ks in seeds.items() if len(ks) > 1}
    if multi:
        import statistics as st
        print(f"\n{'variant (seed sayısı)':28s} {'doğruluk ort±std':>18s} {'K ort±std':>14s} {'FLOPs ort±std':>16s}")
        for b, ks in multi.items():
            f = lambda xs: f"{st.mean(xs):.3f}±{st.pstdev(xs):.3f}"
            print(f"{b + ' (' + str(len(ks)) + ')':28s} {f([scores[k]['accuracy'] for k in ks]):>18s} "
                  f"{f([eff[k]['mean_k'] for k in ks]):>14s} {f([eff[k]['flops_fraction'] for k in ks]):>16s}")
        (RESULTS / "seed_summary.json").write_text(json.dumps(
            {b: {"seeds": ks, "accuracy": [scores[k]["accuracy"] for k in ks], "mean_k": [eff[k]["mean_k"] for k in ks],
                 "flops_fraction": [eff[k]["flops_fraction"] for k in ks]} for b, ks in multi.items()}, indent=1))
    print("\nPareto notu: bir variant, doğruluğu (GA ile) eşit olup FLOPs'u daha yüksekse standard_topk_k1 tarafından domine edilir.")
    k1 = eff.get("standard_topk_k1")
    if k1:
        for name, e in eff.items():
            if e["flops_fraction"] > k1["flops_fraction"] + 1e-6 and name != "standard_topk_k1":
                v = scores[name]["vs"].get("standard_topk_k1")
                same = v and v["acc_ci"][0] <= 0 <= v["acc_ci"][1]
                print(f"  {name:24s} FLOPs {e['flops_fraction']:.3f} > {k1['flops_fraction']:.3f}: doğruluk farkı anlamsız={bool(same)} -> domine edilir" if same
                      else f"  {name:24s} FLOPs {e['flops_fraction']:.3f}: doğruluk farkı anlamlı ya da veri yok")


if __name__ == "__main__":
    main()
