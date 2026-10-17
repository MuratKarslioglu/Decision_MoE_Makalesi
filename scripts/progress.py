"""Hat ilerleme takibi: tüm işler için ilerleme çubukları, genel yüzde ve tahmini bitiş.

Kullanım:  python3 scripts/progress.py            (tek seferlik)
           python3 scripts/progress.py --watch    (10 sn'de bir yenilenir, Ctrl+C ile çıkılır)
Yalnızca standart kütüphane kullanır; GPU/torch gerekmez.
"""
import glob
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
LOGS = RES / "routing_logs"
EVAL_SEC = 750          # bir test değerlendirmesinin ölçülen süresi (~12.5 dk)
ANALYSIS_SEC = 60
MAIN_EVAL = ["standard_topk", "standard_topk_k1", "fixed_mcdm_no_utility", "fixed_mcdm_cosine",
             "ppo_fixedk_cosine", "ppo_fixedk_no_utility", "ppo_direct_fixedk",
             "decisionmoe_b", "decisionmoe_a", "ppo_direct_dynk"]
SWEEP = ["decisionmoe_b_l03", "decisionmoe_b_l30"]
POPE_VARIANTS = ["standard_topk", "standard_topk_k1", "decisionmoe_b", "ppo_direct_dynk"]
POPE_SEC = 1500        # variant başına ~25 dk (3 x 1000 soru)
LATENCY_SEC = 1800
TRAIN_TARGET = 300
SEC_PER_UPDATE_DEFAULT = 45.0
W = 28


def bar(frac: float, width: int = W) -> str:
    frac = max(0.0, min(1.0, frac))
    full = int(frac * width)
    return "█" * full + "░" * (width - full)


def fmt_dur(sec: float) -> str:
    sec = int(max(0, sec))
    return f"{sec // 3600} sa {sec % 3600 // 60:02d} dk" if sec >= 3600 else f"{sec // 60}dk {sec % 60:02d}sn"


def processes() -> str:
    """Yalnızca gerçek python/bash betik süreçleri (başka komutların metni içinde geçen adları saymaz)."""
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return "\n".join(l for l in out.splitlines() if l.startswith(("python ", "python3 ", "bash scripts/")))


def train_state(v: str, target: int, procs: str) -> dict:
    files = sorted(f for f in glob.glob(str(LOGS / f"{v}_*.jsonl")) if re.search(rf"/{re.escape(v)}_\d{{8}}_\d{{6}}\.jsonl$", f))
    step, sec_up, K = 0, SEC_PER_UPDATE_DEFAULT, None
    if files:
        rows = [json.loads(l) for l in open(files[-1]) if '"train/reward"' in l]
        if rows:
            step = rows[-1]["step"]
            sec_up = sum(r.get("time/update_s", sec_up) for r in rows[-10:]) / len(rows[-10:])
            K = rows[-1].get("routing/mean_K")
    running = f"train.py --variant {v} " in procs
    return {"frac": step / target, "step": step, "target": target, "running": running,
            "remaining": (target - step) * sec_up, "extra": f"K={K:.2f}" if K else ""}


def eval_state(v: str, procs: str) -> dict:
    done = (RES / f"eval_{v}_test.json").exists()
    running = f"evaluate.py --variant {v} " in procs
    frac, extra = (1.0 if done else 0.0), ""
    pf = RES / f".progress_eval_{v}.json"
    if running and pf.exists():
        p = json.loads(pf.read_text())
        frac = p["i"] / p["n"]
        extra = f"{p['i']}/{p['n']} soru"
    elif running:   # eski kodla başlamış süreç: süreden tahmin
        etimes = subprocess.run(["ps", "-eo", "etimes,args"], capture_output=True, text=True).stdout
        for line in etimes.splitlines():
            if f" python scripts/evaluate.py --variant {v} " in line and "grep" not in line:
                frac = min(0.99, int(line.split()[0]) / EVAL_SEC)
                extra = "(süreden tahmin)"
    return {"frac": frac, "running": running and not done, "remaining": (1 - frac) * EVAL_SEC, "extra": extra, "done": done}


def render() -> str:
    procs = processes()
    jobs = []   # (grup, etiket, durum, ağırlık_sn)
    def simple(group, label, done, running, sec):
        jobs.append((group, label, {"frac": 1.0 if done else 0.0, "running": running and not done,
                                    "remaining": 0 if done else sec, "extra": ""}, sec))
    UPD = 150
    st = train_state("decisionmoe_b", UPD, procs)
    jobs.append(("ÇEKİRDEK (≈4 sa): decisionmoe_b, latency, POPE", "eğitim decisionmoe_b (150 güncelleme)", st, UPD * SEC_PER_UPDATE_DEFAULT))
    jobs.append(("ÇEKİRDEK (≈4 sa): decisionmoe_b, latency, POPE", "test değerlendirmesi decisionmoe_b",
                 eval_state("decisionmoe_b", procs), EVAL_SEC))
    simple("ÇEKİRDEK (≈4 sa): decisionmoe_b, latency, POPE", "kontrollü latency benchmark", (RES / "latency_benchmark.json").exists(),
           "benchmark_latency.py" in procs, LATENCY_SEC)
    for v in ("standard_topk", "standard_topk_k1", "decisionmoe_b"):
        simple("ÇEKİRDEK (≈4 sa): decisionmoe_b, latency, POPE", f"POPE {v}", (RES / f"pope_{v}.json").exists(),
               "evaluate_pope.py" in procs, POPE_SEC)
    for v in ("decisionmoe_a", "ppo_direct_dynk"):
        t = train_state(v, UPD, procs)
        jobs.append(("OPSİYONEL (çekirdek sonrası ≈4 sa)", f"eğitim {v}", t, UPD * SEC_PER_UPDATE_DEFAULT))
    for v in ("decisionmoe_a", "ppo_direct_dynk"):
        jobs.append(("OPSİYONEL (çekirdek sonrası ≈4 sa)", f"değerlendirme {v}", eval_state(v, procs), EVAL_SEC))
    done_base = [v for v in MAIN_EVAL[:7] if (RES / f"eval_{v}_test.json").exists()]

    core = [j for j in jobs if j[0].startswith("ÇEKİRDEK")]
    def pct(js):
        tw = sum(w for *_, w in js); dw = sum(w * min(1.0, j["frac"]) for _, _, j, w in js)
        return dw / tw, sum(j["remaining"] for _, _, j, _ in js if j["frac"] < 1.0)
    cp, crem = pct(core)
    ap, arem = pct(jobs)
    now = datetime.now()
    out = [f"\033[1mDecisionMoE hat ilerlemesi\033[0m   {now:%H:%M:%S}", ""]
    out.append(f"ÇEKİRDEK {bar(cp, 36)} {100 * cp:5.1f}%   kalan ≈ {fmt_dur(crem)}   bitiş ≈ {now + timedelta(seconds=crem):%d.%m %H:%M}")
    out.append(f"TÜMÜ     {bar(ap, 36)} {100 * ap:5.1f}%   kalan ≈ {fmt_dur(arem)}   bitiş ≈ {now + timedelta(seconds=arem):%d.%m %H:%M}")
    out.append(f"\nBitmiş baseline/fixed-K değerlendirmeleri: {len(done_base)}/7 ✓")
    last = None
    for group, label, j, _ in jobs:
        if group != last:
            out += ["", f"\033[1m{group}\033[0m"]
            last = group
        icon = "✓" if j["frac"] >= 1.0 else ("▶" if j["running"] else "·")
        extra = j.get("extra", "")
        if "step" in j:
            extra = f"{j['step']}/{j['target']} güncelleme {extra}"
        out.append(f"  {icon} {label:36s} {bar(j['frac'])} {100 * min(1.0, j['frac']):5.1f}%  {extra}")
    out += ["", "✓ bitti   ▶ çalışıyor   · sırada"]
    return "\n".join(out)


def main() -> None:
    if "--watch" in sys.argv:
        try:
            while True:
                print("\033[2J\033[H" + render(), flush=True)
                time.sleep(10)
        except KeyboardInterrupt:
            pass
    else:
        print(render())


if __name__ == "__main__":
    main()
