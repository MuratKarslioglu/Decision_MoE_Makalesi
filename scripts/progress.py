"""Arka plandaki eğitim/değerlendirme işini ilerleme çubuğuyla izler (yalnızca standart kütüphane).
Kullanım: python3 scripts/progress.py [yenileme_sn=10]    Çıkış: Ctrl+C. Çalışan ds/train_dynk.py ve ds/eval_test.py süreçlerini /proc üzerinden bulur."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = 30


def bar(done: float, total: float) -> str:
    f = min(1.0, done / total) if total else 0.0
    n = int(f * W)
    return f"[{'█' * n}{'░' * (W - n)}] {100 * f:5.1f}%  ({done:g}/{total:g})"


def procs():
    out = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            args = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
            if len(args) > 2 and args[1].endswith(("train_dynk.py", "eval_test.py")):
                env = dict(e.split("=", 1) for e in Path(f"/proc/{pid}/environ").read_bytes().decode().split("\0") if "=" in e)
                out.append((args[1].split("/")[-1], args[2:], env.get("DS_TASK", "synth")))
        except (OSError, ValueError):
            continue
    return out


def fmt(s: float) -> str:
    return f"{int(s // 3600)}s {int(s % 3600 // 60)}dk" if s >= 3600 else f"{int(s // 60)}dk {int(s % 60)}sn"


def render() -> str:
    lines = [time.strftime("%H:%M:%S"), ""]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    lines.append(f"GPU: {gpu or 'yok'}")
    ps = procs()
    if not ps:
        lines.append("Çalışan eğitim/değerlendirme süreci yok.")
    for script, a, tsk in ps:
        if script == "train_dynk.py":
            name, upd = a[0], int(a[2]) if len(a) > 2 else 60
            lg = ROOT / "logs" / f"ds_train_{name}.json"
            log = json.loads(lg.read_text()) if lg.exists() else []
            u = len(log)
            eta = (log[-1]["t"] / u) * (upd - u) if u else 0
            last = f" | son acc {log[-1]['acc']:.2f} K_prefill {log[-1]['k_prefill']:.1f}" if u else ""
            lines += ["", f"EĞİTİM {name} [{tsk}]", "  " + bar(u, upd) + f"  kalan ≈ {fmt(eta)}" + last]
        else:
            split, pols = a[0], a[1:]
            names = [f"topk_k{k}" for k in (6, 5, 4, 3, 2, 1)] + [p if p.startswith(("topp", "st")) else f"dyn_{p}" for p in pols]
            out = ROOT / "experiments/results" / (f"ds_eval_{split}.json" if tsk == "synth" else f"ds_eval_{tsk}_{split}.json")
            res = json.loads(out.read_text()) if out.exists() else {}
            n = len(next(iter(res.values()))["correct"]) if res else 597
            done = [x for x in names if x in res]
            cur = next((x for x in names if x not in res), None)
            part = 0
            if cur:
                ck = ROOT / "logs" / f"evalckpt_{tsk}_{split}_{cur}.json"
                part = len(json.loads(ck.read_text())["ok"]) if ck.exists() else 0
            lines += ["", f"DEĞERLENDİRME {tsk}/{split}", "  genel:  " + bar(len(done), len(names)) + f"  şu an: {cur or '-'}",
                      "  mevcut: " + bar(part, n) + "  (20 soruda bir güncellenir)"]
    return "\n".join(lines)


if __name__ == "__main__":
    dt = float(sys.argv[1]) if len(sys.argv) > 1 else 10
    try:
        while True:
            print("\033[2J\033[H" + render(), flush=True)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
