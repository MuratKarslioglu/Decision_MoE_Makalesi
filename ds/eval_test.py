"""DeepSeek-VL2 test değerlendirmesi: Top-K tabanları + Dynamic-K politikaları. Kullanım: venv_ds/bin/python ds/eval_test.py <split=test> <politika adları...>
Çıktı: experiments/results/ds_eval_<split>.json (soru bazında correct; token-ağırlıklı expert FLOPs payı; ortalama K)"""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ds"))
import dsloader  # noqa: E402
from dsrouter import DSRouter  # noqa: E402
from dynk import DynKPolicy, flops_fraction  # noqa: E402

import task  # noqa: E402

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
POLS = sys.argv[2:]
OUT = ROOT / "experiments" / "results" / (f"ds_eval_{SPLIT}.json" if task.TASK == "synth" else f"ds_eval_{task.TASK}_{SPLIT}.json")


def main() -> None:
    proc, model = dsloader.load_ds()
    items, run = task.get(SPLIT)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    runs = [(f"topk_k{k}", dict(mode="topk", k=k)) for k in ((6, 4, 3, 2, 1) if task.TASK == "synth" else (6, 5, 4, 3, 2, 1))]
    for p in POLS:
        if p.startswith("topp"):                                       # örn. topp0.5: token düzeyi uyarlanır K sezgiseli
            runs.append((p, dict(mode="topp", k=6, tau=float(p[4:]))))
            continue
        pol = DynKPolicy().cuda()
        pol.load_state_dict(torch.load(ROOT / "artifacts" / f"ds_dynk_{p}.pt"))
        runs.append((f"dyn_{p}", dict(mode="dyn", policy=pol, deterministic=True)))
    for name, kw in runs:
        if name in res:
            continue
        r = DSRouter(model, **kw)
        ok, fl, kp, kd, t0 = [], [], [], [], time.time()
        for it in items:
            r.begin_episode()
            ok.append(run(proc, model, it))
            if kw["mode"] == "dyn":
                fl.append(flops_fraction(r.steps)); kp.append(r.steps[0]["k"]); kd += [s["k"] for s in r.steps[1:]]
        r.close()
        k0 = kw.get("k") if kw["mode"] == "topk" else None
        if kw["mode"] == "topp":
            fl = [r.ksum / (r.kcnt * 6)]
            kp = kd = [r.ksum / r.kcnt]
        res[name] = {"ids": [it["id"] for it in items], "correct": ok, "accuracy": sum(ok) / len(ok),
                     "flops": (k0 / 6) if k0 else sum(fl) / len(fl), "k_prefill": k0 or sum(kp) / len(kp),
                     "k_decode": k0 or (sum(kd) / max(1, len(kd))), "seconds": time.time() - t0}
        print(f"{name:14s} acc={res[name]['accuracy']:.3f} flops={res[name]['flops']:.3f} kP={res[name]['k_prefill']:.2f} "
              f"kD={res[name]['k_decode']:.2f} {res[name]['seconds']:.0f}s", flush=True)
        OUT.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
