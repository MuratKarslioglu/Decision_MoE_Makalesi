"""Gerçek duvar saati gecikme: sabit Top-K, asimetrik (K_prefill,K_decode) ve öğrenilmiş politika. Kullanım: DS_TASK=sqa venv_ds/bin/python ds/latency.py [N=40] [politikalar...]
Her yapılandırma aynı N test sorusunda; ilk 3 soru ısınma. prefill = max_new=1 süresi; decode ms/token = (tam üretim - prefill)/(üretilen token - 1).
Çıktı: experiments/results/ds_latency_<task>.json. Eşzamanlama torch.cuda.synchronize ile; politika yükü (özellik+MLP) dyn satırlarına dahildir."""
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ds"))
import dsloader  # noqa: E402
from dsrouter import DSRouter  # noqa: E402
from dynk import DynKPolicy  # noqa: E402

import task  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
POLS = sys.argv[2:]
OUT = ROOT / "experiments" / "results" / f"ds_latency_{task.TASK}.json"


def timed(proc, model, it, folder, max_new):
    opts = "\n".join(f"{'ABCD'[i]}. {o}" for i, o in enumerate(it["options"]))
    prompt = f"{it['question']}\n{opts}{task.COT}"
    torch.cuda.synchronize()
    t = time.perf_counter()
    txt = task.generate(proc, model, ROOT / folder / it["image_path"], prompt, max_new=max_new)
    torch.cuda.synchronize()
    return time.perf_counter() - t, len(proc.tokenizer.encode(txt))


def main() -> None:
    proc, model = dsloader.load_ds()
    items, _ = task.get("test")
    folder = task.MCQ[task.TASK][0]
    items = items[:N + 3]
    cfgs = [(f"topk_k{k}", dict(mode="topk", k=k)) for k in (6, 5, 4, 3, 2, 1)]
    cfgs += [(f"st{a}_{b}", dict(mode="topk", k=a, k_dec=b)) for a, b in ((2, 4), (3, 4), (2, 6))]
    for p in POLS:
        pol = DynKPolicy().cuda()
        pol.load_state_dict(torch.load(ROOT / "artifacts" / f"ds_dynk_{p}.pt"))
        cfgs.append((f"dyn_{p}", dict(mode="dyn", policy=pol, deterministic=True)))
    res = {}
    for name, kw in cfgs:
        r = DSRouter(model, **kw)
        pre, full, ntok = [], [], []
        for i, it in enumerate(items):
            r.begin_episode()
            tp, _ = timed(proc, model, it, folder, 1)
            r.begin_episode()
            tf, n = timed(proc, model, it, folder, task.MAXNEW)
            if i >= 3:
                pre.append(tp); full.append(tf); ntok.append(n)
        r.close()
        dec = [(f - p) / max(1, n - 1) for f, p, n in zip(full, pre, ntok)]
        res[name] = {"n": len(full), "prefill_ms": 1e3 * st.mean(pre), "total_ms": 1e3 * st.mean(full), "decode_ms_per_tok": 1e3 * st.mean(dec),
                     "tokens": st.mean(ntok), "total_ms_sd": 1e3 * st.pstdev(full)}
        print(f"{name:12s} prefill={res[name]['prefill_ms']:.0f}ms total={res[name]['total_ms']:.0f}ms dec={res[name]['decode_ms_per_tok']:.1f}ms/tok tok={res[name]['tokens']:.0f}", flush=True)
        OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
