"""DeepSeek-VL2 POPE değerlendirmesi. Kullanım: venv_ds/bin/python ds/pope_ds.py <n_per_split> <politikalar: topk_k6 topk_k3 dyn_l015 ...>
Çıktı: experiments/results/ds_pope.json (politika başına alt küme metrikleri, soru bazında tahmin, FLOPs, K; ad bazında devam edilebilir)"""
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

from src.data.pope import POPE_ROOT, POPE_SPLITS, load_pope, parse_yes_no, pope_metrics  # noqa: E402

N = int(sys.argv[1])
OUT = ROOT / "experiments" / "results" / "ds_pope.json"


def main() -> None:
    proc, model = dsloader.load_ds()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in sys.argv[2:]:
        if name in res:
            continue
        if name.startswith("topk_k"):
            kw = dict(mode="topk", k=int(name[6:]))
        else:
            pol = DynKPolicy().cuda()
            pol.load_state_dict(torch.load(ROOT / "artifacts" / f"ds_dynk_{name[4:]}.pt"))
            kw = dict(mode="dyn", policy=pol, deterministic=True)
        r = DSRouter(model, **kw)
        out = {"n_per_split": N, "splits": {}, "items": []}
        fl, kp, kd, t0 = [], [], [], time.time()
        for split in POPE_SPLITS:
            items = load_pope(split, N)
            preds = []
            for it in items:
                r.begin_episode()
                a = dsloader.answer(proc, model, POPE_ROOT / it["image_path"], it["question"])
                preds.append(a)
                out["items"].append({"id": it["id"], "split": split, "label": it["answer"], "pred": a, "parsed": parse_yes_no(a)})
                if kw["mode"] == "dyn":
                    fl.append(flops_fraction(r.steps)); kp.append(r.steps[0]["k"]); kd += [s["k"] for s in r.steps[1:]]
            out["splits"][split] = pope_metrics(preds, [it["answer"] for it in items])
        r.close()
        out["overall"] = pope_metrics([x["pred"] for x in out["items"]], [x["label"] for x in out["items"]])
        k0 = kw.get("k")
        out["flops"] = (k0 / 6) if k0 else sum(fl) / len(fl)
        out["k_prefill"] = k0 or sum(kp) / len(kp)
        out["k_decode"] = k0 or sum(kd) / max(1, len(kd))
        out["seconds"] = time.time() - t0
        res[name] = out
        print(f"{name:12s} acc={out['overall']['accuracy']:.3f} f1={out['overall']['f1']:.3f} yes={out['overall']['yes_ratio']:.2f} "
              f"flops={out['flops']:.3f} unparsed={out['overall']['unparsed']:.3f} {out['seconds']:.0f}s", flush=True)
        OUT.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
