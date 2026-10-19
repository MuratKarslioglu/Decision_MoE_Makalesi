"""Uzun üretim (akıl yürütme zinciri) rejimi pilotu, AI2D çoktan seçmeli (resmi test, ai2d_data/). Soru: decode uzun olunca Top-K duyarlılığı
prefill/decode'a göre ayrışıyor mu? Doğruluk = son 'Answer: X' harfi. FLOPs token-ağırlıklı (prefill + decode token sayısı).
Kullanım: venv_ds/bin/python ds/cot_pilot.py [n=100] [max_new=128]   Çıktı: experiments/results/ds_cot_pilot.json"""
import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ds"))
import dsloader  # noqa: E402
from dsrouter import DSRouter  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
MAXNEW = int(sys.argv[2]) if len(sys.argv) > 2 else 128
OUT = ROOT / "experiments" / "results" / "ds_cot_pilot.json"
COT = "\nThink step by step briefly, then finish with 'Answer: <letter>'."
CONFIGS = [("k6", dict(mode="topk", k=6)), ("k3", dict(mode="topk", k=3)), ("k2", dict(mode="topk", k=2)), ("k1", dict(mode="topk", k=1)),
           ("kp6_kd3", dict(mode="topk", k=6, k_dec=3)), ("kp6_kd2", dict(mode="topk", k=6, k_dec=2)), ("kp6_kd1", dict(mode="topk", k=6, k_dec=1)),
           ("topp_t0.5", dict(mode="topp", k=6, tau=0.5)), ("topp_t0.3", dict(mode="topp", k=6, tau=0.3))]


def parse(txt: str) -> str:
    m = re.findall(r"Answer:\s*\(?([A-D])\b", txt)
    if m:
        return m[-1]
    m = re.findall(r"\b([A-D])\b", txt)
    return m[-1] if m else ""


def generate(proc, model, img_path, prompt):
    img = Image.open(img_path).convert("RGB")
    conv = [{"role": "<|User|>", "content": f"<image>\n{prompt}", "images": [img]}, {"role": "<|Assistant|>", "content": ""}]
    inp = proc(conversations=conv, images=[img], force_batchify=True, system_prompt="").to(model.device)
    tok = proc.tokenizer
    with torch.no_grad():
        emb = model.prepare_inputs_embeds(**inp)
        out = model.language.generate(inputs_embeds=emb, attention_mask=inp.attention_mask, pad_token_id=tok.eos_token_id,
                                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, max_new_tokens=MAXNEW,
                                      do_sample=False, use_cache=True)
    return tok.decode(out[0].cpu().tolist(), skip_special_tokens=True), out.shape[1]


def main() -> None:
    proc, model = dsloader.load_ds()
    data = json.loads((ROOT / "ai2d_data" / "ai2d.json").read_text())[:N]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, kw in CONFIGS:
        if name in res:
            continue
        r = DSRouter(model, **kw)
        ok, ntok, t0 = [], [], time.time()
        for it in data:
            opts = "\n".join(f"{'ABCD'[i]}. {o}" for i, o in enumerate(it["options"]))
            txt, n = generate(proc, model, ROOT / "ai2d_data" / it["image_path"], f"{it['question']}\n{opts}{COT}")
            ok.append(int(parse(txt) == it["answer"])); ntok.append(n)
        fl = r.ksum / (r.kcnt * 6)
        r.close()
        res[name] = {"ids": [d["id"] for d in data], "correct": ok, "accuracy": sum(ok) / len(ok), "flops": fl,
                     "mean_new_tokens": sum(ntok) / len(ntok), "seconds": time.time() - t0, "max_new": MAXNEW}
        print(f"{name:10s} acc={res[name]['accuracy']:.3f} flops={fl:.3f} newtok={res[name]['mean_new_tokens']:.0f} {res[name]['seconds']:.0f}s", flush=True)
        OUT.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
