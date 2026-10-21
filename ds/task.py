"""Görev soyutlaması: DS_TASK=synth (varsayılan; sentetik kısa cevap) | ai2d | sqa | aok (AI2D / ScienceQA-IMG / A-OKVQA çoktan seçmeli, kısa akıl yürütme zinciri).
get(split) -> (items, run) ; run(proc, model, item) -> 0/1 doğru. Eğitim/değerlendirme betikleri yalnızca bunu kullanır."""
import json
import os
import re
from pathlib import Path

import dsloader
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TASK = os.environ.get("DS_TASK", "synth")
COT = "\nThink step by step briefly, then finish with 'Answer: <letter>'."
MAXNEW = 128
MCQ = {"ai2d": ("ai2d_data", "ai2d.json"), "sqa": ("sqa_data", "sqa.json"), "aok": ("aok_data", "aok.json")}


def parse(txt: str) -> str:
    m = re.findall(r"Answer:\s*\(?([A-D])\b", txt)
    if m:
        return m[-1]
    m = re.findall(r"\b([A-D])\b", txt)
    return m[-1] if m else ""


def generate(proc, model, img_path, prompt, max_new=MAXNEW) -> str:
    img = Image.open(img_path).convert("RGB")
    conv = [{"role": "<|User|>", "content": f"<image>\n{prompt}", "images": [img]}, {"role": "<|Assistant|>", "content": ""}]
    inp = proc(conversations=conv, images=[img], force_batchify=True, system_prompt="").to(model.device)
    tok = proc.tokenizer
    with torch.no_grad():
        emb = model.prepare_inputs_embeds(**inp)
        out = model.language.generate(inputs_embeds=emb, attention_mask=inp.attention_mask, pad_token_id=tok.eos_token_id,
                                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, max_new_tokens=max_new,
                                      do_sample=False, use_cache=True)
    return tok.decode(out[0].cpu().tolist(), skip_special_tokens=True)


def get(split: str):
    if TASK in MCQ:
        folder, fname = MCQ[TASK]
        items = [d for d in json.loads((ROOT / folder / fname).read_text()) if d["split"] == split]

        def run(proc, model, it) -> int:
            opts = "\n".join(f"{'ABCD'[i]}. {o}" for i, o in enumerate(it["options"]))
            txt = generate(proc, model, ROOT / folder / it["image_path"], f"{it['question']}\n{opts}{COT}")
            return int(parse(txt) == it["answer"])
        return items, run
    from src.data.dataset import DecisionMoEDataset
    from src.utils.metrics import is_correct
    ds = DecisionMoEDataset(split)

    def run_synth(proc, model, it) -> int:
        return int(is_correct(dsloader.answer(proc, model, ds.root / it["image_path"], it["question"]), it["answer"]))
    return ds.items, run_synth
