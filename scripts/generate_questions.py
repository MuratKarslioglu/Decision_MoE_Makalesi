"""Soru üretimi: her görüntü için yerel Qwen2.5-VL ile 5 soru (T1-T5) üretir.

Çıktı: raw/generated_raw.json (veya --out) — filtrelenmemiş ham kayıtlar.
Prompt, Dataset Pipeline dokümanı §4.2'deki ile birebir aynıdır (değiştirilmez).
Her görüntü için kalite kurallarını (qc_rules.py) geçen adaylar havuzda biriktirilir; eksik tipi olan
görüntüler daha yüksek sıcaklıkla yeniden üretilir (--rounds). Resume destekler.
"""
import argparse
import json
import random
import re
from pathlib import Path

import torch
import qc_rules
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "decisionmoe_dataset"
DEFAULT_OUT = DATASET_DIR / "raw" / "generated_raw.json"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
SOURCES = ["coco", "ai2d", "gqa", "textvqa"]
VALID_TYPES = ["T1", "T2", "T3", "T4", "T5"]
MAX_PIXELS = 640 * 28 * 28  # 8GB VRAM için görüntü token sayısını sınırlar
MAX_NEW_TOKENS = 600
SEED = 42

PROMPT = """You are a dataset generation assistant for a multimodal AI research project.

Given the image, generate exactly 5 questions in the following types.
Each question must be answerable from the image alone.
Output must be valid JSON only — no explanation, no markdown, no preamble.

Question types:
- T1 (Visual Perception): Ask about something directly visible in the image.
  Example: "How many people are in this image?"
- T2 (Spatial Reasoning): Ask about the spatial relationship between two objects.
  Example: "Is the red car to the left or right of the bridge?"
- T3 (Multi-hop QA): Ask a two-step inference question using visual information.
  Example: "What color is the object on the table, and what does that color typically represent?"
- T4 (Hallucination Trap): Ask about something that is NOT in the image.
  The correct answer must be "no" or "none" or similar negation.
  Example: "Is there a dog in this image?" (only valid if there is no dog)
- T5 (OCR + Reasoning): Ask about text visible in the image and apply reasoning.
  Example: "Does the sign show a price above 10 dollars?"
  (Only generate T5 if the image contains readable text. If not, generate a second T3 instead.)

Output format (strict JSON array):
[
  {"question_type": "T1", "question": "...", "answer": "..."},
  {"question_type": "T2", "question": "...", "answer": "..."},
  {"question_type": "T3", "question": "...", "answer": "..."},
  {"question_type": "T4", "question": "...", "answer": "..."},
  {"question_type": "T5", "question": "...", "answer": "..."}
]"""


def list_images(per_source: dict[str, int] | None, seed: int) -> list[tuple[str, Path]]:
    """(source, image_path) listesi; per_source verilirse her kaynaktan rastgele o kadar alır."""
    items = []
    rng = random.Random(seed)
    for source in SOURCES:
        files = sorted((DATASET_DIR / "images" / source).glob("*"))
        files = [f for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if per_source is not None:
            files = rng.sample(files, min(per_source.get(source, 0), len(files)))
        items += [(source, f) for f in files]
    return items


def parse_questions(text: str) -> list[dict]:
    """Model çıktısından JSON dizisini çıkarır; geçerli şemalı öğeleri döndürür (yoksa ValueError)."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError("JSON dizisi bulunamadı")
    items = json.loads(match.group(0))
    if not isinstance(items, list):
        raise ValueError("JSON dizisi değil")
    good = [
        q for q in items
        if isinstance(q, dict) and q.get("question_type") in VALID_TYPES
        and str(q.get("question", "")).strip() and str(q.get("answer", "")).strip()
    ]
    if not good:
        raise ValueError("geçerli soru yok")
    return good


def add_to_pool(entry: dict, items: list[dict]) -> None:
    """Kalite kurallarından geçen adayları görüntünün havuzuna ekler (aynı soru tekrar eklenmez)."""
    for q in items:
        qtype, question, answer = q["question_type"], str(q["question"]).strip(), str(q["answer"]).strip()
        ok, _ = qc_rules.check(qtype, question, answer)
        bucket = entry["pool"].setdefault(qtype, [])
        if ok and all(qc_rules.normalize(question) != qc_rules.normalize(c["question"]) for c in bucket):
            bucket.append({"question": question, "answer": answer})


def is_complete(entry: dict) -> bool:
    """T1-T4 için birer aday + (T5 adayı veya ikinci bir T3 adayı) varsa görüntü tamamdır."""
    pool = entry["pool"]
    if not all(pool.get(t) for t in ("T1", "T2", "T3", "T4")):
        return False
    return bool(pool.get("T5")) or len(pool["T3"]) >= 2


def build_records(entry: dict, source: str, image_path: Path) -> list[dict]:
    """Havuzdan sabit JSON formatında kayıtlar üretir; T5 yoksa ikinci T3 -> id sonu _T3_2."""
    pool = entry["pool"]
    picks = [(t, pool[t][0]) for t in ("T1", "T2", "T3", "T4") if pool.get(t)]
    if pool.get("T5"):
        picks.append(("T5", pool["T5"][0]))
    elif len(pool.get("T3", [])) >= 2:
        picks.append(("T3_2", pool["T3"][1]))
    records = []
    for tag, q in picks:
        records.append({
            "id": f"{source}_{image_path.stem}_{tag}",
            "image_id": image_path.stem,
            "image_path": f"images/{source}/{image_path.name}",
            "source": source,
            "question_type": tag.split("_")[0],
            "question": q["question"],
            "answer": q["answer"],
            "language": "en",
        })
    return records


def generate_batch(model, processor, image_paths: list[Path], temperature: float = 0.4) -> list[str]:
    """Birden çok görüntü için tek forward ile ham model çıktılarını döndürür (sol padding)."""
    images = [Image.open(p).convert("RGB") for p in image_paths]
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text] * len(images), images=images, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=temperature, top_p=0.9)
    return [t.strip() for t in processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--pilot", action="store_true", help="10 görüntülük pilot (3 coco, 3 ai2d, 2 gqa, 2 textvqa)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3, help="eksik tipi olan görüntüler için toplam üretim turu")
    ap.add_argument("--retry-temperature", type=float, default=0.8)
    ap.add_argument("--load-4bit", action="store_true", help="8GB GPU için NF4 kuantizasyon (bitsandbytes)")
    ap.add_argument("--load-8bit", action="store_true", help="8GB GPU için int8 kuantizasyon (bitsandbytes)")
    args = ap.parse_args()

    per_source = {"coco": 3, "ai2d": 3, "gqa": 2, "textvqa": 2} if args.pilot else None
    images = list_images(per_source, SEED)
    if args.limit:
        images = images[: args.limit]

    state = {"images": {}, "records": [], "failed": []}
    if args.out.exists():
        state = json.loads(args.out.read_text(encoding="utf-8"))

    processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=MAX_PIXELS)
    processor.tokenizer.padding_side = "left"
    quant = None
    if args.load_8bit:
        from transformers import BitsAndBytesConfig  # bitsandbytes gerektirir
        quant = BitsAndBytesConfig(load_in_8bit=True)
    elif args.load_4bit:
        from transformers import BitsAndBytesConfig  # bitsandbytes gerektirir
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa",
        quantization_config=quant,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.out.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")

    meta = {f"{s}/{p.name}": (s, p) for s, p in images}
    for key in meta:
        state["images"].setdefault(key, {"pool": {}, "attempts": 0, "complete": False, "raws": []})

    def rebuild_records() -> None:
        state["records"] = [
            r for key, (s, p) in meta.items() for r in build_records(state["images"][key], s, p)
        ]

    for rnd in range(1, args.rounds + 1):
        todo = [k for k in meta if state["images"][k]["attempts"] < rnd and not state["images"][k]["complete"]]
        temperature = 0.4 if rnd == 1 else args.retry_temperature
        with tqdm(total=len(todo), desc=f"tur {rnd}/{args.rounds}", unit="img") as bar:
            for i in range(0, len(todo), args.batch_size):
                chunk = todo[i : i + args.batch_size]
                raws = generate_batch(model, processor, [meta[k][1] for k in chunk], temperature)
                for key, raw in zip(chunk, raws):
                    entry = state["images"][key]
                    entry["attempts"] += 1
                    entry["raws"].append(raw)
                    try:
                        add_to_pool(entry, parse_questions(raw))
                    except Exception as e:  # noqa: BLE001 - sonraki turda tekrar denenir
                        entry["last_error"] = f"{type(e).__name__}: {e}"
                    entry["complete"] = is_complete(entry)
                rebuild_records()
                save()
                bar.update(len(chunk))
                done_n = sum(v["complete"] for v in state["images"].values())
                bar.set_postfix(tam=done_n, kayit=len(state["records"]))
    state["failed"] = [
        {"image": k, "error": v.get("last_error", "eksik soru tipi"), "pool_types": sorted(v["pool"])}
        for k, v in state["images"].items() if not v["complete"]
    ]
    save()
    n = len(state["images"])
    print(f"görüntü: {n}, tam: {n - len(state['failed'])}, eksik: {len(state['failed'])}, kayıt: {len(state['records'])}")


if __name__ == "__main__":
    main()
