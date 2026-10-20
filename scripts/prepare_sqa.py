"""ScienceQA-IMG (lmms-lab/ScienceQA-IMG, çoktan seçmeli) -> sqa_data/{sqa.json, images/}. Kullanım: venv/bin/python scripts/prepare_sqa.py
Yalnızca 2-4 şıklı sorular (A-D). Resmi bölmeler korunur; val ve test sabit tohumla AI2D ile aynı boyutlara indirilir (val 341, test 597), train 2150'ye."""
import hashlib
import io
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "sqa_data"
(ROOT / "images").mkdir(parents=True, exist_ok=True)
SIZE = {"train": 2150, "val": 341, "test": 597}
out, saved = [], {}
for hf, split in (("train", "train"), ("validation", "val"), ("test", "test")):
    rows = pq.read_table(hf_hub_download("lmms-lab/ScienceQA-IMG", f"data/{hf}-00000-of-00001.parquet", repo_type="dataset")).to_pylist()
    rows = [(i, r) for i, r in enumerate(rows) if 2 <= len(r["choices"]) <= 4 and r["image"]]
    random.Random(0).shuffle(rows)
    for i, r in rows[:SIZE[split]]:
        h = hashlib.md5(r["image"]["bytes"]).hexdigest()
        if h not in saved:
            p = f"images/{h}.jpg"
            Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(ROOT / p, quality=92)
            saved[h] = p
        q = (r.get("hint") or "").strip()
        out.append({"id": f"sqa_{split}_{i}", "image_path": saved[h], "question": (q + "\n" if q else "") + r["question"],
                    "options": list(r["choices"]), "answer": "ABCD"[int(r["answer"])], "split": split})
(ROOT / "sqa.json").write_text(json.dumps(out))
import collections  # noqa: E402
print(len(out), len(saved), collections.Counter(o["split"] for o in out))
