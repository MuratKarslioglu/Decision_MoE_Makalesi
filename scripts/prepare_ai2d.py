"""AI2D (lmms-lab/ai2d, çoktan seçmeli, resmi test) -> ai2d_data/{ai2d.json, images/}. Kullanım: venv/bin/python scripts/prepare_ai2d.py
Tüm 3088 soru; bölme GÖRÜNTÜ özetine göre (aynı diyagram tek bölümde): test ~600, val ~300, kalanı train. Kayıt: id, image_path, question, options, answer(A-D), split."""
import hashlib
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "ai2d_data"
(ROOT / "images").mkdir(parents=True, exist_ok=True)
rows = []
for part in ("00000", "00001"):
    rows += pq.read_table(hf_hub_download("lmms-lab/ai2d", f"data/test-{part}-of-00002.parquet", repo_type="dataset")).to_pylist()
out, saved = [], {}
for i, r in enumerate(rows):
    h = hashlib.md5(r["image"]["bytes"]).hexdigest()
    if h not in saved:
        p = f"images/{h}.jpg"
        Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(ROOT / p, quality=92)
        saved[h] = p
    b = int(h[:8], 16) % 100                                    # görüntü özetinden deterministik bölme
    out.append({"id": f"ai2d_{i}", "image_path": saved[h], "question": r["question"], "options": list(r["options"]),
                "answer": "ABCD"[int(r["answer"])], "split": "test" if b < 20 else "val" if b < 30 else "train"})
(ROOT / "ai2d.json").write_text(json.dumps(out))
import collections  # noqa: E402
print(len(out), len(saved), collections.Counter(o["split"] for o in out))
