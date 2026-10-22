"""ScienceQA-IMG için ikinci, bağımsız test dilimi ('test2'): resmi test bölmesinden, ana 'test' (597) ile çakışmayan 597 soru.
prepare_sqa.py ile aynı süzme ve aynı sabit karıştırma; ilk 597 'test' için kullanıldı, sonraki 597 burada. Politikalar bu dilimi hiç görmedi.
Kullanım: venv/bin/python scripts/prepare_sqa_holdout.py"""
import hashlib
import io
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "sqa_data"
data = [d for d in json.loads((ROOT / "sqa.json").read_text()) if d["split"] != "test2"]
rows = pq.read_table(hf_hub_download("lmms-lab/ScienceQA-IMG", "data/test-00000-of-00001.parquet", repo_type="dataset")).to_pylist()
rows = [(i, r) for i, r in enumerate(rows) if 2 <= len(r["choices"]) <= 4 and r["image"]]
random.Random(0).shuffle(rows)
used = {d["id"] for d in data if d["split"] == "test"}
assert all(f"sqa_test_{i}" in used for i, _ in rows[:597]), "ana test dilimi yeniden üretilemedi"
new = []
for i, r in rows[597:1194]:
    h = hashlib.md5(r["image"]["bytes"]).hexdigest()
    p = f"images/{h}.jpg"
    if not (ROOT / p).exists():
        Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(ROOT / p, quality=92)
    q = (r.get("hint") or "").strip()
    new.append({"id": f"sqa_test2_{i}", "image_path": p, "question": (q + "\n" if q else "") + r["question"],
                "options": list(r["choices"]), "answer": "ABCD"[int(r["answer"])], "split": "test2"})
(ROOT / "sqa.json").write_text(json.dumps(data + new))
print(len(new), "test2 sorusu; test ile kesişim:", len({d['id'] for d in new} & used))
