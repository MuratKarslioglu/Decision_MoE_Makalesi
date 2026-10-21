"""A-OKVQA (HuggingFaceM4/A-OKVQA, 4 şıklı çoktan seçmeli, dış bilgi gerektiren) -> aok_data/{aok.json, images/}. Kullanım: venv/bin/python scripts/prepare_aok.py
train: resmi train'in ilk parçasından 2150 soru; val/test: resmi validation (1145) görüntü özetine göre bölünür (test 597, val 341; test etiketleri gizli olduğundan kullanılmaz)."""
import hashlib
import io
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "aok_data"
(ROOT / "images").mkdir(parents=True, exist_ok=True)
REPO = "HuggingFaceM4/A-OKVQA"
tr = pq.read_table(hf_hub_download(REPO, "data/train-00000-of-00002-c1d24de3bacb5e0c.parquet", repo_type="dataset")).to_pylist()
va = pq.read_table(hf_hub_download(REPO, "data/validation-00000-of-00001-b2bd0de231b6326a.parquet", repo_type="dataset")).to_pylist()
random.Random(0).shuffle(tr)
out, saved = [], {}


def add(r, split):
    h = hashlib.md5(r["image"]["bytes"]).hexdigest()
    if h not in saved:
        p = f"images/{h}.jpg"
        Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(ROOT / p, quality=92)
        saved[h] = p
    out.append({"id": f"aok_{r['question_id']}", "image_path": saved[h], "question": r["question"], "options": list(r["choices"]),
                "answer": "ABCD"[int(r["correct_choice_idx"])], "split": split})


for r in tr[:2150]:
    add(r, "train")
vt = [r for r in va if int(hashlib.md5(r["image"]["bytes"]).hexdigest()[:8], 16) % 100 < 60]
vv = [r for r in va if r not in vt]
for r in vt[:597]:
    add(r, "test")
for r in vv[:341]:
    add(r, "val")
(ROOT / "aok.json").write_text(json.dumps(out))
import collections  # noqa: E402
print(len(out), len(saved), collections.Counter(o["split"] for o in out))
