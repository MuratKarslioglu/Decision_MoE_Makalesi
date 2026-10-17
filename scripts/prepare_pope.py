"""POPE (nesne varlığı hallucination benchmark'ı) indirir ve pope_data/ altına açar.

Çıktı: pope_data/images/<image_source>.jpg, pope_data/{random,popular,adversarial}.json
Kayıt biçimi dataset.py ile uyumlu: id, image_id, image_path, source, question_type, question, answer, language.
"""
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "pope_data"
SPLITS = ("random", "popular", "adversarial")


def main() -> None:
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        path = hf_hub_download("lmms-lab/POPE", f"Full/{split}-00000-of-00001.parquet", repo_type="dataset")
        records, seen = [], set()
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for r in batch.to_pylist():
                name = r["image_source"]
                if name not in seen:
                    Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(OUT / "images" / f"{name}.jpg", quality=95)
                    seen.add(name)
                records.append({
                    "id": f"pope_{split}_{r['question_id']}", "image_id": name, "image_path": f"images/{name}.jpg",
                    "source": "pope", "question_type": split, "question": r["question"],
                    "answer": r["answer"].strip().lower(), "language": "en"})
        (OUT / f"{split}.json").write_text(json.dumps(records, indent=1))
        yes = sum(x["answer"] == "yes" for x in records)
        print(f"{split:12s} {len(records)} soru, {len(seen)} görüntü, yes oranı {yes / len(records):.2f}")


if __name__ == "__main__":
    main()
