"""Teslim kontrol listesi (Dataset Pipeline §10): dataset klasörünü doğrular, hataları listeler.

Kullanım: python scripts/validate_dataset.py [--root decisionmoe_dataset]. Hata varsa çıkış kodu 1.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["id", "image_id", "image_path", "source", "question_type", "question", "answer", "language"]
SOURCES = {"coco", "ai2d", "gqa", "textvqa"}
TYPES = {"T1", "T2", "T3", "T4", "T5"}
BALANCE_TOLERANCE = 0.05


def validate(root: Path) -> list[str]:
    errors: list[str] = []
    for src in SOURCES:
        if not (root / "images" / src).is_dir():
            errors.append(f"images/{src} klasörü yok")
    qpath = root / "questions.json"
    if not qpath.exists():
        return errors + ["questions.json yok"]
    qs = json.loads(qpath.read_text(encoding="utf-8"))
    ids = [q.get("id") for q in qs]
    if len(ids) != len(set(ids)):
        errors.append(f"tekrar eden id: {len(ids) - len(set(ids))}")
    for q in qs:
        if list(q) != FIELDS:
            errors.append(f"{q.get('id')}: alanlar/sıra hatalı {list(q)}")
        elif q["question_type"] not in TYPES or q["source"] not in SOURCES or q["language"] != "en":
            errors.append(f"{q['id']}: geçersiz tip/kaynak/dil")
        elif not (root / q["image_path"]).exists():
            errors.append(f"{q['id']}: görüntü yok {q['image_path']}")
    if not (root / "raw" / "generated_raw.json").exists():
        errors.append("raw/generated_raw.json yok")

    split_ids: dict[str, set[str]] = {}
    split_imgs: dict[str, set[tuple]] = {}
    total = 0
    for name in ("train", "val", "test"):
        p = root / "splits" / f"{name}.json"
        if not p.exists():
            errors.append(f"splits/{name}.json yok")
            continue
        recs = json.loads(p.read_text(encoding="utf-8"))
        total += len(recs)
        split_ids[name] = {r["id"] for r in recs}
        split_imgs[name] = {(r["source"], r["image_id"]) for r in recs}
        tc = Counter(r["question_type"] for r in recs)
        full = Counter(q["question_type"] for q in qs)
        for t in TYPES:
            if full[t] and abs(tc[t] / len(recs) - full[t] / len(qs)) > BALANCE_TOLERANCE:
                errors.append(f"{name}: {t} dağılımı ±%5 dışında")
    if total != len(qs):
        errors.append(f"split toplamı {total} != questions.json {len(qs)}")
    names = list(split_ids)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if split_ids[a] & split_ids[b]:
                errors.append(f"id sızıntısı: {a} ∩ {b}")
            if split_imgs[a] & split_imgs[b]:
                errors.append(f"görüntü sızıntısı: {a} ∩ {b} ({len(split_imgs[a] & split_imgs[b])} görüntü)")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT / "decisionmoe_dataset")
    args = ap.parse_args()
    errors = validate(args.root)
    if errors:
        print(f"{len(errors)} hata:")
        for e in errors[:50]:
            print(" -", e)
        sys.exit(1)
    print("dataset doğrulaması geçti")


if __name__ == "__main__":
    main()
