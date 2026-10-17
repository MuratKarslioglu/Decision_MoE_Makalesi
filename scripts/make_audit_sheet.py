"""Etiket denetimi için 200 soruluk tablo üretir (test split, 5 tipten 40'ar, seed 42).

Çıktı: experiments/results/audit_sheet.csv. İnsan doldurur:
  ref_correct   : referans cevap görüntüye göre doğru mu?  (y / n / ?)
  question_ok   : soru görüntüden cevaplanabilir ve belirsiz değil mi?  (y / n)
  model_ok      : model cevabı (model_answer) anlamca doğru mu?  (y / n)   [katı eşleşmeden bağımsız ek bilgi]
Doldurduktan sonra: python scripts/analyze_audit.py
"""
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
PER_TYPE = 40
SEED = 42


def main() -> None:
    qs = {x["id"]: x for x in json.load(open(ROOT / "decisionmoe_dataset" / "splits" / "test.json"))}
    preds = {i["id"]: i["pred"] for i in json.load(open(RES / "eval_standard_topk_test.json"))["items"]}
    rng = random.Random(SEED)
    rows = []
    for t in ("T1", "T2", "T3", "T4", "T5"):
        ids = sorted(i for i, q in qs.items() if q["question_type"] == t and i in preds)
        rows += rng.sample(ids, PER_TYPE)
    rng.shuffle(rows)
    out = RES / "audit_sheet.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["no", "id", "type", "image_path", "question", "reference_answer", "model_answer",
                    "ref_correct", "question_ok", "model_ok", "notes"])
        for n, i in enumerate(rows, 1):
            q = qs[i]
            w.writerow([n, i, q["question_type"], str(ROOT / "decisionmoe_dataset" / q["image_path"]),
                        q["question"], q["answer"], preds[i], "", "", "", ""])
    print(f"{len(rows)} soru -> {out}")


if __name__ == "__main__":
    main()
