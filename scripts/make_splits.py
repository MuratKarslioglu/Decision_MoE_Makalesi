"""Stratified split (Dataset Pipeline §6): questions.json -> splits/{train,val,test}.json (%70/%15/%15).

Varsayılan: GÖRÜNTÜ BAZLI bölme. Aynı görüntüden gelen 5 soru aynı split'te kalır; böylece test
görüntüleri eğitimde hiç görülmez (görüntü sızıntısı yok). Her kaynak içinde görüntüler 70/15/15 bölünür,
birçok seed denenip (source, question_type) oranlarını hedefe en yakın bölme seçilir.
--question-level: görüntüye bakmadan her (source, type) grubunu doğrudan 70/15/15 böler.
Ayrıca boş routing_oracle.json şablonu (yoksa) oluşturur.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "decisionmoe_dataset"
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
N_TRIALS = 300


def split_list(items: list, rng: random.Random) -> dict[str, list]:
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    n_train, n_val = round(n * RATIOS["train"]), round(n * RATIOS["val"])
    return {"train": items[:n_train], "val": items[n_train:n_train + n_val], "test": items[n_train + n_val:]}


def deviation(splits: dict[str, list[dict]]) -> float:
    """(source,type) gruplarının split payları ile hedef oranlar arasındaki en büyük sapma."""
    totals = Counter((r["source"], r["question_type"]) for s in splits.values() for r in s)
    worst = 0.0
    for name, recs in splits.items():
        c = Counter((r["source"], r["question_type"]) for r in recs)
        for g, tot in totals.items():
            if tot >= 10:
                worst = max(worst, abs(c[g] / tot - RATIOS[name]))
    return worst


def image_level_split(questions: list[dict], seed: int) -> dict[str, list[dict]]:
    by_source_image: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for q in questions:
        by_source_image[q["source"]][q["image_id"]].append(q)
    best, best_dev = None, 1e9
    for trial in tqdm(range(N_TRIALS), desc="split denemesi", unit="seed"):
        rng = random.Random(seed + trial)
        splits: dict[str, list[dict]] = {k: [] for k in RATIOS}
        for _source, images in by_source_image.items():
            parts = split_list(sorted(images), rng)
            for name, image_ids in parts.items():
                for iid in image_ids:
                    splits[name] += images[iid]
        dev = deviation(splits)
        if dev < best_dev:
            best, best_dev = splits, dev
    return best


def question_level_split(questions: list[dict], seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for q in questions:
        groups[(q["source"], q["question_type"])].append(q)
    splits: dict[str, list[dict]] = {k: [] for k in RATIOS}
    for _, recs in sorted(groups.items()):
        for name, part in split_list(recs, rng).items():
            splits[name] += part
    return splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=DATASET_DIR / "questions.json")
    ap.add_argument("--out-dir", type=Path, default=DATASET_DIR / "splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--question-level", action="store_true")
    args = ap.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    splits = question_level_split(questions, args.seed) if args.question_level else image_level_split(questions, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, recs in splits.items():
        recs.sort(key=lambda r: r["id"])
        (args.out_dir / f"{name}.json").write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")

    n = len(questions)
    print(f"toplam {n} soru | mod: {'soru bazlı' if args.question_level else 'görüntü bazlı'} | en büyük sapma: {deviation(splits):.3f}")
    for name, recs in splits.items():
        tc = Counter(r["question_type"] for r in recs)
        print(f"  {name:5} {len(recs):5} ({len(recs) / n:.1%}) tipler: " + " ".join(f"{t}={tc[t]}" for t in sorted(tc)))

    oracle = args.questions.parent / "routing_oracle.json"
    if not oracle.exists():
        oracle.write_text("{}\n", encoding="utf-8")
        print("routing_oracle.json boş şablon oluşturuldu (ana geliştirici dolduracak)")


if __name__ == "__main__":
    main()
