"""Kalite filtresi (Dataset Pipeline §5): raw/generated_raw.json -> questions.json.

Sırayla uygulanır:
  Kural 1  soru uzunluğu 5-40 kelime
  Kural 2  T4 doğrulaması (yalnız COCO): sorulan nesne COCO annotation'ında görüntüde YOKSA geçer
  Kural 3  T1-T3 cevapları yes/no/none/true/false olamaz (qc_rules.FORBIDDEN_ANSWERS_FOR)
  Kural 4  görüntü başına soru duplikasyonu (MiniLM kosinüs > 0.85 ise sonraki atılır)
Ek (dokümandaki kuralların üstüne): prompt-örnek sızıntısı, T4 cevabı olumsuz değilse ret, image_path varlığı.
Çıktı: questions.json (8 alanlı sabit format), metadata/filter_report.json, metadata/t4_manual_review_sample.json.
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import qc_rules
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "decisionmoe_dataset"
COCO_ANN = PROJECT_ROOT / "dataset_workspace" / "downloads" / "coco" / "annotations" / "instances_val2017.json"
DEDUP_MODEL = "all-MiniLM-L6-v2"
DEDUP_THRESHOLD = 0.85
MANUAL_REVIEW_SAMPLE = 60
TYPE_ORDER = ["T1", "T2", "T3", "T4", "T5"]
FIELDS = ["id", "image_id", "image_path", "source", "question_type", "question", "answer", "language"]

# COCO kategorileri için eş anlamlılar (basit keyword eşleşmesi)
SYNONYMS = {
    "person": ["people", "man", "men", "woman", "women", "boy", "girl", "child", "children", "kid", "kids",
               "player", "skateboarder", "surfer", "skier", "rider", "pedestrian", "baby", "lady"],
    "bicycle": ["bike", "bikes"],
    "motorcycle": ["motorbike", "motorbikes"],
    "airplane": ["plane", "planes", "aircraft", "jet"],
    "couch": ["sofa", "sofas"],
    "tv": ["television", "televisions", "tv"],
    "cell phone": ["phone", "phones", "smartphone", "mobile phone"],
    "dining table": ["table", "tables"],
    "potted plant": ["plant", "plants"],
    "sports ball": ["ball", "balls"],
    "wine glass": ["glass", "glasses"],
    "hot dog": ["hotdog"],
    "teddy bear": ["teddy"],
    "traffic light": ["stoplight"],
    "refrigerator": ["fridge"],
    "mouse": ["computer mouse"],
}
PREPOSITIONS = r"(?:on|in|at|near|next|behind|under|above|below|wearing|with|holding|beside|inside|outside|by|over|around|among)"
TARGET_RE = re.compile(rf"\b(?:is|are) there (?:a |an |any |some |the )?(.*?)(?:\b{PREPOSITIONS}\b|\?|$)", re.I)


def load_coco_categories(ann_path: Path) -> tuple[dict[int, set[str]], dict[str, str]]:
    """(image_id -> kategori adları kümesi, arama terimi -> kanonik kategori)."""
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in data["categories"]}
    image_cats: dict[int, set[str]] = defaultdict(set)
    for a in data["annotations"]:
        image_cats[a["image_id"]].add(names[a["category_id"]])
    terms = {n: n for n in names.values()}
    for canon, syns in SYNONYMS.items():
        for s in syns:
            terms.setdefault(s, canon)
    return image_cats, terms


def mentioned_categories(text: str, terms: dict[str, str]) -> set[str]:
    text = text.lower()
    found = set()
    for term, canon in terms.items():
        if re.search(rf"\b{re.escape(term)}(?:s|es)?\b", text):
            found.add(canon)
    return found


def validate_t4(question: str, image_cats: set[str], terms: dict[str, str]) -> str:
    """'ok' (nesne görüntüde yok), 'present' (var -> at), 'unverified' (COCO kategorisi tanınmadı)."""
    m = TARGET_RE.search(question)
    target = m.group(1) if m else question
    cats = mentioned_categories(target, terms)
    if not cats:
        return "unverified"
    return "present" if cats & image_cats else "ok"


def dedup_image_questions(records: list[dict], model, threshold: float) -> set[str]:
    """Atılacak id'ler. Aynı görüntüdeki sorular T1..T5 sırasıyla gezilir, öncekine benzeyen atılır."""
    from sentence_transformers import util
    recs = sorted(records, key=lambda r: (TYPE_ORDER.index(r["question_type"]), r["id"]))
    emb = model.encode([r["question"] for r in recs], convert_to_tensor=True, show_progress_bar=False)
    sim = util.cos_sim(emb, emb)
    kept: list[int] = []
    dropped = set()
    for i, r in enumerate(recs):
        if any(sim[i, j].item() > threshold for j in kept):
            dropped.add(r["id"])
        else:
            kept.append(i)
    return dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=DATASET_DIR / "raw" / "generated_raw.json")
    ap.add_argument("--out", type=Path, default=DATASET_DIR / "questions.json")
    ap.add_argument("--report", type=Path, default=DATASET_DIR / "metadata" / "filter_report.json")
    ap.add_argument("--device", default="cpu", help="MiniLM cihazı (üretim sürerken cpu)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = json.loads(args.raw.read_text(encoding="utf-8"))["records"]
    report: dict = {"raw": len(raw), "dropped": Counter()}
    survivors = []

    image_cats, terms = load_coco_categories(COCO_ANN)
    t4_stats = Counter()
    for r in tqdm(raw, desc="kurallar 1-3 + COCO T4", unit="soru"):
        t, q, a = r["question_type"], r["question"], r["answer"]
        if not (DATASET_DIR / r["image_path"]).exists():
            report["dropped"]["missing_image"] += 1
            continue
        if not qc_rules.word_count_ok(q):                       # Kural 1
            report["dropped"]["rule1_length"] += 1
            continue
        if t == "T4" and r["source"] == "coco":                  # Kural 2
            verdict = validate_t4(q, image_cats.get(int(r["image_id"]), set()), terms)
            t4_stats[verdict] += 1
            if verdict == "present":
                report["dropped"]["rule2_t4_object_present"] += 1
                continue
        if not qc_rules.answer_allowed(t, a):                    # Kural 3
            report["dropped"]["rule3_forbidden_answer"] += 1
            continue
        if qc_rules.leaks_prompt_example(t, q):                  # ek
            report["dropped"]["extra_prompt_example_leak"] += 1
            continue
        if t == "T4" and not qc_rules.t4_is_negation(a):         # ek
            report["dropped"]["extra_t4_not_negation"] += 1
            continue
        survivors.append(r)
    report["coco_t4_validation"] = dict(t4_stats)

    # Kural 4: görüntü başına duplikasyon
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(DEDUP_MODEL, device=args.device)
    by_image = defaultdict(list)
    for r in survivors:
        by_image[(r["source"], r["image_id"])].append(r)
    drop_ids: set[str] = set()
    for recs in tqdm(by_image.values(), desc="kural 4 dedup", unit="img"):
        if len(recs) > 1:
            drop_ids |= dedup_image_questions(recs, model, DEDUP_THRESHOLD)
    report["dropped"]["rule4_duplicate"] = len(drop_ids)
    final = sorted((r for r in survivors if r["id"] not in drop_ids), key=lambda r: r["id"])
    final = [{k: r[k] for k in FIELDS} for r in final]

    ids = [r["id"] for r in final]
    assert len(ids) == len(set(ids)), "benzersiz olmayan id"
    args.out.write_text(json.dumps(final, ensure_ascii=False, indent=1), encoding="utf-8")

    # Kaynağı COCO olmayan T4'ler için manuel inceleme örneklemi (doküman: sample bazlı)
    non_coco_t4 = [r for r in final if r["question_type"] == "T4" and r["source"] != "coco"]
    sample = random.Random(args.seed).sample(non_coco_t4, min(MANUAL_REVIEW_SAMPLE, len(non_coco_t4)))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    (args.report.parent / "t4_manual_review_sample.json").write_text(
        json.dumps(sample, ensure_ascii=False, indent=1), encoding="utf-8")

    report["final"] = len(final)
    report["dropped"] = dict(report["dropped"])
    report["by_type"] = dict(Counter(r["question_type"] for r in final))
    report["by_source"] = dict(Counter(r["source"] for r in final))
    report["by_source_type"] = {f"{s}/{t}": c for (s, t), c in
                                sorted(Counter((r["source"], r["question_type"]) for r in final).items())}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("raw", "final", "dropped", "by_type", "by_source", "coco_t4_validation")},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
