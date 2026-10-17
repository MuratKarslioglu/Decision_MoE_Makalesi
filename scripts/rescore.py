"""Kayıtlı tahminleri (items[].pred) ikincil metriklerle yeniden puanlar; GPU gerekmez.

Katı doğruluğun mevcut JSON'larla birebir aynı çıktığını doğrular, sonra token-F1 ve MiniLM benzerliği ekler.
Çıktı: experiments/results/rescored.json + tip bazlı tablo. Kullanım: python scripts/rescore.py
"""
import glob
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.metrics import is_correct  # noqa: E402
from src.utils.secondary_metrics import semantic_similarity, token_f1  # noqa: E402

RES = ROOT / "experiments" / "results"
SEM_THRESHOLDS = (0.5, 0.7)
TYPES = ("T1", "T2", "T3", "T4", "T5")


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    refs = {x["id"]: x["answer"] for x in json.load(open(ROOT / "decisionmoe_dataset" / "splits" / "test.json"))}
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    out = {}
    for f in sorted(glob.glob(str(RES / "eval_*_test.json"))):
        d = json.load(open(f))
        items = d.get("items")
        if not items:
            print(f"atlandı (soru bazlı kayıt yok): {d['variant']}")
            continue
        ref = [refs[i["id"]] for i in items]
        pred = [i["pred"] for i in items]
        strict = [float(is_correct(p, r)) for p, r in zip(pred, ref)]
        assert abs(mean(strict) - d["accuracy"]) < 1e-9, f"{d['variant']}: katı doğruluk yeniden üretilemedi"
        f1 = [token_f1(p, r) for p, r in zip(pred, ref)]
        sim = semantic_similarity(pred, ref, st)
        row = {"strict_acc": mean(strict), "token_f1": mean(f1), "sem_cos": mean(sim),
               **{f"sem>={t}": mean([float(s >= t) for s in sim]) for t in SEM_THRESHOLDS}, "by_type": {}}
        for t in TYPES:
            idx = [k for k, i in enumerate(items) if i["type"] == t]
            row["by_type"][t] = {"strict": mean([strict[k] for k in idx]), "f1": mean([f1[k] for k in idx]),
                                 "sem_cos": mean([sim[k] for k in idx]),
                                 "sem>=0.7": mean([float(sim[k] >= 0.7) for k in idx])}
        out[d["variant"]] = row
    (RES / "rescored.json").write_text(json.dumps(out, indent=1))
    print(f"{'variant':24s} {'strict':>7s} {'F1':>6s} {'cos':>6s} {'cos>=.7':>8s} | T3: {'strict':>6s} {'F1':>6s} {'cos>=.7':>8s}")
    for v, r in out.items():
        t3 = r["by_type"]["T3"]
        print(f"{v:24s} {r['strict_acc']:7.3f} {r['token_f1']:6.3f} {r['sem_cos']:6.3f} {r['sem>=0.7']:8.3f} | "
              f"    {t3['strict']:6.3f} {t3['f1']:6.3f} {t3['sem>=0.7']:8.3f}")


if __name__ == "__main__":
    main()
