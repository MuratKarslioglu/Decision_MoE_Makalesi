"""Doldurulmuş denetim tablosunu özetler: geçersiz etiket / belirsiz soru oranı (Wilson %95 GA), tip bazlı."""
import csv
import math
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "experiments" / "results"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """k/n ve Wilson %95 güven aralığı."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def main() -> None:
    rows = [r for r in csv.DictReader(open(RES / "audit_sheet.csv", encoding="utf-8")) if r["ref_correct"].strip()]
    if not rows:
        raise SystemExit("tablo henüz doldurulmamış (ref_correct sütunu boş)")
    print(f"{len(rows)} doldurulmuş satır")
    print(f"{'tip':6s} {'n':>4s} {'ref yanlış/belirsiz':>22s} {'soru sorunlu':>22s} {'model anlamca doğru':>22s}")
    for t in ["T1", "T2", "T3", "T4", "T5", "hepsi"]:
        sub = [r for r in rows if t == "hepsi" or r["type"] == t]
        n = len(sub)
        bad_ref = sum(r["ref_correct"].strip().lower() != "y" for r in sub)
        bad_q = sum(r["question_ok"].strip().lower() == "n" for r in sub)
        ok_m = [r for r in sub if r["model_ok"].strip()]
        good_m = sum(r["model_ok"].strip().lower() == "y" for r in ok_m)
        f = lambda k, m: "{:5.1%} [{:4.1%},{:4.1%}]".format(*wilson(k, m))
        print(f"{t:6s} {n:4d} {f(bad_ref, n):>22s} {f(bad_q, n):>22s} {f(good_m, len(ok_m)):>22s}")
    overall_bad = sum(r["ref_correct"].strip().lower() != "y" for r in rows) / len(rows)
    print(f"\nTetikleyici D (geçersiz etiket > %15): {'EVET -> tabloları denetlenmiş alt kümede raporla' if overall_bad > 0.15 else 'hayır'}")


if __name__ == "__main__":
    main()
