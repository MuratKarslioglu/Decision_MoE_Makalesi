"""Val'de seçilen TOPSIS ayarının (g4_p3_s2_l1) test doğrulaması, tek seferlik. K=3, alan oracle. Çıktı: experiments/results/ds_mcdm_test.json
Karşılaştırma: ds_eval_test.json içindeki topk_k3 (aynı sorular, soru bazında eşleştirilmiş). Kullanım: venv_ds/bin/python ds/mcdm_test.py"""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ds"))
import dsloader  # noqa: E402
from dsrouter import DSRouter  # noqa: E402

from src.data.dataset import DecisionMoEDataset  # noqa: E402
from src.data.domains import domain_of  # noqa: E402
from src.utils.metrics import is_correct  # noqa: E402

W = {"topsis_g4_p3_s2_l1": (.4, .3, .2, .1), "topsis_g7_p3": (.7, .3, 0, 0)}
OUT = ROOT / "experiments" / "results" / "ds_mcdm_test.json"


def main() -> None:
    proc, model = dsloader.load_ds()
    prior = torch.load(ROOT / "artifacts" / "ds_domain_prior.pt")
    ds = DecisionMoEDataset("test")
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, w in W.items():
        if name in res:
            continue
        r = DSRouter(model, "mcdm", 3, mcdm={**prior, "w": torch.tensor(w, dtype=torch.float32)})
        ok, t0 = [], time.time()
        for it in ds.items:
            r.domain = domain_of(it)
            ok.append(int(is_correct(dsloader.answer(proc, model, ds.root / it["image_path"], it["question"]), it["answer"])))
        r.close()
        res[name] = {"ids": [it["id"] for it in ds.items], "correct": ok, "accuracy": sum(ok) / len(ok), "seconds": time.time() - t0}
        print(f"{name:20s} acc={res[name]['accuracy']:.3f} {res[name]['seconds']:.0f}s", flush=True)
        OUT.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
