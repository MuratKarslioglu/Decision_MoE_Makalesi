"""DeepSeek-VL2 Dynamic-K PPO eğitimi (train split). Kullanım: venv_ds/bin/python ds/train_dynk.py <ad> <lambda_flops> [güncelleme=60] [bölüm/güncelleme=48] [seed=0]
Çıktı: artifacts/ds_dynk_<ad>.pt + logs/ds_train_<ad>.json (güncelleme başına doğruluk, ortalama K, FLOPs payı)"""
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ds"))
import dsloader  # noqa: E402
from dsrouter import DSRouter  # noqa: E402
from dynk import DynKPolicy, episode_rewards, flops_fraction, ppo_update  # noqa: E402

import task  # noqa: E402

NAME, LAM = sys.argv[1], float(sys.argv[2])
UPDATES = int(sys.argv[3]) if len(sys.argv) > 3 else 60
EPS = int(sys.argv[4]) if len(sys.argv) > 4 else 48
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0


def main() -> None:
    random.seed(SEED); torch.manual_seed(SEED)
    proc, model = dsloader.load_ds()
    items, run = task.get("train")
    policy = DynKPolicy().cuda()
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    router = DSRouter(model, "dyn", policy=policy)
    log, t0, u0 = [], time.time(), 0
    ck = ROOT / "artifacts" / f"ckpt_{task.TASK}_{NAME}.pt"             # her güncellemede tam durum: kapanma sonrası kaldığı yerden
    if ck.exists():
        c = torch.load(ck, weights_only=False)
        policy.load_state_dict(c["policy"]); opt.load_state_dict(c["opt"]); log, u0 = c["log"], c["u"]
        random.setstate(c["py_rng"]); torch.set_rng_state(c["torch_rng"]); torch.cuda.set_rng_state(c["cuda_rng"])
        t0 -= log[-1]["t"] if log else 0
        print(f"checkpoint'ten devam: {u0}/{UPDATES}", flush=True)
    for u in range(u0, UPDATES):
        eps, accs, fl, kp, kd = [], [], [], [], []
        for it in random.sample(items, EPS):
            router.begin_episode()
            with torch.no_grad():
                a = float(run(proc, model, it))
            steps = router.steps
            eps.append((steps, episode_rewards(steps, a, LAM)))
            accs.append(a); fl.append(flops_fraction(steps)); kp.append(steps[0]["k"])
            kd += [s["k"] for s in steps[1:]]
        st = ppo_update(policy, opt, eps)
        rec = {"update": u + 1, "acc": sum(accs) / EPS, "flops": sum(fl) / EPS, "k_prefill": sum(kp) / EPS,
               "k_decode": sum(kd) / max(1, len(kd)), **st, "t": time.time() - t0}
        log.append(rec)
        print(json.dumps({k: round(v, 3) for k, v in rec.items()}), flush=True)
        torch.save(policy.state_dict(), ROOT / "artifacts" / f"ds_dynk_{NAME}.pt")
        (ROOT / "logs" / f"ds_train_{NAME}.json").write_text(json.dumps(log))
        torch.save({"policy": policy.state_dict(), "opt": opt.state_dict(), "log": log, "u": u + 1, "py_rng": random.getstate(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, str(ck) + ".tmp")
        os.replace(str(ck) + ".tmp", ck)
    print("EGITIM_TAMAM", flush=True)


if __name__ == "__main__":
    main()
