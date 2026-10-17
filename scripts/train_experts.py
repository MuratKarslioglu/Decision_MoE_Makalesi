"""Expert farklılaştırma: alan d verisi yalnızca expert d'ye yönlendirilerek (gate atlanır) expert LoRA'ları eğitilir.

Gövde, gate ve görüntü kulesi donuk; yalnızca expert fc1/fc2 LoRA. Yalnızca train split.
Kullanım: python scripts/train_experts.py [adım=2400] [accum=8] [lr=2e-4] [kontrast_lambda=0] [başlangıç_lora|-] [çıktı_klasörü=experts_diverse] [marj=0.5]
Kontrast (lambda>0): her örnekte rastgele yanlış expert e için  L_d + lambda*relu(L_d - L_e + marj)  (marj=0.5 nat):
doğru expertin kaybı yanlış expertinkinden en az marj kadar düşük olmaya zorlanır.
Çıktı: artifacts/<klasör>/lora.pt + train_log.json
"""
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backbone.expert_lora import add_expert_lora, load_expert_lora, lora_state  # noqa: E402
from src.backbone.loader import load_backbone, load_model_config, prepare_inputs  # noqa: E402
from src.data.dataset import DecisionMoEDataset  # noqa: E402
from src.data.domains import DOMAIN_NAMES, domain_of  # noqa: E402
from src.routing.router import find_moe_layers  # noqa: E402
from src.routing.runner import SHORT_ANSWER_SUFFIX  # noqa: E402

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2400
ACCUM = int(sys.argv[2]) if len(sys.argv) > 2 else 8
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 2e-4
CONTRAST = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
INIT = None if len(sys.argv) < 6 or sys.argv[5] == "-" else Path(sys.argv[5])
MARGIN = float(sys.argv[7]) if len(sys.argv) > 7 else 0.5
OUT = ROOT / "artifacts" / (sys.argv[6] if len(sys.argv) > 6 else "experts_diverse")
LOG_EVERY = 100 if STEPS >= 400 else 4


def main() -> None:
    random.seed(0)
    torch.manual_seed(0)
    tokenizer, model, image_processor = load_backbone(load_model_config())
    if INIT is not None:
        load_expert_lora(model, INIT)
    else:
        add_expert_lora(model)
    OUT.mkdir(parents=True, exist_ok=True)
    params = [p for n, p in model.named_parameters() if n.endswith((".A", ".B"))]
    for p in params:
        p.requires_grad_(True)
    print(f"LoRA parametre: {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.05, 1 - s / max(1, STEPS // ACCUM)))

    cur = {"d": 0}
    for _, moe in find_moe_layers(model):
        layer = moe.deepspeed_moe
        exps = layer.experts.deepspeed_experts

        def fwd(hidden, used_token=None, _e=exps, _l=layer):
            y = _e[cur["d"]](hidden)
            _l.l_aux = torch.zeros((), device=hidden.device)
            _l.exp_counts = torch.zeros(len(_e), device=hidden.device)
            return y[0] if isinstance(y, tuple) else y
        layer.forward = fwd

    from torch.utils.checkpoint import checkpoint
    for blk in model.get_model().layers:                    # 8 GB için aktivasyon yeniden hesaplama
        def ck(*a, _f=blk.forward, **k):
            return checkpoint(_f, *a, use_reentrant=False, **k)
        blk.forward = ck

    ds = DecisionMoEDataset("train")
    by_dom: dict[int, list] = {d: [] for d in range(len(DOMAIN_NAMES))}
    for it in ds.items:
        by_dom[domain_of(it)].append(it)
    print({DOMAIN_NAMES[d]: len(v) for d, v in by_dom.items()}, flush=True)
    model.eval()                                   # dropout kapalı; yalnızca LoRA gradyan alır
    log, t0, run = [], time.time(), []
    for step in range(STEPS):
        d = step % len(DOMAIN_NAMES)
        it = dict(random.choice(by_dom[d]))
        it["question"] += SHORT_ANSWER_SUFFIX
        ids, pix = prepare_inputs(it, ds.root, tokenizer, image_processor, with_answer=True)
        n_prompt = prepare_inputs(it, ds.root, tokenizer, image_processor)[0].shape[1]
        labels = ids.clone()
        labels[:, :n_prompt] = -100
        cur["d"] = d
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=ids, images=pix, labels=labels, use_cache=False)[0]
            total = loss
            if CONTRAST > 0:
                cur["d"] = random.choice([e for e in range(len(DOMAIN_NAMES)) if e != d])
                loss_wrong = model(input_ids=ids, images=pix, labels=labels, use_cache=False)[0]
                total = loss + CONTRAST * torch.relu(loss - loss_wrong + MARGIN)
                cur["d"] = d
        scaler.scale(total / ACCUM).backward()
        run.append((d, loss.item()))
        if (step + 1) % ACCUM == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
        if (step + 1) % LOG_EVERY == 0:
            per = {DOMAIN_NAMES[k]: round(sum(l for dd, l in run if dd == k) / max(1, sum(dd == k for dd, _ in run)), 3)
                   for k in range(len(DOMAIN_NAMES))}
            log.append({"step": step + 1, "loss": per, "mem_gb": torch.cuda.max_memory_allocated() / 1e9})
            print(step + 1, per, f"{torch.cuda.max_memory_allocated() / 1e9:.1f}GB {time.time() - t0:.0f}s", flush=True)
            run = []
            torch.save(lora_state(model), OUT / "lora.pt")
            (OUT / "train_log.json").write_text(json.dumps(log))
    torch.save(lora_state(model), OUT / "lora.pt")
    print("EGITIM_TAMAM", flush=True)


if __name__ == "__main__":
    main()
