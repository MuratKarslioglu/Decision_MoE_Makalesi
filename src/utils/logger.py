"""Deney logger: wandb (varsa) + her zaman yerel JSONL (Briefing §7). Anahtar adları sabittir."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "experiments" / "results" / "routing_logs"
CRITERIA = ("task_utility", "modality_match", "specialization", "reliability", "latency", "flops", "load_balance")


def routing_metrics(weights: torch.Tensor, mean_k: float, expert_load: torch.Tensor) -> dict[str, float]:
    """Briefing'deki sabit routing/* anahtarlarını üretir. weights: (7,) kriter ağırlığı veya (4,) expert önceliği, expert_load: (4,)."""
    out = {"routing/mean_K": mean_k}
    if weights.numel() == len(CRITERIA):
        out.update({f"routing/w_{n}": float(weights[i]) for i, n in enumerate(CRITERIA)})
    else:   # DIRECT kolu: 4 expert öncelik vektörü u_t
        out.update({f"routing/u_expert_{i}": float(weights[i]) for i in range(weights.numel())})
    out.update({f"routing/expert_{i}_load": float(expert_load[i]) for i in range(expert_load.numel())})
    return out


class RunLogger:
    """wandb.init + wandb.log sarmalayıcısı. WANDB_API_KEY yoksa veya wandb kurulu değilse yalnızca JSONL."""

    def __init__(self, variant_name: str, config: dict[str, Any], project: str = "decisionmoe",
                 tags: list[str] | None = None, use_wandb: bool = True) -> None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        seed = config.get("seed")
        tag = "" if seed in (None, 42) else f"_s{seed}"
        self.run_name = f"{variant_name}{tag}_{stamp}"
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.path = LOG_DIR / f"{self.run_name}.jsonl"
        self._fh = open(self.path, "a", encoding="utf-8")
        self._wandb = None
        if use_wandb and os.environ.get("WANDB_API_KEY"):
            try:
                import wandb
                wandb.init(project=project, name=self.run_name, config=config, tags=[variant_name, "ablation", *(tags or [])])
                self._wandb = wandb
            except Exception as e:  # noqa: BLE001 - ağ/kimlik sorunu eğitimi durdurmasın
                print(f"wandb devre dışı: {e}")
        self.log({"config": config}, step=0)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self._fh.write(json.dumps({"step": step, **metrics}, default=float) + "\n")
        self._fh.flush()
        if self._wandb is not None and "config" not in metrics:
            self._wandb.log(metrics, step=step)

    def close(self) -> None:
        self._fh.close()
        if self._wandb is not None:
            self._wandb.finish()
