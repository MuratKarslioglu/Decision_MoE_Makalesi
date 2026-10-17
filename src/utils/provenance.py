"""Sonuç dosyaları için kaynak (provenance) kaydı: hangi kod, yapılandırma ve sürümlerle üretildi?

Her eval/eğitim çıktısına `meta` olarak eklenir; makaledeki sayıların yeniden üretilebilirliğini destekler.
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("torch", "transformers", "accelerate", "bitsandbytes", "deepspeed", "numpy", "scipy", "sentence_transformers")


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "n/a"
    except (OSError, subprocess.SubprocessError):
        return "n/a"


def file_hash(path: Path) -> str:
    """Dosyanın SHA-256 özetinin ilk 12 karakteri (yoksa 'missing')."""
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.exists() else "missing"


def package_versions() -> dict[str, str]:
    """Kurulu paketlerin sürümleri (kurulu değilse 'n/a')."""
    from importlib import metadata

    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name.replace("_", "-"))
        except metadata.PackageNotFoundError:
            out[name] = "n/a"
    return out


def collect_meta(seed: int | None = None, variant_file: str | Path | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Bir koşunun kaynak kaydı: commit, config özetleri, sürümler, donanım, seed."""
    meta: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")) if _git("rev-parse", "--short", "HEAD") != "n/a" else None,
        "python": platform.python_version(),
        "packages": package_versions(),
        "config_hashes": {p.name: file_hash(p) for p in sorted((ROOT / "configs").glob("*.yaml"))},
        "prototypes_hash": file_hash(ROOT / "artifacts" / "prototypes.pt"),
        "seed": seed,
    }
    if variant_file is not None:
        meta["variant_file"] = {"path": str(variant_file), "hash": file_hash(Path(variant_file))}
    try:
        import torch

        meta["gpu"] = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - torch yoksa/başlatılamazsa kayıt yine de yazılsın
        meta["gpu"] = "n/a"
    if extra:
        meta.update(extra)
    return meta
