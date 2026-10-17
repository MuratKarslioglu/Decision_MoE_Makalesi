"""MoE-LLaVA backbone yükleme ve dondurma (Briefing §2).

moellava'nın `load_pretrained_model` fonksiyonu MoE modellerinde 8/4-bit'e izin vermez
(`assert not load_8bit` FIXME), bu yüzden Eval sınıfı doğrudan yüklenir. 8 GB VRAM için
bitsandbytes kuantizasyonu configs/model.yaml'dan okunur.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"


def load_model_config(path: Path = MODEL_CONFIG_PATH) -> dict[str, Any]:
    """configs/model.yaml dosyasını okur."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """Kodda cihaz hardcode edilmez (Briefing §7)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _init_single_process_dist() -> None:
    """DeepSpeed MoE katmanı process group ister; tek süreçli gloo grubu yeterlidir."""
    import os

    import torch.distributed as dist

    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    dist.init_process_group("gloo", rank=0, world_size=1)


def _build_quantization_config(cfg: dict[str, Any]):
    """quantization.mode'a göre BitsAndBytesConfig (veya None)."""
    from transformers import BitsAndBytesConfig

    q = cfg["quantization"]
    dtype = getattr(torch, q["compute_dtype"])
    if q["mode"] == "int8":
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=q["skip_modules"])
    if q["mode"] == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype, llm_int8_skip_modules=q["skip_modules"],
        )
    return None


def freeze_backbone(model: torch.nn.Module) -> int:
    """Tüm backbone ağırlıklarını dondurur, eval moduna alır. Dönüş: dondurulan parametre sayısı."""
    n = 0
    for p in model.parameters():
        p.requires_grad_(False)
        n += p.numel()
    model.eval()
    return n


def load_backbone(cfg: dict[str, Any] | None = None, freeze: bool | None = None):
    """MoE-LLaVA'yı yükler.

    Returns:
        tokenizer, model (EvalMoELLaVAPhiForCausalLM), image_processor (CLIPImageProcessor)
    """
    from moellava.model.language_model.llava_phi_moe import EvalMoELLaVAPhiForCausalLM
    from transformers import AutoTokenizer

    cfg = cfg or load_model_config()
    b = cfg["backbone"]
    _init_single_process_dist()

    kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    qcfg = _build_quantization_config(cfg)
    if qcfg is not None:
        kwargs.update(quantization_config=qcfg, device_map={"": 0})
    kwargs["torch_dtype"] = getattr(torch, cfg["quantization"]["compute_dtype"])

    tokenizer = AutoTokenizer.from_pretrained(b["hf_id"], use_fast=False, padding_side="right")
    model = EvalMoELLaVAPhiForCausalLM.from_pretrained(b["hf_id"], **kwargs)
    model.config.eos_token_id = tokenizer.eos_token_id

    image_tower = model.get_image_tower()
    if not image_tower.is_loaded:
        image_tower.load_model()
    image_tower.to(device=get_device(), dtype=kwargs["torch_dtype"])
    image_processor = image_tower.image_processor

    if qcfg is None:
        model.to(get_device())
    lora = os.environ.get("DM_EXPERT_LORA")           # uzmanlaşmış expert LoRA'ları (scripts/train_experts.py); yoksa orijinal model
    if lora:
        from src.backbone.expert_lora import load_expert_lora
        load_expert_lora(model, Path(lora))
    if b["freeze"] if freeze is None else freeze:
        freeze_backbone(model)
    return tokenizer, model, image_processor


def prepare_inputs(item: dict[str, Any], dataset_root: Path, tokenizer, image_processor, with_answer: bool = False):
    """Bir soru kaydını model girdisine çevirir.

    Returns:
        input_ids: (1, T) long, görüntü yer tutucusu -200 içerir
        pixel_values: (1, 3, 336, 336) float16
    """
    from moellava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from moellava.mm_utils import tokenizer_image_token
    from PIL import Image

    device = get_device()
    prompt = f"USER: {DEFAULT_IMAGE_TOKEN}\n{item['question']} ASSISTANT:"
    if with_answer:
        prompt += f" {item['answer']}"
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image = Image.open(dataset_root / item["image_path"]).convert("RGB")
    pixel_values = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].to(device, dtype=torch.float16)
    return input_ids, pixel_values
