"""DeepSeek-VL2-Tiny nf4 yükleme ve soru çalıştırma (venv_ds). MoE-LLaVA hattına dokunmaz.

Uyumluluk yamaları: xformers yerine SDPA, nicemlenmiş modelde `.to` no-op (accelerate/4.38), CPU'da kalan rotary tamponları GPU'ya.
"""
from __future__ import annotations

import glob
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SNAP = glob.glob("/home/noron/.cache/huggingface/hub/models--deepseek-ai--deepseek-vl2-tiny/snapshots/*")[0]
SUFFIX = " Answer with a single word or short phrase."


def _shim_xformers() -> None:
    x, ops = types.ModuleType("xformers"), types.ModuleType("xformers.ops")
    ops.memory_efficient_attention = lambda q, k, v, p=0.0: F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=p).transpose(1, 2)
    x.ops = ops
    sys.modules.update({"xformers": x, "xformers.ops": ops})


def load_ds():
    """Returns: processor, model (nf4, GPU, eval)."""
    _shim_xformers()
    import transformers.modeling_utils as mu
    from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
    from transformers import BitsAndBytesConfig

    orig_to = mu.PreTrainedModel.to
    mu.PreTrainedModel.to = lambda self, *a, **k: self if getattr(self, "is_quantized", False) else orig_to(self, *a, **k)
    proc = DeepseekVLV2Processor.from_pretrained(SNAP)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["vision", "projector", "lm_head"])
    m = DeepseekVLV2ForCausalLM.from_pretrained(SNAP, trust_remote_code=True, quantization_config=q, torch_dtype=torch.bfloat16,
                                                device_map={"": 0}, low_cpu_mem_usage=True).eval()
    for mod in m.modules():
        for n, b in list(mod._buffers.items()):
            if b is not None and b.device.type == "cpu":
                mod._buffers[n] = b.cuda()
    for p in m.parameters():
        p.requires_grad_(False)
    return proc, m


def answer(proc, model, image_path: Path, question: str, max_new_tokens: int = 10) -> str:
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    conv = [{"role": "<|User|>", "content": f"<image>\n{question}{SUFFIX}", "images": [img]}, {"role": "<|Assistant|>", "content": ""}]
    inp = proc(conversations=conv, images=[img], force_batchify=True, system_prompt="").to(model.device)
    tok = proc.tokenizer
    with torch.no_grad():
        emb = model.prepare_inputs_embeds(**inp)
        out = model.language.generate(inputs_embeds=emb, attention_mask=inp.attention_mask, pad_token_id=tok.eos_token_id,
                                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                                      max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    return tok.decode(out[0].cpu().tolist(), skip_special_tokens=True).strip().split("\n")[0].strip().rstrip(".")
