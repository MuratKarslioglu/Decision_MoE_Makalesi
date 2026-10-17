"""Backbone smoke testi: yükle, dondurmayı doğrula, VRAM ölç, tek görüntüyle kısa üretim yap."""
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backbone.loader import get_device, load_backbone, load_model_config  # noqa: E402


def main() -> None:
    cfg = load_model_config()
    tokenizer, model, image_processor = load_backbone(cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"parametre: {total/1e9:.2f}B | eğitilebilir: {trainable} (0 olmalı)")
    print(f"VRAM: {torch.cuda.memory_allocated()/2**30:.2f} GB (tepe {torch.cuda.max_memory_allocated()/2**30:.2f} GB)")

    root = Path(__file__).resolve().parents[1] / "decisionmoe_dataset"
    q = json.load(open(root / "questions.json"))
    item = q[0] if isinstance(q, list) else next(iter(q.values()))[0]
    img = Image.open(root / item["image_path"]).convert("RGB")
    from moellava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from moellava.mm_utils import tokenizer_image_token

    prompt = f"USER: {DEFAULT_IMAGE_TOKEN}\n{item['question']} ASSISTANT:"
    ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(get_device())
    pix = image_processor.preprocess(img, return_tensors="pt")["pixel_values"].to(get_device(), dtype=torch.float16)
    with torch.inference_mode():
        out = model.generate(ids, images=pix, do_sample=False, max_new_tokens=cfg["smoke_test"]["max_new_tokens"])
    print("soru:", item["question"])
    new = [t for t in out[0].tolist() if 0 <= t < len(tokenizer)]
    print("cevap:", repr(tokenizer.decode(new, skip_special_tokens=True)))
    print("beklenen:", item["answer"])
    print(f"tepe VRAM: {torch.cuda.max_memory_allocated()/2**30:.2f} GB")


if __name__ == "__main__":
    main()
