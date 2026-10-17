"""Expert prototype vektörlerini hesaplar ve artifacts/prototypes.pt olarak kaydeder (bir kere çalışır)."""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backbone.loader import load_backbone, load_model_config, prepare_inputs  # noqa: E402
from src.backbone.prototype import PrototypeAccumulator  # noqa: E402
from moellava.constants import IMAGE_TOKEN_INDEX  # noqa: E402

DATASET_ROOT = ROOT / "decisionmoe_dataset"


def stratified_sample(items: list[dict], n: int, seed: int) -> list[dict]:
    """(source, question_type) gruplarından eşit paylı örnekleme."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for it in items:
        groups[(it["source"], it["question_type"])].append(it)
    rng = random.Random(seed)
    per = max(1, n // len(groups))
    out = []
    for g in groups.values():
        rng.shuffle(g)
        out += g[:per]
    rng.shuffle(out)
    return out[:n]


N_IMAGE_TOKENS = 576   # CLIP ViT-L/14-336: (336/14)^2


def image_token_mask(input_ids: torch.Tensor, n_image_tokens: int) -> torch.Tensor:
    """Görüntü yer tutucusu (-200) n_image_tokens token'a genişleyince görüntü konumlarını işaretler.

    input_ids: (T,) -> (T - 1 + n_image_tokens,) bool
    """
    pos = (input_ids == IMAGE_TOKEN_INDEX).nonzero()[0].item()
    mask = torch.zeros(input_ids.numel() - 1 + n_image_tokens, dtype=torch.bool)
    mask[pos : pos + n_image_tokens] = True
    return mask


def main() -> None:
    cfg = load_model_config()
    pc = cfg["prototype"]
    items = json.load(open(DATASET_ROOT / "splits" / f"{pc['split']}.json"))
    items = stratified_sample(items, pc["n_samples"], pc["seed"])
    tokenizer, model, image_processor = load_backbone(cfg)
    acc = PrototypeAccumulator(model, cfg["backbone"]["n_experts"], cfg["backbone"]["hidden_dim"])
    print(f"{len(acc.layers)} MoE katmanı, {len(items)} örnek")
    with torch.inference_mode():
        for it in tqdm(items, desc="prototype"):
            ids, pix = prepare_inputs(it, DATASET_ROOT, tokenizer, image_processor)
            acc.image_mask = image_token_mask(ids[0], N_IMAGE_TOKENS)
            model(input_ids=ids, images=pix)
    acc.close()
    out = ROOT / pc["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prototypes": acc.prototypes().cpu(), "counts": acc.counts.cpu(), "modality_rates": acc.modality_rates().cpu(),
                "layers": [n for n, _ in acc.layers], "n_samples": len(items), "seed": pc["seed"]}, out)
    print("kaydedildi:", out)
    print("expert başına token sayısı (katman x expert):")
    print(acc.counts.cpu())
    dead = (acc.counts == 0).sum().item()
    print(f"hiç token almayan expert sayısı: {dead}")


if __name__ == "__main__":
    main()
