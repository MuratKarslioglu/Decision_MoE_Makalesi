"""Alan prototipleri: expert e'nin prototipi = alan e örneklerinin MoE-girişi hidden state ortalaması.

Uzmanlaşmış expert'lerde (train_experts.py) gate'e göre değil, expert'in eğitildiği alana göre prototip tanımlanır; böylece
task_utility kriteri (cosine) alan bilgisini taşır. modality_rates vb. orijinal prototypes.pt'den kopyalanır.
Kullanım: DM_EXPERT_LORA=<lora.pt> python scripts/compute_domain_prototypes.py <tag> [alan başına örnek=100]
Çıktı: artifacts/prototypes_<tag>.pt (train split)
"""
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backbone.loader import load_backbone, load_model_config, prepare_inputs  # noqa: E402
from src.data.dataset import DecisionMoEDataset  # noqa: E402
from src.data.domains import domain_of  # noqa: E402
from src.routing.router import find_moe_layers  # noqa: E402

TAG = sys.argv[1]
PER = int(sys.argv[2]) if len(sys.argv) > 2 else 100


def main() -> None:
    random.seed(0)
    tokenizer, model, image_processor = load_backbone(load_model_config())
    ds = DecisionMoEDataset("train")
    layers = find_moe_layers(model)
    base = torch.load(ROOT / "artifacts" / "prototypes.pt")
    sums = torch.zeros_like(base["prototypes"], dtype=torch.float32, device="cuda")
    cnt = torch.zeros(sums.shape[:2], device="cuda")
    cur = {"d": 0}

    def make(i):
        def hook(_m, args):
            x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
            sums[i, cur["d"]] += x.sum(0)
            cnt[i, cur["d"]] += x.shape[0]
        return hook
    hs = [m.register_forward_pre_hook(make(i)) for i, (_, m) in enumerate(layers)]
    by = {d: [it for it in ds.items if domain_of(it) == d] for d in range(4)}
    with torch.inference_mode():
        for d in range(4):
            cur["d"] = d
            for it in tqdm(random.sample(by[d], min(PER, len(by[d]))), desc=f"alan {d}"):
                ids, pix = prepare_inputs(it, ds.root, tokenizer, image_processor)
                model(input_ids=ids, images=pix)
    for h in hs:
        h.remove()
    out = dict(base)
    out["prototypes"] = (sums / cnt.clamp(min=1).unsqueeze(-1)).cpu()
    out["domain_counts"] = cnt.cpu()
    torch.save(out, ROOT / "artifacts" / f"prototypes_{TAG}.pt")
    print("kaydedildi", TAG, cnt[0].tolist())


if __name__ == "__main__":
    main()
