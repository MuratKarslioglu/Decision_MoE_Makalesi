#!/usr/bin/env bash
# MoE-LLaVA backbone + CLIP görüntü kulesi indirmesi (~13 GB). Kesilirse aynı komutla kaldığı yerden devam eder.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source venv_moe/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=0
python - <<'P'
from huggingface_hub import snapshot_download
for repo in ("LanguageBind/MoE-LLaVA-Phi2-2.7B-4e", "openai/clip-vit-large-patch14-336", "microsoft/phi-2"):
    print("indiriliyor:", repo, flush=True)
    kw = {"allow_patterns": ["*.json", "*.txt", "*.py", "*.model"]} if repo == "microsoft/phi-2" else {}
    print("bitti:", snapshot_download(repo, **kw), flush=True)
P
echo "TAMAMLANDI"
