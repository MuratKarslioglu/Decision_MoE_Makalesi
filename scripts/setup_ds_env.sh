#!/usr/bin/env bash
# venv_ds: DeepSeek-VL2-Tiny için ayrı ortam (venv_moe'dan bağımsız; transformers 4.38.2 gerekir).
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 -m venv venv_ds
venv_ds/bin/python -m pip install -U pip
venv_ds/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
venv_ds/bin/python -m pip install "transformers==4.38.2" "accelerate" "bitsandbytes" timm sentencepiece attrdict einops pillow pyyaml numpy scipy "huggingface_hub<1.0" pytest
[ -d third_party/DeepSeek-VL2 ] || git clone --depth 1 https://github.com/deepseek-ai/DeepSeek-VL2.git third_party/DeepSeek-VL2
venv_ds/bin/python -m pip install --no-deps -e third_party/DeepSeek-VL2
echo SETUP_TAMAM
