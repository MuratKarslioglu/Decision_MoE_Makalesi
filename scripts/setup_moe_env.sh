#!/usr/bin/env bash
# venv_moe'yi sıfırdan kurar (MoE-LLaVA backbone için). Ana venv'den ayrıdır.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 -m venv venv_moe && source venv_moe/bin/activate
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # RTX 5060 (sm_120) için cu128
pip install -r requirements_moe.txt --extra-index-url https://download.pytorch.org/whl/cu128
[ -d third_party/MoE-LLaVA ] || git clone --depth 1 https://github.com/PKU-YuanGroup/MoE-LLaVA.git third_party/MoE-LLaVA
pip install --no-deps -e third_party/MoE-LLaVA
# bitsandbytes 0.50 + torch 2.11 yaması (int8 yolu için; nf4'te şart değil)
sed -i 's/torch.argwhere(outliers.any(dim=0)).view(-1)/torch.argwhere(outliers.any(dim=0)).reshape(-1)/' \
  venv_moe/lib/python3.12/site-packages/bitsandbytes/backends/cuda/ops.py
