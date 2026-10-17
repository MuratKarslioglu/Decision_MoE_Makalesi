#!/usr/bin/env bash
# Uzmanlaşmış expert profili (DM_TAG=spec): baseline + fixed MCDM + PPO-direct. Takip: tail -f spec_log.txt
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export DM_TAG=spec DM_EXPERT_LORA=artifacts/experts_contrast/lora.pt PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=./venv_moe/bin/python
for v in standard_topk fixed_mcdm_cosine; do
  echo "=== EVAL $v $(date +%T)"; $PY scripts/evaluate.py --variant $v || echo "HATA eval $v"
done
echo "=== TRAIN ppo_direct_fixedk $(date +%T)"
$PY scripts/train.py --variant ppo_direct_fixedk --updates 80 --eval-every 40 --no-wandb || echo "HATA train"
echo "=== EVAL ppo_direct_fixedk $(date +%T)"; $PY scripts/evaluate.py --variant ppo_direct_fixedk || echo "HATA eval ppo"
echo SPEC_TAMAM
