#!/usr/bin/env bash
# 4 policy variant'ını eğitir, ardından 7 variant'ı test split'te değerlendirir. Sırayla çalışır (8 GB GPU).
# Kullanım: bash scripts/start_ablation.sh   (UPDATES ve EVAL_LIMIT ortam değişkenleriyle ayarlanır)
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source venv_moe/bin/activate
UPDATES=${UPDATES:-300}
for v in ppo_fixedk_no_utility ppo_fixedk_cosine decisionmoe_a decisionmoe_b; do
  done_updates=$(python -c "import torch,os;p='artifacts/policy_$v.pt';print(torch.load(p)['update'] if os.path.exists(p) else 0)" 2>/dev/null || echo 0)
  if [ "$done_updates" -ge "$UPDATES" ]; then echo "$v zaten tamam ($done_updates güncelleme)"; continue; fi
  echo "=== EĞİTİM $v $(date +%T)"
  python scripts/train.py --variant $v --updates $UPDATES --eval-every 50 --resume --no-wandb 2>&1 | grep -a -E "^\[|VAL|τ_|Error|Traceback"
done
for v in standard_topk fixed_mcdm_no_utility fixed_mcdm_cosine ppo_fixedk_no_utility ppo_fixedk_cosine decisionmoe_a decisionmoe_b; do
  echo "=== DEĞERLENDİRME $v $(date +%T)"
  python scripts/evaluate.py --variant $v --split test ${EVAL_LIMIT:+--limit $EVAL_LIMIT} 2>&1 | grep -a -E "\"accuracy\"|hallucination_rate|mean_k|Error|Traceback"
done
echo "TAMAMLANDI $(date +%T)"
