#!/usr/bin/env bash
# Verimlilik odaklı hat (run2). Önce Dynamic-K varyantları (öğrenilebilir sinyal FLOPs/K ekseninde), sonra sabit-K PPO'lar.
# Sonunda 10 variant test split'inde değerlendirilir. UPDATES_DYN / UPDATES_FIXED / EVAL_LIMIT ortam değişkenleriyle ayarlanır.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source venv_moe/bin/activate
UPDATES_DYN=${UPDATES_DYN:-300}
UPDATES_FIXED=${UPDATES_FIXED:-100}
TRAIN_DYN="decisionmoe_b decisionmoe_a ppo_direct_dynk"
TRAIN_FIXED="ppo_fixedk_cosine ppo_fixedk_no_utility ppo_direct_fixedk"
EVAL="standard_topk standard_topk_k1 fixed_mcdm_no_utility fixed_mcdm_cosine $TRAIN_FIXED $TRAIN_DYN"
train() {  # $1=variant $2=updates
  done_updates=$(python -c "import torch,os;p='artifacts/policy_$1.pt';print(torch.load(p)['update'] if os.path.exists(p) else 0)" 2>/dev/null || echo 0)
  if [ "$done_updates" -ge "$2" ]; then echo "$1 zaten tamam ($done_updates güncelleme)"; return; fi
  echo "=== EĞİTİM $1 hedef=$2 $(date +%T)"
  python scripts/train.py --variant $1 --updates $2 --eval-every 50 --resume --no-wandb 2>&1 | grep -a --line-buffered -E "^\[|VAL|τ_|devam|Error|Traceback"
}
for v in $TRAIN_DYN; do train $v $UPDATES_DYN; done
for v in $TRAIN_FIXED; do train $v $UPDATES_FIXED; done
for v in $EVAL; do
  [ -f experiments/results/eval_${v}_test.json ] && { echo "$v değerlendirmesi zaten var"; continue; }
  echo "=== DEĞERLENDİRME $v $(date +%T)"
  python scripts/evaluate.py --variant $v --split test ${EVAL_LIMIT:+--limit $EVAL_LIMIT} 2>&1 | grep -a --line-buffered -E "\"accuracy\"|hallucination_rate|mean_k|flops_fraction|Error|Traceback"
done
echo "TAMAMLANDI $(date +%T)"
