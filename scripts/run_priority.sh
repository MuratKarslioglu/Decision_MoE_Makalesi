#!/usr/bin/env bash
# Öncelikli hat: önce çekirdek (decisionmoe_b -> değerlendirme -> latency -> POPE), sonra opsiyoneller.
# Bir adım hata verirse log'a yazılır ve hat devam eder. UPDATES (varsayılan 150) --resume ile sonradan uzatılabilir.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source venv_moe/bin/activate
UPDATES=${UPDATES:-150}
train() {  # $1=variant $2=updates
  done_updates=$(python -c "import torch,os;p='artifacts/policy_$1.pt';print(torch.load(p)['update'] if os.path.exists(p) else 0)" 2>/dev/null || echo 0)
  if [ "$done_updates" -ge "$2" ]; then echo "$1 zaten tamam ($done_updates güncelleme)"; return; fi
  echo "=== EĞİTİM $1 hedef=$2 $(date +%T)"
  python scripts/train.py --variant $1 --updates $2 --eval-every 50 --resume --no-wandb 2>&1 | grep -a --line-buffered -E "^\[|VAL|τ_|devam|reward λ|Error|Traceback"
}
evaluate() {  # $1=variant
  [ -f experiments/results/eval_$1_test.json ] && { echo "$1 değerlendirmesi zaten var"; return; }
  echo "=== DEĞERLENDİRME $1 $(date +%T)"
  python scripts/evaluate.py --variant $1 --split test 2>&1 | grep -a --line-buffered -E "\"accuracy\"|flops_fraction|mean_k|Error|Traceback" || echo "HATA: $1 değerlendirmesi"
}
train decisionmoe_b $UPDATES
evaluate decisionmoe_b
echo "=== LATENCY BENCHMARK $(date +%T)"
python scripts/benchmark_latency.py --variants standard_topk standard_topk_k1 fixed_mcdm_cosine ppo_direct_fixedk decisionmoe_b --n 100 --reps 3 2>&1 | grep -a --line-buffered -E "^rep|oran|TAMAM|Error|Traceback" || echo "HATA: latency benchmark"
echo "=== POPE $(date +%T)"
python scripts/evaluate_pope.py --variants standard_topk standard_topk_k1 decisionmoe_b --n-per-split 1000 2>&1 | grep -a --line-buffered -E "acc=|GENEL|TAMAM|Error|Traceback" || echo "HATA: POPE"
echo "CEKIRDEK_TAMAM $(date +%T)"
train decisionmoe_a $UPDATES
train ppo_direct_dynk $UPDATES
evaluate decisionmoe_a
evaluate ppo_direct_dynk
echo "TAMAMLANDI $(date +%T)"
