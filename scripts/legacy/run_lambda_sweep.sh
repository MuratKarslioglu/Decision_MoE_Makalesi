#!/usr/bin/env bash
# λ2 (FLOPs cezası) taraması: DecisionMoE-B'yi λ2=0.3 ve 1.0 ile eğitir, test'te değerlendirir, sonuçları analiz eder.
# Ana ablation hattı (run_ablation_all.sh) bitene kadar bekler (aynı GPU).
cd "$(dirname "${BASH_SOURCE[0]}")/.."
while ps -eo args | grep -E "bash scripts/run_ablation_all" | grep -v grep >/dev/null; do sleep 60; done
source venv_moe/bin/activate
UPDATES=${UPDATES:-300}
for v in decisionmoe_b_l03 decisionmoe_b_l30; do
  done_updates=$(python -c "import torch,os;p='artifacts/policy_$v.pt';print(torch.load(p)['update'] if os.path.exists(p) else 0)" 2>/dev/null || echo 0)
  if [ "$done_updates" -ge "$UPDATES" ]; then echo "$v zaten tamam"; else
    echo "=== EĞİTİM $v $(date +%T)"
    python scripts/train.py --variant $v --updates $UPDATES --eval-every 50 --resume --no-wandb 2>&1 | grep -a --line-buffered -E "^\[|VAL|reward λ|devam|Error|Traceback"
  fi
done
for v in decisionmoe_b_l03 decisionmoe_b_l30; do
  [ -f experiments/results/eval_${v}_test.json ] && { echo "$v değerlendirmesi zaten var"; continue; }
  echo "=== DEĞERLENDİRME $v $(date +%T)"
  python scripts/evaluate.py --variant $v --split test 2>&1 | grep -a --line-buffered -E "\"accuracy\"|mean_k|flops_fraction|Error|Traceback"
done
echo "=== ANALİZ $(date +%T)"
source venv/bin/activate
python scripts/analyze_results.py 2>&1 | tail -30
echo "TAMAMLANDI $(date +%T)"
