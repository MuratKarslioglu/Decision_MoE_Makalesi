#!/usr/bin/env bash
# A-OKVQA (3. veri seti): Top-K + sabit asimetrik + top-p taban çizgileri, 3 seed PPO (λ=0.4), test. Idempotent. Takip: python3 scripts/progress.py
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=./venv_ds/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DS_TASK=aok
echo "=== BASELINES $(date +%T)"
$PY ds/eval_test.py test st1_4 st2_3 st2_4 st3_3 st3_4 st3_5 topp0.4 topp0.5 topp0.6 2>&1 | grep -a -E "acc=|devam|Error|Traceback"
for cfg in "o040 0.4 0" "o040s1 0.4 1" "o040s2 0.4 2"; do
  set -- $cfg
  echo "=== TRAIN $1 λ=$2 seed $3 $(date +%T)"
  $PY ds/train_dynk.py $1 $2 80 48 $3 2>&1 | grep -a -E "TAMAM|devam|Error|Traceback"
done
echo "=== EVAL PPO $(date +%T)"
$PY ds/eval_test.py test o040 o040s1 o040s2 2>&1 | grep -a -E "acc=|devam|Error|Traceback"
echo AOK_TAMAM
