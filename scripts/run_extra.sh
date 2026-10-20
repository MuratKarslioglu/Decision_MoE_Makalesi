#!/usr/bin/env bash
# İSTEĞE BAĞLI pekiştirme: ScienceQA'da λ=0.4 için 2 ek seed (3 ve 4) + test. Idempotent: kapanırsa aynı komutla devam eder. Takip: python3 scripts/progress.py
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=./venv_ds/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DS_TASK=sqa
for cfg in "q040s3 0.4 3" "q040s4 0.4 4"; do
  set -- $cfg
  echo "=== TRAIN $1 λ=$2 seed $3 $(date +%T)"
  $PY ds/train_dynk.py $1 $2 80 48 $3 2>&1 | grep -a -E "TAMAM|devam|Error|Traceback"
done
echo "=== EVAL $(date +%T)"
$PY ds/eval_test.py test q040s3 q040s4 2>&1 | grep -a -E "acc=|devam|Error|Traceback"
echo EXTRA_TAMAM
