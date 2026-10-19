#!/usr/bin/env bash
# DeepSeek-VL2 hattı: 3 λ için Dynamic-K PPO eğitimi, sonra test değerlendirmesi. Takip: tail -f logs/ds_run.txt
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=./venv_ds/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for cfg in "l015 0.15" "l005 0.05" "l040 0.40"; do
  set -- $cfg
  echo "=== TRAIN $1 λ=$2 $(date +%T)"
  $PY ds/train_dynk.py $1 $2 100 64 2>&1 | grep -a -E "update|TAMAM|Error|Traceback"
done
echo "=== EVAL $(date +%T)"
$PY ds/eval_test.py test l015 l005 l040 2>&1 | grep -a -E "acc=|Error|Traceback"
echo DS_TAMAM
