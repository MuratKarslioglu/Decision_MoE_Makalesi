#!/usr/bin/env bash
# AI2D (akıl yürütme zinciri) rejimi: Dynamic-K PPO (3 λ, hepsi raporlanır) + test (597 soru) değerlendirmesi. Takip: tail -f logs/ds_run3.txt
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=./venv_ds/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DS_TASK=ai2d
while pgrep -f "ds/eval_test.py val" >/dev/null; do sleep 10; done
for cfg in "a020 0.2" "a040 0.4" "a080 0.8"; do
  set -- $cfg
  echo "=== TRAIN $1 λ=$2 $(date +%T)"
  $PY ds/train_dynk.py $1 $2 80 48 2>&1 | grep -a -E "TAMAM|Error|Traceback"
done
echo "=== EVAL test $(date +%T)"
$PY ds/eval_test.py test topp0.5 a020 a040 a080 2>&1 | grep -a -E "acc=|Error|Traceback"
echo DS3_TAMAM
