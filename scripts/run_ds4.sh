#!/usr/bin/env bash
# AI2D rejimi pekiştirme: (1) sezgisel τ taraması (test), (2) a040/a080 için 2'şer ek seed, (3) test değerlendirmesi. Takip: tail -f logs/ds_run4.txt
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=./venv_ds/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DS_TASK=ai2d
echo "=== TOPP sweep $(date +%T)"
$PY ds/eval_test.py test topp0.3 topp0.4 topp0.6 topp0.7 2>&1 | grep -a -E "acc=|Error|Traceback"
for cfg in "a040s1 0.4 1" "a040s2 0.4 2" "a080s1 0.8 1" "a080s2 0.8 2"; do
  set -- $cfg
  echo "=== TRAIN $1 λ=$2 seed $3 $(date +%T)"
  $PY ds/train_dynk.py $1 $2 80 48 $3 2>&1 | grep -a -E "TAMAM|Error|Traceback"
done
echo "=== EVAL seeds $(date +%T)"
$PY ds/eval_test.py test a040s1 a040s2 a080s1 a080s2 2>&1 | grep -a -E "acc=|Error|Traceback"
echo DS4_TAMAM
