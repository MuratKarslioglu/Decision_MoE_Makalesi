#!/usr/bin/env bash
# DeepSeek ikinci hat: POPE dış doğrulama, sonra dyn_l015 için 2 ek seed. Takip: tail -f logs/ds_run2.txt
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=./venv_ds/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== POPE $(date +%T)"
$PY ds/pope_ds.py 1000 topk_k6 topk_k3 dyn_l015 dyn_l005 topk_k2 2>&1 | grep -a -E "acc=|Error|Traceback"
for s in 1 2; do
  echo "=== TRAIN l015 seed $s $(date +%T)"
  $PY ds/train_dynk.py l015s$s 0.15 100 64 $s 2>&1 | grep -a -E "TAMAM|Error|Traceback"
done
echo "=== EVAL seeds $(date +%T)"
$PY ds/eval_test.py test l015s1 l015s2 2>&1 | grep -a -E "acc=|Error|Traceback"
echo DS2_TAMAM
