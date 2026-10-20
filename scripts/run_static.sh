#!/usr/bin/env bash
# Sabit asimetrik (K_prefill,K_decode) baseline'ları: öğrenilmiş politika bunlardan iyi mi? AI2D ve ScienceQA test. Idempotent. Takip: python3 scripts/progress.py
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=./venv_ds/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ST="st1_4 st2_3 st2_4 st2_5 st3_4 st3_5 st2_6 st3_3"
for t in ai2d sqa; do
  echo "=== STATIC $t $(date +%T)"
  DS_TASK=$t $PY ds/eval_test.py test $ST 2>&1 | grep -a -E "acc=|devam|Error|Traceback"
done
echo STATIC_TAMAM
