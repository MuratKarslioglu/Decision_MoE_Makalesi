#!/usr/bin/env bash
# ScienceQA-IMG doğrulaması (idempotent: kapanma sonrası aynı komutla yeniden başlatılır, biten adımlar atlanır, yarım kalanlar checkpoint'ten sürer): (1) AI2D politikalarının transferi + Top-K, (2) SQA'da 3 seed PPO, (3) test. Takip: tail -f logs/ds_sqa.txt
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=./venv_ds/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DS_TASK=sqa
echo "=== TRANSFER+TOPK $(date +%T)"
$PY ds/eval_test.py test a040 a040s1 a040s2 a080 topp0.5 topp0.4 topp0.6 2>&1 | grep -a -E "acc=|Error|Traceback"
for cfg in "q040 0.4 0" "q040s1 0.4 1" "q040s2 0.4 2"; do
  set -- $cfg
  echo "=== TRAIN $1 λ=$2 seed $3 $(date +%T)"
  $PY ds/train_dynk.py $1 $2 80 48 $3 2>&1 | grep -a -E "TAMAM|Error|Traceback"
done
echo "=== EVAL SQA-trained $(date +%T)"
$PY ds/eval_test.py test q040 q040s1 q040s2 2>&1 | grep -a -E "acc=|Error|Traceback"
echo SQA_TAMAM
