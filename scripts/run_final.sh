#!/usr/bin/env bash
# Yöntem makalesi için son pekiştirme (idempotent; kapanırsa aynı komutla kaldığı yerden sürer). Takip: python3 scripts/progress.py
#  A) ScienceQA-IMG λ süpürmesi (q020, q080; seed 0) -> DecisionMoE-K'nın maliyet-doğruluk eğrisi
#  B) AI2D ve A-OKVQA için λ=0.4 ek seed'ler (3 ve 4) -> her görevde 5 seed
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=./venv_ds/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {  # görev ad lambda seed
  echo "=== TRAIN $1/$2 λ=$3 seed $4 $(date +%T)"
  DS_TASK=$1 $PY ds/train_dynk.py $2 $3 80 48 $4 2>&1 | grep -a -E "TAMAM|devam|Error|Traceback"
}
run sqa q020 0.2 0; run sqa q080 0.8 0
run ai2d a040s3 0.4 3; run ai2d a040s4 0.4 4
run aok o040s3 0.4 3; run aok o040s4 0.4 4
for cfg in "sqa q020 q080" "ai2d a040s3 a040s4" "aok o040s3 o040s4"; do
  set -- $cfg; echo "=== EVAL $1 $(date +%T)"
  DS_TASK=$1 $PY ds/eval_test.py test $2 $3 2>&1 | grep -a -E "acc=|devam|Error|Traceback"
done
echo FINAL_TAMAM
