#!/usr/bin/env bash
# Gecikme ölçümü + bağımsız ScienceQA test2 dilimi (idempotent; kapanırsa aynı komutla sürer). Takip: python3 scripts/progress.py
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=./venv_ds/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -n "$(venv/bin/python - <<'P'
import json;print('x' if any(d['split']=='test2' for d in json.load(open('sqa_data/sqa.json'))) else '')
P
)" ] || venv/bin/python scripts/prepare_sqa_holdout.py
echo "=== LATENCY $(date +%T)"
DS_TASK=sqa $PY ds/latency.py 40 q040 q040s2 q040s3 q080 2>&1 | grep -a -E "ms|Error|Traceback"
echo "=== EVAL sqa test2 $(date +%T)"
DS_TASK=sqa $PY ds/eval_test.py test2 q040 q040s1 q040s2 q040s3 q040s4 q020 q080 st2_4 2>&1 | grep -a -E "acc=|devam|Error|Traceback"
echo LAT_TAMAM
