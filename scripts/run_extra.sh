#!/usr/bin/env bash
# Ek POPE değerlendirmeleri: decisionmoe_b seed 1/2 politikaları ve fixed_mcdm_cosine. Takip: tail -f extra_log.txt
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=./venv_moe/bin/python
for s in 1 2; do $PY scripts/evaluate_pope.py --variants decisionmoe_b --policy-seed $s 2>&1 | grep -a -E "GENEL|HATA|Error|Traceback" || echo "HATA pope s$s"; done
$PY scripts/evaluate_pope.py --variants fixed_mcdm_cosine ppo_direct_fixedk 2>&1 | grep -a -E "GENEL|HATA|Error|Traceback"
echo EXTRA_TAMAM
