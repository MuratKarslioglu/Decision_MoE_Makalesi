#!/usr/bin/env bash
# decisionmoe_b için ek seed'ler: eğitim + test değerlendirmesi. Takip: tail -f seeds_log.txt
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=./venv_moe/bin/python
for s in 1 2; do
  echo "=== SEED $s EĞİTİM $(date +%T)"
  $PY scripts/train.py --variant decisionmoe_b --updates 150 --seed $s --resume --no-wandb || echo "HATA train $s"
  echo "=== SEED $s DEĞERLENDİRME $(date +%T)"
  $PY scripts/evaluate.py --variant decisionmoe_b --seed $s || echo "HATA eval $s"
done
echo SEEDLER_TAMAM
