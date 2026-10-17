#!/usr/bin/env bash
# Kuantize pilot: kullanım -> bash scripts/run_pilot.sh 8bit | 4bit
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source venv/bin/activate
MODE="${1:-8bit}"
if [ "$MODE" = "4bit" ]; then
  python scripts/generate_questions.py --pilot --batch-size 8 --load-4bit --out decisionmoe_dataset/raw/pilot_4bit_raw.json 2>&1 | tee pilot_4bit_log.txt
else
  python scripts/generate_questions.py --pilot --batch-size 4 --load-8bit --out decisionmoe_dataset/raw/pilot_8bit_raw.json 2>&1 | tee pilot_8bit_log.txt
fi
