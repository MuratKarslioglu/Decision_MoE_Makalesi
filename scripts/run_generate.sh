#!/usr/bin/env bash
# Tam soru üretimi (2000 görüntü, 8-bit). Kesilirse aynı komutla kaldığı yerden devam eder.
# Çıktı yalnızca generate_log.txt'ye gider (terminale yazılmaz), böylece terminal/VSCode kapansa da kilitlenmez.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source venv/bin/activate
python -u scripts/generate_questions.py --batch-size 4 --load-8bit --out decisionmoe_dataset/raw/generated_raw.json 2>&1 \
  | tr '\r' '\n' | grep --line-buffered -v -E 'MatMul8bitLt|Loading weights' >> generate_log.txt
