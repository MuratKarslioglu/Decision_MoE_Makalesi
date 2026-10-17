#!/usr/bin/env bash
# Üretim durumu: süreç, son ilerleme, kayıt sayıları.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "== süreç"; pgrep -af generate_questions.py || echo "ÇALIŞMIYOR"
echo "== son ilerleme"; tail -c 2000 generate_log.txt | tr '\r' '\n' | grep -a "tur " | tail -1 | cut -c1-160
echo "== kayıtlar"
python3 - <<'P'
import json
d = json.load(open('decisionmoe_dataset/raw/generated_raw.json'))
print('görüntü:', len(d['images']), '| kayıt:', len(d['records']), '| başarısız:', len(d['failed']))
P
