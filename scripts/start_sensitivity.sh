#!/usr/bin/env bash
# Duyarlılık testini bağımsız arka planda başlatır. Takip: tail -f sensitivity_log.txt
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source venv_moe/bin/activate
setsid nohup python -u scripts/sensitivity_test.py ${1:-400} > sensitivity_log.txt 2>&1 < /dev/null &
echo "başlatıldı. Takip: tail -f $(pwd)/sensitivity_log.txt"
