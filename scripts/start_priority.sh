#!/usr/bin/env bash
# Öncelikli hattı bağımsız arka planda başlatır. Takip: tail -f priority_log.txt
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if ps -eo pid,args | grep -E "bash scripts/run_(ablation|priority|lambda)|python scripts/(train|evaluate)" | grep -v grep >/dev/null; then echo "HATA: başka bir hat çalışıyor"; exit 1; fi
setsid nohup bash scripts/run_priority.sh > priority_log.txt 2>&1 < /dev/null &
echo "başlatıldı. Takip: tail -f $(pwd)/priority_log.txt"
