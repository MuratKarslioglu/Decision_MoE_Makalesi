#!/usr/bin/env bash
# 9 variant'lık hattı VSCode/terminalden bağımsız başlatır. Takip: tail -f ablation_all_log.txt
# DİKKAT: önce çalışan run_ablation.sh varsa durdurun (aynı GPU).
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if ps -eo pid,args | grep -E "bash scripts/run_ablation|python scripts/(train|evaluate)" | grep -v grep >/dev/null; then echo "HATA: başka bir ablation betiği çalışıyor"; exit 1; fi
setsid nohup bash scripts/run_ablation_all.sh > ablation_all_log.txt 2>&1 < /dev/null &
echo "başlatıldı. Takip: tail -f $(pwd)/ablation_all_log.txt"
