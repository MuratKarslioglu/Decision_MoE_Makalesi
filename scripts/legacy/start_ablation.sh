#!/usr/bin/env bash
# Ablation hattını VSCode/terminalden bağımsız başlatır. Takip: tail -f ablation_log.txt
cd "$(dirname "${BASH_SOURCE[0]}")/.."
setsid nohup bash scripts/run_ablation.sh > ablation_log.txt 2>&1 < /dev/null &
echo "başlatıldı. Takip: tail -f $(pwd)/ablation_log.txt"
