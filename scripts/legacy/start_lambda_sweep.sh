#!/usr/bin/env bash
# λ2 taramasını bağımsız arka planda başlatır (ana hat bitene kadar bekler). Takip: tail -f lambda_sweep_log.txt
cd "$(dirname "${BASH_SOURCE[0]}")/.."
setsid nohup bash scripts/run_lambda_sweep.sh > lambda_sweep_log.txt 2>&1 < /dev/null &
echo "başlatıldı. Takip: tail -f $(pwd)/lambda_sweep_log.txt"
