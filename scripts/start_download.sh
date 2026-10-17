#!/usr/bin/env bash
# İndirmeyi VSCode/terminalden bağımsız arka planda başlatır.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
setsid nohup bash scripts/download_backbone.sh > download_log.txt 2>&1 < /dev/null &
echo "başlatıldı. Takip: tail -f $(pwd)/download_log.txt"
