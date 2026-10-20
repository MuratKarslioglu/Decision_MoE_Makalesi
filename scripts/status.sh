#!/usr/bin/env bash
# Çalışan işleri, GPU'yu, eğitim/değerlendirme ilerlemesini ve log sonunu gösterir.
# Kullanım: scripts/status.sh [log=logs/ds_sqa.txt]   Sürekli izleme: watch -n 30 scripts/status.sh
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG=${1:-logs/ds_sqa.txt}
echo "--- $(date +%T) | $(pwd)"
echo "--- çalışan işler"; pgrep -af "[t]rain_dynk.py|[e]val_test.py|[r]un_.*\.sh" | grep -v "bash -c" || echo "(yok)"
echo "--- GPU"; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
echo "--- eğitim logları (son 3)"
for f in $(ls -t logs/ds_train_*.json 2>/dev/null | head -3); do
  python3 -c "import json;l=json.load(open('$f'));print('$(basename $f)',len(l),'güncelleme | acc',round(l[-1]['acc'],3),'flops',round(l[-1]['flops'],3))"
done
echo "--- değerlendirme ara kaydı (soru sayısı)"
for f in logs/evalckpt_*.json; do [ -e "$f" ] && echo "$(basename $f): $(python3 -c "import json;print(len(json.load(open('$f'))['ok']))")"; done
echo "--- log sonu ($LOG)"; tail -6 "$LOG" 2>/dev/null
