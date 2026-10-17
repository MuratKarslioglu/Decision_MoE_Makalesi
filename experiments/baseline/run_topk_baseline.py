"""Standard Top-K baseline'ı (K=2, RL/MCDM yok) verilen split'te değerlendirir; evaluate.py'ye yönlendirir."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
split = sys.argv[1] if len(sys.argv) > 1 else "test"
sys.exit(subprocess.call([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--variant", "standard_topk",
                          "--split", split, *sys.argv[2:]]))
