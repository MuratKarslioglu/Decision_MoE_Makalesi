"""paper/draft.md -> paper/body.tex (pandoc; Unicode matematik simgeleri LaTeX'e çevrilir). Kullanım: venv/bin/python scripts/md2tex.py"""
import re
from pathlib import Path

import pypandoc

ROOT = Path(__file__).resolve().parents[1]
s = (ROOT / "paper/draft.md").read_text()
s = s[s.index("## 1 Introduction"):s.index("## Reproducibility")]
s = re.sub(r"\*\(Tüm 12 künye.*?\)\*\n", "", s, flags=re.S)
s = re.sub(r"\\cite\{([^}]*)\}", lambda m: "[" + "; ".join("@" + k.strip() for k in m.group(1).split(",")) + "]", s)
s = re.sub(r"^(#{2,3}) \d+(\.\d+)? ", r"\1 ", s, flags=re.M)
SUB = dict(zip("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
s = re.sub(r"([λτΔ])([₀-₉])", lambda m: "`$" + {"λ": "\\lambda", "τ": "\\tau", "Δ": "\\Delta"}[m.group(1)] + "_" + SUB[m.group(2)] + "$`{=latex}", s)
for a, b in {"λ": "$\\lambda$", "τ": "$\\tau$", "Δ": "$\\Delta$", "≈": "$\\approx$", "≤": "$\\leq$", "≥": "$\\geq$", "−": "-", "×": "$\\times$",
             "→": "$\\rightarrow$", "±": "$\\pm$", "γ": "$\\gamma$", "μ": "$\\mu$", "σ": "$\\sigma$", "–": "--", "—": "---", "≡": "$\\equiv$", "↔": "$\\leftrightarrow$", "∈": "$\\in$"}.items():
    s = s.replace(a, f"`{b}`{{=latex}}" if b.startswith("$") else b)
tex = pypandoc.convert_text(s, "latex", format="markdown+pipe_tables+tex_math_dollars", extra_args=["--shift-heading-level-by=-1", "--natbib", "--wrap=preserve"])
(ROOT / "paper/body.tex").write_text(tex)
print(len(tex), "karakter yazıldı; kalan Unicode:", sorted({c for c in tex if ord(c) > 127}))
