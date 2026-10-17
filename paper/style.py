"""Makale figürleri için sabit Matplotlib stil ayarları (Briefing §9). Değiştirilmez."""
PAPER_RC = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (6.5, 4.0),  # IJCV single-column
}

COLORS = {
    "decisionmoe_b": "#2E86AB",  # Ana model
    "decisionmoe_a": "#A23B72",  # Utility-free
    "baseline": "#F18F01",       # Standard Top-K
    "ablation": "#C73E1D",       # Diğer variant'lar
    "neutral": "#6B6B6B",
}


def apply_style() -> None:
    """PAPER_RC'yi matplotlib'e uygular."""
    import matplotlib as mpl

    mpl.rcParams.update(PAPER_RC)


def variant_color(name: str) -> str:
    """Variant adından sabit renk."""
    if name in COLORS:
        return COLORS[name]
    if name.startswith("standard_topk"):
        return COLORS["baseline"]
    return COLORS["ablation"]
