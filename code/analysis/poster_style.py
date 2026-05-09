"""
Poster-quality style overrides for plots.py.

Import this module before generating figures to apply poster-appropriate
font sizes, line weights, and a classic academic color palette.

Usage (in generate_figures.py):
    import analysis.poster_style  # noqa: F401  — import for side effects
"""
import matplotlib as mpl
import analysis.plots as _plots

mpl.rcParams.update({
    "font.family":           "sans-serif",
    "font.size":             14,
    "axes.titlesize":        18,
    "axes.titleweight":      "bold",
    "axes.labelsize":        15,
    "xtick.labelsize":       13,
    "ytick.labelsize":       13,
    "legend.fontsize":       13,
    "legend.title_fontsize": 13,
    "lines.linewidth":       2.5,
    "axes.linewidth":        1.2,
    "grid.linewidth":        0.8,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.grid":             True,
    "axes.grid.axis":        "y",
    "grid.alpha":            0.35,
    "grid.linestyle":        "--",
})

# Classic academic color palette
_plots.MODEL_COLORS.update({
    "gpt-4":            "#7f7f7f",   # neutral gray — established baseline
    "gemini-2.5-flash": "#2e5fac",   # deep navy blue
})
_plots.METHOD_COLORS.update({
    "io":      "#c0392b",   # crimson
    "cot":     "#d4822a",   # amber
    "cot_sc":  "#1a7340",   # forest green
    "tot_bfs": "#2e5fac",   # deep navy
})

# Override line widths and annotation sizes
_plots.LINEWIDTH_MAIN      = 2.5
_plots.LINEWIDTH_SECONDARY = 2.0
_plots.BAR_ANNOT_FONTSIZE  = 12
