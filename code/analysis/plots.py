"""Generate bar charts and comparison plots from experiment summaries."""
from __future__ import annotations
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
import pandas as pd
import seaborn as sns

METHOD_ORDER = ["io", "cot", "cot_sc", "tot_bfs"]
MODEL_ORDER = ["gpt-4", "gemini-2.5-flash"]

MODEL_LABELS = {
    "gpt-4": "GPT-4 (paper)",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
}
METHOD_LABELS = {
    "io": "IO",
    "cot": "CoT",
    "cot_sc": "CoT-SC",
    "tot_bfs": "ToT (BFS)",
}

# Yao et al. 2023, Table 2 — GPT-4 paper baselines
PAPER_BASELINES = {
    "model":         ["gpt-4"] * 4,
    "method":        ["io", "cot", "cot_sc", "tot_bfs"],
    "success_rate":  [0.073, 0.040, 0.090, 0.740],
    "avg_api_calls": [1.0, 1.0, 100.0, None],
}

MODEL_COLORS = {
    "gpt-4":            "#7f7f7f",
    "gemini-2.5-flash": "#1f77b4",
}
METHOD_COLORS = {
    "io":      "#d62728",
    "cot":     "#ff7f0e",
    "cot_sc":  "#2ca02c",
    "tot_bfs": "#1f77b4",
}

mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

# Style constants — overridden by poster_style.py at import time
LINEWIDTH_MAIN = 1.8       # method lines in difficulty curve
LINEWIDTH_SECONDARY = 1.4  # human reference line
BAR_ANNOT_FONTSIZE = 9     # percentage labels on bar charts


def plot_success_rate(df: pd.DataFrame, output_path: str, include_paper_baselines: bool = False):
    """Bar chart: success rate by method, optionally grouped with GPT-4 paper baselines."""
    plot_df = df.copy()
    if include_paper_baselines:
        baseline_df = pd.DataFrame(PAPER_BASELINES)[["model", "method", "success_rate"]]
        plot_df = pd.concat([plot_df, baseline_df], ignore_index=True)

    active_models = [m for m in MODEL_ORDER if m in plot_df["model"].values]
    palette = {m: MODEL_COLORS.get(m, "#999999") for m in active_models}

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=plot_df,
        x="method",
        y="success_rate",
        hue="model",
        order=[m for m in METHOD_ORDER if m in plot_df["method"].values],
        hue_order=active_models,
        palette=palette,
        ax=ax,
        edgecolor="white",
        linewidth=0.5,
    )

    for container in ax.containers:
        ax.bar_label(container, fmt=lambda v: f"{v*100:.0f}%", padding=3, fontsize=BAR_ANNOT_FONTSIZE)

    ax.set_xlabel("Method", labelpad=8)
    ax.set_ylabel("Success Rate", labelpad=8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.set_title("Game of 24 — Success Rate by Method", pad=12)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(
        [METHOD_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()]
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [MODEL_LABELS.get(l, l) for l in labels], title="Model", frameon=False)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    if d := os.path.dirname(output_path):
        os.makedirs(d, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_tot_delta(df: pd.DataFrame, output_path: str):
    """Line chart: ToT vs IO delta — shows whether tree search adds more value for weaker models."""
    io_rates = df[df["method"] == "io"].set_index("model")["success_rate"]
    tot_rates = df[df["method"] == "tot_bfs"].set_index("model")["success_rate"]
    delta = (tot_rates - io_rates).dropna().reset_index()
    delta.columns = ["model", "delta"]

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=delta, x="model", y="delta", ax=ax)
    ax.set_title("ToT Benefit over IO Baseline")
    ax.set_ylabel("Success Rate Delta (ToT − IO)")
    ax.set_xlabel("Model")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_thought_tree(root_node, output_path: str):
    """Recursive tree layout with nodes colored by value score."""
    VALUE_COLORS = {
        "sure": "#4caf50",
        "likely": "#ff9800",
        "impossible": "#f44336",
    }
    DEFAULT_COLOR = "#9e9e9e"

    positions: dict = {}
    labels: dict = {}
    colors: dict = {}
    edges: list[tuple] = []
    counter = [0]

    def _assign_pos(node, depth: int) -> float:
        if not node.children:
            x = float(counter[0])
            counter[0] += 1
        else:
            child_xs = [_assign_pos(c, depth + 1) for c in node.children]
            x = sum(child_xs) / len(child_xs)
            for child in node.children:
                edges.append((id(node), id(child)))
        positions[id(node)] = (x, -depth)
        val = node.step_outputs.get("value", "")
        colors[id(node)] = VALUE_COLORS.get(val, DEFAULT_COLOR)
        remaining = node.step_outputs.get("remaining", "")
        labels[id(node)] = node.thought if not remaining else f"{node.thought}\n({remaining})"
        return x

    _assign_pos(root_node, 0)

    fig, ax = plt.subplots(figsize=(max(8, counter[0] * 1.2), 5))
    for src_id, dst_id in edges:
        x0, y0 = positions[src_id]
        x1, y1 = positions[dst_id]
        ax.plot([x0, x1], [y0, y1], "k-", lw=0.8, zorder=1)

    for nid, (x, y) in positions.items():
        ax.scatter(x, y, s=300, c=colors[nid], zorder=2, edgecolors="black", linewidths=0.5)
        ax.annotate(labels[nid], (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7)

    legend_patches = [mpatches.Patch(color=c, label=l) for l, c in VALUE_COLORS.items()]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)
    ax.set_title("Tree of Thoughts — Thought Tree")
    ax.axis("off")
    fig.tight_layout()
    if d := os.path.dirname(output_path):
        os.makedirs(d, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_api_cost(df: pd.DataFrame, output_path: str):
    """Bar chart: average API calls per problem by method.

    CoT-SC shows api_calls=1 due to Gemini batch counting; true cost = 5. Annotated with *.
    """
    avg = (
        df.groupby("method")["api_calls"]
        .mean()
        .reindex([m for m in METHOD_ORDER if m in df["method"].values])
        .reset_index()
    )
    avg.columns = ["method", "avg_api_calls"]
    avg["label"] = avg["method"].map(METHOD_LABELS).fillna(avg["method"])

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=avg, x="label", y="avg_api_calls", ax=ax,
        color=MODEL_COLORS["gemini-2.5-flash"], edgecolor="white", linewidth=0.5,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3, fontsize=BAR_ANNOT_FONTSIZE)

    if "cot_sc" in avg["method"].values:
        cot_sc_idx = avg["method"].tolist().index("cot_sc")
        patch = ax.patches[cot_sc_idx]
        ax.text(patch.get_x() + patch.get_width() / 2, patch.get_height() + 0.2,
                "*", ha="center", va="bottom", fontsize=BAR_ANNOT_FONTSIZE + 3, color="#e05c00")
    ax.set_xlabel("Method", labelpad=8)
    ax.set_ylabel("Avg API Calls per Problem", labelpad=8)
    ax.set_title("Game of 24 — API Call Cost by Method", pad=12)
    has_cot_sc_footnote = "cot_sc" in avg["method"].values
    bottom = 0.14 if has_cot_sc_footnote else 0
    fig.tight_layout(rect=[0, bottom, 1, 1])
    if has_cot_sc_footnote:
        fig.text(0.02, 0.02,
                 "* CoT-SC true cost ≈ 5 calls/problem (Gemini batch counted as 1)",
                 fontsize=BAR_ANNOT_FONTSIZE - 1, color="#555555", va="bottom")
    if d := os.path.dirname(output_path):
        os.makedirs(d, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_difficulty_curve(
    method_dfs: dict[str, pd.DataFrame],
    dataset_path: str,
    output_path: str,
    window: int = 10,
):
    """Line chart: per-problem success rate (rolling mean) vs. difficulty rank.

    method_dfs: {method: DataFrame with columns [problem_id, success, ...]}
    dataset_path: path to 24.csv — used to extract human solved rate as reference
    window: rolling mean window (center=True)

    X-axis: problem rank 901–1000 (increases with difficulty)
    Y-axis: rolling success rate (0–1)
    """
    dataset = pd.read_csv(dataset_path)
    dataset["Rank"] = dataset["Rank"].astype(int)
    dataset["human_rate"] = dataset["Solved rate"].str.rstrip("%").astype(float) / 100
    rank_to_human = dict(zip(dataset["Rank"], dataset["human_rate"]))

    fig, ax = plt.subplots(figsize=(9, 5))

    for method in METHOD_ORDER:
        if method not in method_dfs:
            continue
        mdf = method_dfs[method].copy().sort_values("problem_id").reset_index(drop=True)
        mdf["rolling_success"] = (
            mdf["success"].astype(int)
            .rolling(window=window, min_periods=1, center=True)
            .mean()
        )
        ax.plot(
            mdf["problem_id"],
            mdf["rolling_success"],
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method),
            linewidth=LINEWIDTH_MAIN,
            alpha=0.9,
        )

    # Human solved rate reference line (dashed gray)
    in_range = sorted(r for r in rank_to_human if 901 <= r <= 1000)
    if in_range:
        human_vals = pd.Series([rank_to_human[r] for r in in_range])
        human_rolling = human_vals.rolling(window=window, min_periods=1, center=True).mean()
        ax.plot(in_range, human_rolling, label="Human (AMT)",
                color="#aaaaaa", linewidth=LINEWIDTH_SECONDARY, linestyle="--", alpha=0.8)

    ax.set_xlabel("Problem Difficulty Rank (↑ harder)", labelpad=8)
    ax.set_ylabel(f"Success Rate (rolling mean, w={window})", labelpad=8)
    ax.set_title("Game of 24 — Performance vs. Problem Difficulty", pad=12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.set_xlim(901, 1000)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    if d := os.path.dirname(output_path):
        os.makedirs(d, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
