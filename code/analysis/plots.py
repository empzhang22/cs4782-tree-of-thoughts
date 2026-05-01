"""Generate bar charts and comparison plots from experiment summaries."""
from __future__ import annotations
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


METHOD_ORDER = ["io", "cot", "cot_sc", "tot_bfs"]
MODEL_ORDER = ["gpt-4o"]
METHOD_LABELS = {
    "io": "IO",
    "cot": "CoT",
    "cot_sc": "CoT-SC",
    "tot_bfs": "ToT (BFS)",
}


def plot_success_rate(df: pd.DataFrame, output_path: str):
    """Bar chart: success rate by method × model (reproduces paper Table 2 style)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="method",
        y="success_rate",
        hue="model",
        order=[m for m in METHOD_ORDER if m in df["method"].values],
        hue_order=[m for m in MODEL_ORDER if m in df["model"].values],
        ax=ax,
    )
    ax.set_xlabel("Method")
    ax.set_ylabel("Success Rate")
    ax.set_title("Game of 24 — Success Rate by Method and Model")
    ax.set_xticklabels([METHOD_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()])
    ax.legend(title="Model")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    if d := os.path.dirname(output_path):
        os.makedirs(d, exist_ok=True)
    fig.savefig(output_path, dpi=150)
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
    """Bar chart: average API calls per problem by method."""
    avg = (
        df.groupby("method")["api_calls"]
        .mean()
        .reindex([m for m in METHOD_ORDER if m in df["method"].values])
        .reset_index()
    )
    avg.columns = ["method", "avg_api_calls"]
    avg["label"] = avg["method"].map(METHOD_LABELS).fillna(avg["method"])

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=avg, x="label", y="avg_api_calls", ax=ax, color="#5c85d6")
    ax.set_xlabel("Method")
    ax.set_ylabel("Avg API Calls per Problem")
    ax.set_title("Game of 24 — API Call Cost by Method")
    fig.tight_layout()
    if d := os.path.dirname(output_path):
        os.makedirs(d, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")
