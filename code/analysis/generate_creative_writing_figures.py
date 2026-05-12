#!/usr/bin/env python3
"""Generate Creative Writing summary tables and figures from experiment results.

Usage:
    python code/analysis/generate_creative_writing_figures.py
    python code/analysis/generate_creative_writing_figures.py --results-dir results/creative_writing --output-dir results/creative_writing/figures
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import analysis.poster_style  # noqa: F401 - applies shared rcParams
from experiments.evaluate import load_results

MODEL_NAME = "gemini-2.5-flash"

METHOD_DIR_MAP = {
    "io": "io_gemini25flash",
    "cot": "cot_gemini25flash",
    "cot_sc": "cot_sc_gemini25flash",
    "tot_bfs": "tot_bfs_gemini25flash",
}

METHOD_LABELS = {
    "io": "IO",
    "cot": "CoT",
    "cot_sc": "CoT-SC",
    "tot_bfs": "ToT (BFS)",
}

METHOD_ORDER = ["io", "cot", "cot_sc", "tot_bfs"]
METHOD_COLORS = {
    "io": "#d62728",
    "cot": "#ff7f0e",
    "cot_sc": "#2ca02c",
    "tot_bfs": "#1f77b4",
}


def discover_results(results_dir: str) -> dict[str, str]:
    found = {}
    for method, dirname in METHOD_DIR_MAP.items():
        candidate = os.path.join(results_dir, dirname, "results.jsonl")
        if os.path.isfile(candidate):
            found[method] = candidate
        else:
            print(f"[warn] Not found: {candidate}")
    if not found:
        raise FileNotFoundError(f"No result files found under {results_dir}")
    print(f"Discovered {len(found)} result files: {list(found.keys())}")
    return found


def load_all(result_paths: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_frames = []
    summary_rows = []
    for method, path in result_paths.items():
        df = load_results(path)
        df["method"] = method
        df["model"] = MODEL_NAME
        raw_frames.append(df)

        row = {
            "model": MODEL_NAME,
            "method": method,
            "n": len(df),
            "constraint_rate": round(df["constraint_satisfied"].mean(), 4)
            if "constraint_satisfied" in df
            else round(df["success"].mean(), 4),
            "avg_api_calls": round(df["api_calls"].mean(), 2),
            "total_api_calls": int(df["api_calls"].sum()),
            "avg_latency_s": round(df["latency_s"].mean(), 2),
        }
        if "coherency_score" in df:
            row["avg_coherency_score"] = round(df["coherency_score"].mean(), 2)
        summary_rows.append(row)

    return pd.DataFrame(summary_rows), pd.concat(raw_frames, ignore_index=True)


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    present = [m for m in METHOD_ORDER if m in set(df["method"])]
    return df.set_index("method").loc[present].reset_index()


def plot_constraint_rate(summary_df: pd.DataFrame, output_path: str) -> None:
    plot_df = _ordered(summary_df)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=plot_df,
        x="method",
        y="constraint_rate",
        order=plot_df["method"],
        palette={m: METHOD_COLORS[m] for m in plot_df["method"]},
        ax=ax,
        edgecolor="white",
        linewidth=0.5,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt=lambda v: f"{v * 100:.0f}%", padding=3, fontsize=9)
    ax.set_xlabel("Method", labelpad=8)
    ax.set_ylabel("Constraint Satisfaction Rate", labelpad=8)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0))
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([METHOD_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()])
    ax.set_ylim(0, 1.05)
    ax.set_title("Creative Writing - Constraint Satisfaction", pad=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_coherency(summary_df: pd.DataFrame, output_path: str) -> bool:
    if "avg_coherency_score" not in summary_df:
        print("[warn] No coherency_score column found. Run experiments with --evaluate before plotting coherency.")
        return False
    plot_df = _ordered(summary_df.dropna(subset=["avg_coherency_score"]))
    if plot_df.empty:
        print("[warn] No scored runs available for coherency plot.")
        return False

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=plot_df,
        x="method",
        y="avg_coherency_score",
        order=plot_df["method"],
        palette={m: METHOD_COLORS[m] for m in plot_df["method"]},
        ax=ax,
        edgecolor="white",
        linewidth=0.5,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)
    ax.set_xlabel("Method", labelpad=8)
    ax.set_ylabel("Average LM Coherency Score", labelpad=8)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([METHOD_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()])
    ax.set_ylim(0, 10.5)
    ax.set_title("Creative Writing - Coherency Score", pad=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return True


def _scored_runs(raw_df: pd.DataFrame) -> pd.DataFrame:
    if "coherency_score" not in raw_df:
        return pd.DataFrame()
    scored = raw_df.dropna(subset=["coherency_score"]).copy()
    if scored.empty:
        return scored
    scored = scored[scored["method"].isin(METHOD_ORDER)]
    scored["method_label"] = scored["method"].map(METHOD_LABELS).fillna(scored["method"])
    return scored


def plot_coherency_boxplot(raw_df: pd.DataFrame, output_path: str) -> bool:
    plot_df = _scored_runs(raw_df)
    if plot_df.empty:
        print("[warn] No scored runs available for coherency box plot.")
        return False

    order = [m for m in METHOD_ORDER if m in set(plot_df["method"])]
    label_order = [METHOD_LABELS[m] for m in order]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(
        data=plot_df,
        x="method_label",
        y="coherency_score",
        order=label_order,
        palette={METHOD_LABELS[m]: METHOD_COLORS[m] for m in order},
        width=0.55,
        fliersize=3,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="method_label",
        y="coherency_score",
        order=label_order,
        color="#222222",
        alpha=0.35,
        size=3,
        jitter=0.18,
        ax=ax,
    )
    ax.set_xlabel("Method", labelpad=8)
    ax.set_ylabel("LM Coherency Score", labelpad=8)
    ax.set_ylim(0, 10.5)
    ax.set_title("Creative Writing - Coherency Score Distribution", pad=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return True


def plot_coherency_sample_boxplot(raw_df: pd.DataFrame, output_path: str) -> bool:
    if "coherency_scores" not in raw_df:
        print("[warn] No coherency_scores column found. Run experiments with --evaluate first.")
        return False

    rows = []
    for _, record in raw_df.iterrows():
        method = record.get("method")
        scores = record.get("coherency_scores")
        if method not in METHOD_ORDER or not isinstance(scores, list):
            continue
        for score in scores:
            rows.append({
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "score": score,
            })

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        print("[warn] No individual coherency score samples available for box plot.")
        return False

    order = [m for m in METHOD_ORDER if m in set(plot_df["method"])]
    label_order = [METHOD_LABELS[m] for m in order]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(
        data=plot_df,
        x="method_label",
        y="score",
        order=label_order,
        palette={METHOD_LABELS[m]: METHOD_COLORS[m] for m in order},
        width=0.55,
        fliersize=3,
        ax=ax,
    )
    ax.set_xlabel("Method", labelpad=8)
    ax.set_ylabel("Individual LM Coherency Score Samples", labelpad=8)
    ax.set_ylim(0, 10.5)
    ax.set_title("Creative Writing - Individual Score Samples", pad=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return True


def plot_api_cost(summary_df: pd.DataFrame, output_path: str) -> None:
    plot_df = _ordered(summary_df)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=plot_df,
        x="method",
        y="avg_api_calls",
        order=plot_df["method"],
        palette={m: METHOD_COLORS[m] for m in plot_df["method"]},
        ax=ax,
        edgecolor="white",
        linewidth=0.5,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3, fontsize=9)
    ax.set_xlabel("Method", labelpad=8)
    ax.set_ylabel("Avg API Calls per Prompt", labelpad=8)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([METHOD_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()])
    ax.set_title("Creative Writing - API Call Cost", pad=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(script_dir, "../.."))
    parser = argparse.ArgumentParser(
        description="Generate Creative Writing result figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", default=os.path.join(repo_root, "results/creative_writing"))
    parser.add_argument("--output-dir", default=os.path.join(repo_root, "results/creative_writing/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    result_paths = discover_results(args.results_dir)
    summary_df, raw_df = load_all(result_paths)

    summary_path = os.path.join(args.output_dir, "summary.csv")
    raw_path = os.path.join(args.output_dir, "all_results.csv")
    summary_df.to_csv(summary_path, index=False)
    raw_df.to_csv(raw_path, index=False)

    print("\nSummary:")
    columns = [c for c in ["method", "constraint_rate", "avg_coherency_score", "avg_api_calls", "total_api_calls"] if c in summary_df]
    print(summary_df[columns].to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {raw_path}")

    for ext in ("png", "pdf"):
        plot_constraint_rate(summary_df, os.path.join(args.output_dir, f"constraint_satisfaction.{ext}"))
        plot_api_cost(summary_df, os.path.join(args.output_dir, f"api_cost.{ext}"))
        plot_coherency(summary_df, os.path.join(args.output_dir, f"coherency_score.{ext}"))
        plot_coherency_boxplot(raw_df, os.path.join(args.output_dir, f"coherency_score_boxplot.{ext}"))
        plot_coherency_sample_boxplot(raw_df, os.path.join(args.output_dir, f"coherency_score_samples_boxplot.{ext}"))

    print(f"\nAll Creative Writing outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
