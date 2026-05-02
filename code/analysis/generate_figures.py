#!/usr/bin/env python3
"""
Generate all Game of 24 publication figures from experiment results.

Usage:
    python code/analysis/generate_figures.py
    python code/analysis/generate_figures.py --results-dir results/game24 --output-dir results/figures
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd
import analysis.poster_style  # noqa: F401 — applies poster rcParams and color overrides

from experiments.evaluate import load_results, summarize
from analysis.plots import (
    METHOD_ORDER,
    plot_success_rate,
    plot_api_cost,
    plot_difficulty_curve,
)

MODEL_NAME = "gemini-2.5-flash"

METHOD_DIR_MAP = {
    "io":      "io_gemini25flash",
    "cot":     "cot_gemini25flash",
    "cot_sc":  "cot_sc_gemini25flash",
    "tot_bfs": "tot_bfs_gemini25flash",
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


def load_all(result_paths: dict[str, str]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    summary_rows = []
    raw_dfs: dict[str, pd.DataFrame] = {}
    for method, path in result_paths.items():
        df = load_results(path)
        df["method"] = method
        df["model"] = MODEL_NAME
        raw_dfs[method] = df
        stats = summarize(df)
        summary_rows.append({
            "model":         MODEL_NAME,
            "method":        method,
            "success_rate":  stats["success_rate"],
            "avg_api_calls": stats["avg_api_calls"],
            "avg_latency_s": stats["avg_latency_s"],
            "n":             stats["n"],
        })
    return pd.DataFrame(summary_rows), raw_dfs


def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.normpath(os.path.join(script_dir, "../.."))
    p = argparse.ArgumentParser(
        description="Generate Game of 24 publication figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-dir",  default=os.path.join(repo_root, "results/game24"))
    p.add_argument("--output-dir",   default=os.path.join(repo_root, "results/figures"))
    p.add_argument("--dataset-path", default=os.path.join(repo_root, "data/game24/24.csv"))
    p.add_argument("--rolling-window", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    result_paths = discover_results(args.results_dir)
    summary_df, raw_dfs = load_all(result_paths)

    print("\nSummary:")
    print(summary_df[["method", "success_rate", "avg_api_calls"]].to_string(index=False))
    print()

    # Figure 1: Gemini-only success rate
    for ext in ("png", "pdf"):
        plot_success_rate(
            summary_df,
            os.path.join(args.output_dir, f"success_rate.{ext}"),
            include_paper_baselines=False,
        )

    # Figure 2: Gemini vs. GPT-4 paper comparison
    for ext in ("png", "pdf"):
        plot_success_rate(
            summary_df,
            os.path.join(args.output_dir, f"success_rate_comparison.{ext}"),
            include_paper_baselines=True,
        )

    # Figure 3: Difficulty curve
    for ext in ("png", "pdf"):
        plot_difficulty_curve(
            raw_dfs,
            args.dataset_path,
            os.path.join(args.output_dir, f"difficulty_curve.{ext}"),
            window=args.rolling_window,
        )

    # Figure 4: API cost
    all_raw = pd.concat(list(raw_dfs.values()), ignore_index=True)
    for ext in ("png", "pdf"):
        plot_api_cost(all_raw, os.path.join(args.output_dir, f"api_cost.{ext}"))

    print(f"\nAll figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
