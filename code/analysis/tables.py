"""Generate result tables (CSV + LaTeX) from experiment summaries."""
from __future__ import annotations
import os
import pandas as pd


def build_summary_table(results: list[dict]) -> pd.DataFrame:
    """
    results: list of dicts with keys: model, method, success_rate, avg_api_calls, avg_latency_s
    """
    df = pd.DataFrame(results)
    df["success_pct"] = (df["success_rate"] * 100).round(1)
    return df


def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"CSV saved: {path}")


def save_latex(df: pd.DataFrame, path: str, caption: str = "Game of 24 Results"):
    cols = ["model", "method", "success_pct", "avg_api_calls"]
    latex = df[cols].to_latex(
        index=False,
        caption=caption,
        label="tab:game24_results",
        float_format="%.1f",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(latex)
    print(f"LaTeX saved: {path}")
