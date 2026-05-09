"""Aggregate metrics from one or more JSONL result files."""
from __future__ import annotations
import json
import os
import sys
import argparse

import pandas as pd


def load_results(jsonl_path: str) -> pd.DataFrame:
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def summarize(df: pd.DataFrame) -> dict:
    return {
        "n": len(df),
        "success_rate": round(df["success"].mean(), 4),
        "avg_api_calls": round(df["api_calls"].mean(), 2),
        "avg_latency_s": round(df["latency_s"].mean(), 2),
        "total_api_calls": int(df["api_calls"].sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="Path(s) to results.jsonl files")
    parser.add_argument("--csv", default=None, help="Save summary table to CSV")
    args = parser.parse_args()

    rows = []
    for path in args.results:
        df = load_results(path)
        summary = summarize(df)
        label = os.path.basename(os.path.dirname(path))
        rows.append({"run": label, **summary})
        print(f"{label}: success={summary['success_rate']:.1%}  "
              f"api_calls={summary['avg_api_calls']:.1f}  "
              f"latency={summary['avg_latency_s']:.1f}s")

    summary_df = pd.DataFrame(rows)
    if args.csv:
        summary_df.to_csv(args.csv, index=False)
        print(f"\nSummary saved to {args.csv}")

    return summary_df


if __name__ == "__main__":
    main()
