#!/usr/bin/env python3
"""Compare saved crossword outputs against gold 5x5 answers."""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tasks.crossword.task import CrosswordTask  # noqa: E402


def load_jsonl(path: str) -> list[dict]:
    records = []
    decoder = json.JSONDecoder()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            while line:
                record, idx = decoder.raw_decode(line)
                records.append(record)
                line = line[idx:].strip()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score crossword JSONL result files against the complete crossword dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("results", nargs="+", help="One or more results.jsonl files")
    parser.add_argument("--dataset-path", default=os.path.join("data", "crossword", "crosswords_5x5_complete.json"))
    parser.add_argument("--csv", default=None, help="Optional CSV path for the summary table")
    args = parser.parse_args()

    task = CrosswordTask(args.dataset_path)
    item_by_id = {item["id"]: item for item in task.test_items}
    rows = []

    for path in args.results:
        records = load_jsonl(path)
        deduped = {}
        for record in records:
            deduped[record.get("problem_id")] = record
        records = [deduped[key] for key in sorted(deduped)]
        scored = []
        for record in records:
            puzzle_id = record.get("puzzle_id", record.get("problem_id"))
            item = item_by_id[puzzle_id]
            grid = record.get("grid") or task.parse_grid(record.get("output", ""))
            metrics = task.score_grid(grid, item)
            scored.append(metrics)

        if not scored:
            continue

        label = os.path.basename(os.path.dirname(path))
        row = {
            "run": label,
            "n": len(scored),
            "letter_accuracy": round(sum(m["letter_accuracy"] for m in scored) / len(scored), 4),
            "word_accuracy": round(sum(m["word_accuracy"] for m in scored) / len(scored), 4),
            "game_accuracy": round(sum(1 for m in scored if m["game_correct"]) / len(scored), 4),
        }
        rows.append(row)
        print(
            f"{label}: letters={row['letter_accuracy']:.1%} "
            f"words={row['word_accuracy']:.1%} games={row['game_accuracy']:.1%}"
        )

    summary = pd.DataFrame(rows)
    if args.csv:
        summary.to_csv(args.csv, index=False)
        print(f"\nSummary saved to {args.csv}")


if __name__ == "__main__":
    main()
