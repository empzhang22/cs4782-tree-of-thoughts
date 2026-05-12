#!/usr/bin/env python3
"""CLI entry point for running Tree of Thoughts experiments."""
import argparse
import os
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: dict) -> dict:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for key in ("dataset_path", "cache_dir", "output_dir"):
        value = cfg.get(key)
        if isinstance(value, str) and not os.path.isabs(value):
            cfg[key] = os.path.normpath(os.path.join(base_dir, value))
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Run a ToT experiment from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--n-problems", type=int, default=None, help="Override n_problems from config")
    parser.add_argument("--problem-offset", type=int, default=None, help="Override problem_offset from config")
    parser.add_argument("--dry-run", action="store_true", help="Estimate API call count without running")
    parser.add_argument("--evaluate", action="store_true", help="Re-evaluate existing JSONL results")
    parser.add_argument("--resume", action="store_true", help="Append to existing results and skip completed problem_ids")
    args = parser.parse_args()

    cfg = resolve_paths(load_config(args.config))
    if args.n_problems is not None:
        cfg["n_problems"] = args.n_problems
    if args.problem_offset is not None:
        cfg["problem_offset"] = args.problem_offset
    if args.resume:
        cfg["resume"] = True

    print("Config loaded:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    if args.dry_run:
        n = cfg.get("n_problems", 100)
        b = cfg.get("breadth_limit", 5)
        T = cfg.get("max_depth", 3)
        k = cfg.get("n_generate", 5)
        e = cfg.get("n_evaluate", 5)
        n_samples = cfg.get("n_generate", 1)
        method = cfg.get("method", "unknown")
        task = cfg.get("task", "unknown")
        if task == "creative_writing":
            if method in {"io", "cot"}:
                estimate = n * n_samples
            elif method == "cot_sc":
                estimate = n * (k + e)
            elif method == "tot_bfs":
                estimate = n * (2 * k + 2 * e)
            else:
                estimate = n
        elif task == "crossword":
            if method in {"io", "cot", "cot_sc"}:
                estimate = n * n_samples
            elif method == "tot_dfs":
                # Proposal calls plus value checks vary with pruning/backtracking.
                estimate = n * cfg.get("max_steps", 10) * (k + cfg.get("n_max_propose", 5))
            else:
                estimate = n
        elif "tot" in method:
            # depth 1: 1 propose + k value calls; depths 2..T: b propose + b*k value calls each
            estimate = n * ((1 + k) + (T - 1) * b * (1 + k))
        elif method == "cot_sc":
            estimate = n * n_samples
        else:
            estimate = n
        print(f"\nDry-run estimate: ~{estimate} LLM calls for {n} problems")
        return

    from experiments.runner import run
    run(cfg, evaluate_only=args.evaluate)


if __name__ == "__main__":
    main()
