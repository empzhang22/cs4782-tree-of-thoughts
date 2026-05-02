from __future__ import annotations
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.openai_client import OpenAIClient
from llm.gemini_client import GeminiClient
from llm.cache import ResponseCache
from tasks.game24.task import Game24Task
from tot.node import ThoughtNode
from tot.generator import ThoughtGenerator
from tot.evaluator import ThoughtEvaluator
from tot.bfs import bfs
from baselines.io_prompting import run_io
from baselines.cot_prompting import run_cot
from baselines.cot_sc import run_cot_sc


def _extract_answer_expr(node: ThoughtNode) -> str:
    expr = node.step_outputs.get("expression", "")
    if "=" in expr:
        return expr.split("|")[-1].strip()
    return expr


def run(cfg: dict, evaluate_only: bool = False):
    task_name = cfg["task"]
    method = cfg["method"]
    model = cfg["model"]
    n_problems = cfg.get("n_problems", 100)
    offset = cfg.get("problem_offset", 0)
    output_dir = cfg["output_dir"]
    cache_dir = cfg.get("cache_dir", "../results/cache")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    cache = ResponseCache(cache_dir)
    if model.startswith("gemini"):
        llm = GeminiClient(model=model, cache=cache)
    else:
        llm = OpenAIClient(model=model, cache=cache)

    if task_name == "game24":
        task = Game24Task(
            dataset_path=cfg["dataset_path"],
            problem_offset=offset,
        )
    else:
        raise ValueError(f"Unknown task: {task_name}")

    # Build ToT components once (reused across problems)
    if method == "tot_bfs":
        generator = ThoughtGenerator(llm, task.propose_prompt_fn)
        evaluator = ThoughtEvaluator(llm, task.value_prompt_fn)

        def propose_parse(text: str, node: ThoughtNode):
            return task.parse_propose_with_state(text, node)

    output_path = os.path.join(output_dir, "results.jsonl")

    with open(output_path, "w") as out:
        for i in range(n_problems):
            problem_id = offset + i + 1
            puzzle = task.get_input(i)
            t0 = time.time()
            llm.call_count = 0

            if method == "io":
                result = run_io(llm, puzzle,
                                temperature=cfg.get("temperature_generate", 0.7),
                                max_tokens=cfg.get("max_tokens", 128))
                success = task.success_from_expression(result["output"], puzzle)
                output = result["output"]

            elif method == "cot":
                result = run_cot(llm, puzzle,
                                 temperature=cfg.get("temperature_generate", 0.7),
                                 max_tokens=cfg.get("max_tokens", 256))
                success = task.success_from_expression(result["output"], puzzle)
                output = result["output"]

            elif method == "cot_sc":
                result = run_cot_sc(llm, puzzle,
                                    n_samples=cfg.get("n_generate", 5),
                                    temperature=cfg.get("temperature_generate", 0.7),
                                    max_tokens=cfg.get("max_tokens", 256))
                success = task.success_from_expression(result["output"], puzzle)
                output = result["output"]

            elif method == "tot_bfs":
                root = task.make_root(i)
                frontier = bfs(
                    root=root,
                    generator=generator,
                    evaluator=evaluator,
                    propose_parse_fn=propose_parse,
                    is_terminal_fn=task.is_terminal,
                    max_depth=cfg.get("max_depth", 3),
                    breadth_limit=cfg.get("breadth_limit", 5),
                    temperature_generate=cfg.get("temperature_generate", 0.7),
                    temperature_evaluate=cfg.get("temperature_evaluate", 0.0),
                    n_generate=cfg.get("n_generate", 5),
                )
                success = any(task.is_success(n) for n in frontier)
                best = frontier[0] if frontier else root
                output = _extract_answer_expr(best)

            else:
                raise ValueError(f"Unknown method: {method}")

            latency = time.time() - t0
            record = {
                "problem_id": problem_id,
                "input": puzzle,
                "output": output,
                "success": success,
                "api_calls": llm.call_count,
                "latency_s": round(latency, 2),
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            status = "✓" if success else "✗"
            print(f"[{i+1}/{n_problems}] {status} {puzzle} → {output} ({latency:.1f}s)")

    cache.close()
    print(f"\nResults saved to {output_path}")
