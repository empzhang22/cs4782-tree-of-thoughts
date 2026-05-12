from __future__ import annotations
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.cache import ResponseCache
from tasks.game24.task import Game24Task
from tasks.creative_writing.task import CreativeWritingTask
from tasks.crossword.task import CrosswordTask
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


def _select_by_vote(llm, task: CreativeWritingTask, nodes: list[ThoughtNode], n_votes: int, temperature: float, max_tokens: int):
    if len(nodes) == 1:
        nodes[0].value = 1.0
        return nodes[0], [1]

    prompt = task.vote_prompt_fn(nodes)
    responses = llm.complete(prompt, n=n_votes, temperature=temperature, max_tokens=max_tokens)
    vote_counts = [0] * len(nodes)
    for response in responses:
        vote_counts[task.parse_vote(response, len(nodes))] += 1
    for node, votes in zip(nodes, vote_counts):
        node.value = votes
    best_idx = max(range(len(nodes)), key=lambda i: vote_counts[i])
    return nodes[best_idx], vote_counts


def _run_creative_io(llm, task: CreativeWritingTask, prompt_input: str, n_samples: int, temperature: float, max_tokens: int) -> dict:
    prompt = task.standard_prompt_fn(prompt_input)
    responses = llm.complete(prompt, n=n_samples, temperature=temperature, max_tokens=max_tokens)
    outputs = [task.extract_passage(r) for r in responses]
    return {"output": outputs[0], "all_outputs": outputs}


def _run_creative_cot(llm, task: CreativeWritingTask, prompt_input: str, n_samples: int, temperature: float, max_tokens: int) -> dict:
    prompt = task.cot_prompt_fn(prompt_input)
    responses = llm.complete(prompt, n=n_samples, temperature=temperature, max_tokens=max_tokens)
    outputs = [task.extract_passage(r) for r in responses]
    plans = [task.extract_plan(r) for r in responses]
    return {"output": outputs[0], "plan": plans[0] if plans else "", "all_outputs": outputs, "all_plans": plans}


def _run_creative_cot_sc(
    llm,
    task: CreativeWritingTask,
    root: ThoughtNode,
    n_samples: int,
    n_votes: int,
    temperature_generate: float,
    temperature_evaluate: float,
    max_tokens_generate: int,
    max_tokens_evaluate: int,
) -> dict:
    prompt = task.cot_prompt_fn(root.thought)
    responses = llm.complete(prompt, n=n_samples, temperature=temperature_generate, max_tokens=max_tokens_generate)
    candidates = []
    for response in responses:
        passage = task.extract_passage(response)
        plan = task.extract_plan(response)
        candidates.append(root.add_child(passage, {"stage": "passage", "plan": plan, "passage": passage}))
    best, vote_counts = _select_by_vote(llm, task, candidates, n_votes, temperature_evaluate, max_tokens_evaluate)
    return {
        "output": best.step_outputs.get("passage", best.thought),
        "plan": best.step_outputs.get("plan", ""),
        "all_outputs": [c.step_outputs.get("passage", c.thought) for c in candidates],
        "vote_counts": vote_counts,
    }


def _run_creative_tot_bfs(
    llm,
    task: CreativeWritingTask,
    root: ThoughtNode,
    n_generate: int,
    n_votes: int,
    temperature_generate: float,
    temperature_evaluate: float,
    max_tokens_generate: int,
    max_tokens_evaluate: int,
) -> dict:
    generator = ThoughtGenerator(llm, task.sample_prompt_fn)
    plans = generator.sample(
        root,
        task.parse_sample,
        n_generate=n_generate,
        temperature=temperature_generate,
        max_tokens=max_tokens_generate,
    )
    best_plan, plan_vote_counts = _select_by_vote(
        llm, task, plans, n_votes, temperature_evaluate, max_tokens_evaluate
    )
    passages = generator.sample(
        best_plan,
        task.parse_sample,
        n_generate=n_generate,
        temperature=temperature_generate,
        max_tokens=max_tokens_generate,
    )
    best_passage, passage_vote_counts = _select_by_vote(
        llm, task, passages, n_votes, temperature_evaluate, max_tokens_evaluate
    )
    return {
        "output": best_passage.step_outputs.get("passage", best_passage.thought),
        "plan": best_plan.step_outputs.get("plan", best_plan.thought),
        "all_plans": [p.step_outputs.get("plan", p.thought) for p in plans],
        "all_outputs": [p.step_outputs.get("passage", p.thought) for p in passages],
        "plan_vote_counts": plan_vote_counts,
        "passage_vote_counts": passage_vote_counts,
    }


def _run_crossword_io(llm, task: CrosswordTask, clue_text: str, n_samples: int, temperature: float, max_tokens: int) -> dict:
    prompt = task.standard_prompt_fn(clue_text)
    responses = llm.complete(prompt, n=n_samples, temperature=temperature, max_tokens=max_tokens)
    grids = [task.parse_grid(response) for response in responses]
    return {"output": task.format_grid(grids[0]) if grids[0] else responses[0], "all_outputs": responses, "grid": grids[0]}


def _run_crossword_cot(llm, task: CrosswordTask, clue_text: str, n_samples: int, temperature: float, max_tokens: int) -> dict:
    prompt = task.cot_prompt_fn(clue_text)
    responses = llm.complete(prompt, n=n_samples, temperature=temperature, max_tokens=max_tokens)
    grids = [task.parse_grid(response) for response in responses]
    return {"output": task.format_grid(grids[0]) if grids[0] else responses[0], "all_outputs": responses, "grid": grids[0]}


def _run_crossword_cot_sc(llm, task: CrosswordTask, clue_text: str, n_samples: int, temperature: float, max_tokens: int) -> dict:
    prompt = task.cot_prompt_fn(clue_text)
    responses = llm.complete(prompt, n=n_samples, temperature=temperature, max_tokens=max_tokens)
    grids = [task.parse_grid(response) for response in responses]
    grid_keys = [
        "\n".join("".join(row) for row in grid) if grid else ""
        for grid in grids
    ]
    majority_key = Counter(grid_keys).most_common(1)[0][0]
    best_idx = grid_keys.index(majority_key)
    grid = grids[best_idx]
    return {
        "output": task.format_grid(grid) if grid else responses[best_idx],
        "all_outputs": responses,
        "grid": grid,
        "vote_counts": dict(Counter(grid_keys)),
    }


def _evaluate_crossword_state(
    llm,
    task: CrosswordTask,
    item: dict,
    state,
    temperature: float,
    max_tokens: int,
) -> float:
    entries = task.constrained_entries(item, state)
    if not entries:
        return 0.1
    total = 0.0
    for _, clue, pattern in entries:
        prompt = task.value_prompt_fn(clue, pattern)
        response = llm.complete(prompt, n=1, temperature=temperature, max_tokens=max_tokens)[0]
        score = task.parse_value(response)
        if score == 0.0:
            return 0.0
        total += score
    return total / len(entries)


def _run_crossword_tot_dfs(
    llm,
    task: CrosswordTask,
    root: ThoughtNode,
    n_generate: int,
    n_max_propose: int,
    max_steps: int,
    prune_threshold: float,
    temperature_generate: float,
    temperature_evaluate: float,
    max_tokens_generate: int,
    max_tokens_evaluate: int,
) -> dict:
    item = root.step_outputs["item"]
    best = {
        "state": root.step_outputs["state"],
        "metrics": task.score_grid(task.state_to_grid(root.step_outputs["state"]), item),
        "steps": [],
    }

    def consider(state):
        grid = task.state_to_grid(state)
        metrics = task.score_grid(grid, item)
        current_key = (
            task.is_filled(state),
            metrics["game_correct"],
            metrics["words_correct"],
            metrics["letters_correct"],
        )
        best_key = (
            task.is_filled(best["state"]),
            best["metrics"]["game_correct"],
            best["metrics"]["words_correct"],
            best["metrics"]["letters_correct"],
        )
        if current_key > best_key:
            best["state"] = state
            best["metrics"] = metrics
            best["steps"] = list(state.steps)
        return metrics["game_correct"]

    def dfs(state, depth: int) -> bool:
        solved = consider(state)
        if solved:
            return True
        if depth >= max_steps or task.is_filled(state):
            return best["metrics"]["game_correct"]

        node = ThoughtNode(root.thought, step_outputs={"item": item, "state": state})
        prompt = task.propose_prompt_fn(node)
        responses = llm.complete(prompt, n=n_generate, temperature=temperature_generate, max_tokens=max_tokens_generate)
        proposals = []
        seen = set()
        for response in responses:
            for key, word, confidence_score in task.parse_proposals(response):
                proposal_id = (key, word)
                if proposal_id in seen:
                    continue
                seen.add(proposal_id)
                proposals.append((key, word, confidence_score))
        proposals.sort(key=lambda p: p[2], reverse=True)

        for key, word, _ in proposals[:n_max_propose]:
            next_state = task.apply_word(state, key, word)
            state_score = _evaluate_crossword_state(
                llm,
                task,
                item,
                next_state,
                temperature=temperature_evaluate,
                max_tokens=max_tokens_evaluate,
            )
            if state_score <= prune_threshold:
                continue
            if dfs(next_state, depth + 1):
                return True
        return False

    dfs(root.step_outputs["state"], 0)
    grid = task.state_to_grid(best["state"])
    finish_output = None
    if not task.is_filled(best["state"]):
        finish_prompt = task.finish_prompt_fn(item, best["state"])
        finish_output = llm.complete(
            finish_prompt,
            n=1,
            temperature=temperature_generate,
            max_tokens=max_tokens_generate,
        )[0]
        finished_grid = task.parse_grid(finish_output)
        if finished_grid is not None:
            finished_metrics = task.score_grid(finished_grid, item)
            if (
                True,
                finished_metrics["game_correct"],
                finished_metrics["words_correct"],
                finished_metrics["letters_correct"],
            ) >= (
                task.is_filled(best["state"]),
                best["metrics"]["game_correct"],
                best["metrics"]["words_correct"],
                best["metrics"]["letters_correct"],
            ):
                grid = finished_grid
                best["metrics"] = finished_metrics
    return {
        "output": task.format_grid(grid),
        "grid": grid,
        "steps": best["steps"],
        "search_metrics": best["metrics"],
        "finish_output": finish_output,
    }


def _evaluate_creative_outputs(
    llm,
    task: CreativeWritingTask,
    output_path: str,
    n_scores: int,
    temperature: float,
    max_tokens: int,
) -> None:
    records = []
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    for i, record in enumerate(records):
        llm.call_count = 0
        prompt = task.score_prompt_fn(record["output"])
        responses = llm.complete(prompt, n=n_scores, temperature=temperature, max_tokens=max_tokens)
        scores = [score for response in responses if (score := task.parse_score(response)) is not None]
        record["coherency_scores"] = scores
        record["coherency_score"] = round(sum(scores) / len(scores), 2) if scores else 0
        record["eval_api_calls"] = llm.call_count
        print(f"[{i+1}/{len(records)}] score={record['coherency_score']} calls={llm.call_count}")

    with open(output_path, "w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record) + "\n")

    print(f"\nUpdated scored results in {output_path}")


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
        from llm.gemini_client import GeminiClient
        llm = GeminiClient(model=model, cache=cache, rpm=cfg.get("rpm", 1000))
    else:
        from llm.openai_client import OpenAIClient
        llm = OpenAIClient(model=model, cache=cache)

    if task_name == "game24":
        task = Game24Task(
            dataset_path=cfg["dataset_path"],
            problem_offset=offset,
        )
    elif task_name == "creative_writing":
        task = CreativeWritingTask(
            dataset_path=cfg["dataset_path"],
            problem_offset=offset,
        )
    elif task_name == "crossword":
        task = CrosswordTask(
            dataset_path=cfg["dataset_path"],
            problem_offset=offset,
        )
    else:
        raise ValueError(f"Unknown task: {task_name}")

    # Build ToT components once (reused across problems)
    if task_name == "game24" and method == "tot_bfs":
        generator = ThoughtGenerator(llm, task.propose_prompt_fn)
        evaluator = ThoughtEvaluator(llm, task.value_prompt_fn)

        def propose_parse(text: str, node: ThoughtNode):
            return task.parse_propose_with_state(text, node)

    output_path = os.path.join(output_dir, "results.jsonl")

    if evaluate_only:
        if task_name != "creative_writing":
            raise ValueError("--evaluate is only implemented for creative_writing results.")
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"No existing results to evaluate: {output_path}")
        _evaluate_creative_outputs(
            llm,
            task,
            output_path,
            n_scores=cfg.get("n_score", 5),
            temperature=cfg.get("temperature_score", 1.0),
            max_tokens=cfg.get("max_tokens_score", 256),
        )
        cache.close()
        return

    completed_problem_ids = set()
    if cfg.get("resume") and os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as existing:
            for line in existing:
                line = line.strip()
                if not line:
                    continue
                try:
                    completed_problem_ids.add(json.loads(line)["problem_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        if completed_problem_ids:
            print(f"Resume enabled: skipping {len(completed_problem_ids)} completed problem(s)")

    output_mode = "a" if cfg.get("resume") else "w"
    with open(output_path, output_mode, encoding="utf-8") as out:
        for i in range(n_problems):
            problem_id = offset + i + 1
            if problem_id in completed_problem_ids:
                continue
            puzzle = task.get_input(i)
            t0 = time.time()
            llm.call_count = 0

            if task_name == "creative_writing":
                root = task.make_root(i)
                if method == "io":
                    result = _run_creative_io(
                        llm,
                        task,
                        puzzle,
                        n_samples=cfg.get("n_generate", 1),
                        temperature=cfg.get("temperature_generate", 1.0),
                        max_tokens=cfg.get("max_tokens", 1024),
                    )
                elif method == "cot":
                    result = _run_creative_cot(
                        llm,
                        task,
                        puzzle,
                        n_samples=cfg.get("n_generate", 1),
                        temperature=cfg.get("temperature_generate", 1.0),
                        max_tokens=cfg.get("max_tokens", 1536),
                    )
                elif method == "cot_sc":
                    result = _run_creative_cot_sc(
                        llm,
                        task,
                        root,
                        n_samples=cfg.get("n_generate", 10),
                        n_votes=cfg.get("n_evaluate", 5),
                        temperature_generate=cfg.get("temperature_generate", 1.0),
                        temperature_evaluate=cfg.get("temperature_evaluate", 1.0),
                        max_tokens_generate=cfg.get("max_tokens", 1536),
                        max_tokens_evaluate=cfg.get("max_tokens_evaluate", 512),
                    )
                elif method == "tot_bfs":
                    result = _run_creative_tot_bfs(
                        llm,
                        task,
                        root,
                        n_generate=cfg.get("n_generate", 5),
                        n_votes=cfg.get("n_evaluate", 5),
                        temperature_generate=cfg.get("temperature_generate", 1.0),
                        temperature_evaluate=cfg.get("temperature_evaluate", 1.0),
                        max_tokens_generate=cfg.get("max_tokens", 1536),
                        max_tokens_evaluate=cfg.get("max_tokens_evaluate", 512),
                    )
                else:
                    raise ValueError(f"Unknown method for creative_writing: {method}")

                output = result["output"]
                success = task.constraints_satisfied(output, task.get_sentences(i))

            elif task_name == "crossword":
                item = task.get_item(i)
                root = task.make_root(i)
                if method == "io":
                    result = _run_crossword_io(
                        llm,
                        task,
                        puzzle,
                        n_samples=cfg.get("n_generate", 1),
                        temperature=cfg.get("temperature_generate", 0.7),
                        max_tokens=cfg.get("max_tokens", 512),
                    )
                elif method == "cot":
                    result = _run_crossword_cot(
                        llm,
                        task,
                        puzzle,
                        n_samples=cfg.get("n_generate", 1),
                        temperature=cfg.get("temperature_generate", 0.7),
                        max_tokens=cfg.get("max_tokens", 768),
                    )
                elif method == "cot_sc":
                    result = _run_crossword_cot_sc(
                        llm,
                        task,
                        puzzle,
                        n_samples=cfg.get("n_generate", 10),
                        temperature=cfg.get("temperature_generate", 0.7),
                        max_tokens=cfg.get("max_tokens", 768),
                    )
                elif method == "tot_dfs":
                    result = _run_crossword_tot_dfs(
                        llm,
                        task,
                        root,
                        n_generate=cfg.get("n_generate", 5),
                        n_max_propose=cfg.get("n_max_propose", 5),
                        max_steps=cfg.get("max_steps", 10),
                        prune_threshold=cfg.get("prune_threshold", 0.0),
                        temperature_generate=cfg.get("temperature_generate", 0.7),
                        temperature_evaluate=cfg.get("temperature_evaluate", 0.0),
                        max_tokens_generate=cfg.get("max_tokens", 512),
                        max_tokens_evaluate=cfg.get("max_tokens_evaluate", 256),
                    )
                else:
                    raise ValueError(f"Unknown method for crossword: {method}")

                output = result["output"]
                metrics = task.score_grid(result.get("grid"), item)
                success = metrics["game_correct"]

            elif method == "io":
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
                success_node = next((n for n in frontier if task.is_success(n)), None)
                success = success_node is not None
                best = success_node if success_node else (frontier[0] if frontier else root)
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
            if task_name == "creative_writing":
                record["sentences"] = task.get_sentences(i)
                record["constraint_satisfied"] = success
                for key in ("plan", "all_plans", "all_outputs", "vote_counts", "plan_vote_counts", "passage_vote_counts"):
                    if key in result:
                        record[key] = result[key]
            elif task_name == "crossword":
                record["puzzle_id"] = item["id"]
                record["grid"] = result.get("grid")
                record["target_grid"] = item["grid"]
                record["letter_accuracy"] = metrics["letter_accuracy"]
                record["word_accuracy"] = metrics["word_accuracy"]
                record["game_accuracy"] = 1.0 if metrics["game_correct"] else 0.0
                record["letters_correct"] = metrics["letters_correct"]
                record["words_correct"] = metrics["words_correct"]
                record["pred_words"] = metrics["pred_words"]
                if "steps" in result:
                    record["steps"] = result["steps"]
                if "all_outputs" in result:
                    record["all_outputs"] = result["all_outputs"]
                if "vote_counts" in result:
                    record["vote_counts"] = result["vote_counts"]
            out.write(json.dumps(record) + "\n")
            out.flush()
            status = "ok" if success else "fail"
            print(f"[{i+1}/{n_problems}] {status} {puzzle} -> {output} ({latency:.1f}s)")

    cache.close()
    print(f"\nResults saved to {output_path}")
