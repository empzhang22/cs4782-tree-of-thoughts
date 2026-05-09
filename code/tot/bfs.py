from __future__ import annotations
import math
from typing import Callable

from .node import ThoughtNode
from .generator import ThoughtGenerator
from .evaluator import ThoughtEvaluator


def bfs(
    root: ThoughtNode,
    generator: ThoughtGenerator,
    evaluator: ThoughtEvaluator,
    propose_parse_fn: Callable[[str, ThoughtNode], list[tuple[str, dict]]],
    is_terminal_fn: Callable[[ThoughtNode], bool],
    max_depth: int = 3,
    breadth_limit: int = 5,
    temperature_generate: float = 0.7,
    temperature_evaluate: float = 0.0,
    n_generate: int = 5,
) -> list[ThoughtNode]:
    """BFS over thought tree. Returns final frontier nodes."""
    frontier = [root]

    for _ in range(max_depth):
        candidates: list[ThoughtNode] = []
        for node in frontier:
            if is_terminal_fn(node):
                candidates.append(node)
                continue
            children = generator.propose(
                node,
                propose_parse_fn,
                n_generate=n_generate,
                temperature=temperature_generate,
            )
            candidates.extend(children)

        # Evaluate all non-terminal candidates
        for node in candidates:
            if math.isnan(node.value) and not is_terminal_fn(node):
                evaluator.value(node, temperature=temperature_evaluate)

        # Keep top-b by value
        candidates.sort(key=lambda n: n.value, reverse=True)
        frontier = candidates[:breadth_limit]

        if all(is_terminal_fn(n) for n in frontier):
            break

    return frontier
