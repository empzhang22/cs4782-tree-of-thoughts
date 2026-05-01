from __future__ import annotations
import math
from typing import Callable

from .node import ThoughtNode
from .generator import ThoughtGenerator
from .evaluator import ThoughtEvaluator


def dfs(
    root: ThoughtNode,
    generator: ThoughtGenerator,
    evaluator: ThoughtEvaluator,
    propose_parse_fn: Callable[[str, ThoughtNode], list[tuple[str, dict]]],
    is_terminal_fn: Callable[[ThoughtNode], bool],
    max_depth: int = 3,
    n_generate: int = 5,
    prune_threshold: float = 0.0,
    temperature_generate: float = 0.7,
    temperature_evaluate: float = 0.0,
) -> list[ThoughtNode]:
    """DFS with pruning. Returns all terminal nodes found."""
    results: list[ThoughtNode] = []

    def _recurse(node: ThoughtNode):
        if is_terminal_fn(node):
            results.append(node)
            return
        if node.depth >= max_depth:
            return

        children = generator.propose(
            node,
            propose_parse_fn,
            n_generate=n_generate,
            temperature=temperature_generate,
        )

        for child in children:
            evaluator.value(child, temperature=temperature_evaluate)
            if child.value <= prune_threshold and prune_threshold > 0.0:
                continue
            _recurse(child)

    _recurse(root)
    return results
