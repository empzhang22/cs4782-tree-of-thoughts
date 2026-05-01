from __future__ import annotations
from typing import Callable

from llm.base import LLMClient
from .node import ThoughtNode

_SCORE_MAP = {"sure": 1.0, "likely": 0.5, "impossible": 0.0}


class ThoughtEvaluator:
    """Assigns value scores to nodes using Value or Vote mode."""

    def __init__(self, llm: LLMClient, prompt_fn: Callable[[ThoughtNode], str]):
        self.llm = llm
        self.prompt_fn = prompt_fn

    def value(
        self,
        node: ThoughtNode,
        temperature: float = 0.0,
        max_tokens: int = 16,
    ) -> float:
        """Single call → sure/likely/impossible → numeric score."""
        prompt = self.prompt_fn(node)
        response = self.llm.complete(prompt, n=1, temperature=temperature, max_tokens=max_tokens)[0].lower()
        for label, score in _SCORE_MAP.items():
            if label in response:
                node.value = score
                return score
        node.value = 0.0
        return 0.0

    def vote(
        self,
        nodes: list[ThoughtNode],
        prompt_fn: Callable[[list[ThoughtNode]], str],
        temperature: float = 0.0,
        max_tokens: int = 16,
    ) -> ThoughtNode:
        """Single call across all nodes → pick best by majority vote."""
        prompt = prompt_fn(nodes)
        response = self.llm.complete(prompt, n=1, temperature=temperature, max_tokens=max_tokens)[0]
        # Try to parse an index from the response
        import re
        match = re.search(r"\b([1-9]\d*)\b", response)
        idx = (int(match.group(1)) - 1) if match else 0
        idx = max(0, min(idx, len(nodes) - 1))
        best = nodes[idx]
        best.value = 1.0
        return best
