from __future__ import annotations
from typing import Callable

from llm.base import LLMClient
from .node import ThoughtNode


class ThoughtGenerator:
    """Generates child thoughts from a node using Propose or Sample mode."""

    def __init__(self, llm: LLMClient, prompt_fn: Callable[[ThoughtNode], str]):
        self.llm = llm
        self.prompt_fn = prompt_fn

    def propose(
        self,
        node: ThoughtNode,
        parse_fn: Callable[[str, ThoughtNode], list[tuple[str, dict]]],
        n_generate: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> list[ThoughtNode]:
        """Single LLM call → parse multiple candidates."""
        prompt = self.prompt_fn(node)
        responses = self.llm.complete(prompt, n=1, temperature=temperature, max_tokens=max_tokens)
        pairs = parse_fn(responses[0], node)
        return [node.add_child(thought, step_outputs=outputs) for thought, outputs in pairs[:n_generate]]

    def sample(
        self,
        node: ThoughtNode,
        parse_fn: Callable[[str, ThoughtNode], tuple[str, dict]],
        n_generate: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> list[ThoughtNode]:
        """n independent LLM calls → one thought each."""
        prompt = self.prompt_fn(node)
        responses = self.llm.complete(prompt, n=n_generate, temperature=temperature, max_tokens=max_tokens)
        return [node.add_child(thought, step_outputs=outputs) for thought, outputs in (parse_fn(r, node) for r in responses)]
