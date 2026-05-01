from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Optional


@dataclass
class ThoughtNode:
    thought: str
    parent: Optional["ThoughtNode"] = field(default=None, repr=False)
    children: list["ThoughtNode"] = field(default_factory=list, repr=False)
    depth: int = 0
    value: float = math.nan
    step_outputs: dict = field(default_factory=dict)

    def get_path(self) -> list["ThoughtNode"]:
        """Return chain from root to self (inclusive)."""
        path = []
        node: Optional[ThoughtNode] = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def add_child(self, thought: str, step_outputs: dict | None = None) -> "ThoughtNode":
        child = ThoughtNode(
            thought=thought,
            parent=self,
            depth=self.depth + 1,
            step_outputs=step_outputs or {},
        )
        self.children.append(child)
        return child
