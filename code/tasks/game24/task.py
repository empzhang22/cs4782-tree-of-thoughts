from __future__ import annotations
import re
import pandas as pd

from tot.node import ThoughtNode
from tasks.base_task import BaseTask
from .prompts import propose_prompt, value_prompt, value_last_step_prompt

_SAFE_CHARS = re.compile(r"^[0-9 +\-*/().]+$")
_STEP_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*[+\-*/]\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)\s*\(left:\s*([^)]+)\)"
)
# Fallback: matches "a op b = c" without (left: ...) for two-line format
_ARITH_ONLY_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)\s*$"
)
# Fallback: matches "Remaining numbers: x, y" or "Remaining: x y" etc.
_REMAINING_LABEL_RE = re.compile(
    r"(?:remaining(?:\s+numbers?)?|left)\s*[:\[\(]\s*(\d.*)", re.IGNORECASE
)


def _safe_eval(expr: str) -> float:
    expr = expr.strip()
    if not _SAFE_CHARS.match(expr):
        raise ValueError(f"Unsafe expression: {expr!r}")
    return eval(expr, {"__builtins__": {}}, {})  # noqa: S307


def _parse_remaining(text: str) -> list[float]:
    return [float(t) for t in re.findall(r"\d+(?:\.\d+)?", text)]


def _normalize_llm_output(text: str) -> str:
    """Strip LaTeX and markdown formatting so _STEP_RE can match."""
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)   # \text{left: } → left:
    text = text.replace('\\times', '*').replace('\\div', '/')
    text = re.sub(r'\\\(|\\\)|\\\[|\\\]', '', text)     # math delimiters
    text = re.sub(r'\\(?:quad|,|;|!|:)', ' ', text)     # spacing macros
    text = re.sub(r'\\\\|\\$', ' ', text, flags=re.MULTILINE)  # LaTeX \\ newline / trailing \
    text = re.sub(r'\\[a-zA-Z]+\*?', ' ', text)         # \begin \end \boxed etc.
    text = re.sub(r'[{}&]', '', text)                    # braces and & (aligned env)
    text = re.sub(r'\*{2,}', '', text)                   # markdown **bold**
    text = re.sub(r'[ \t]{2,}', ' ', text)               # collapse whitespace
    return text


class Game24Task(BaseTask):
    def __init__(self, dataset_path: str, problem_offset: int = 0):
        df = pd.read_csv(dataset_path)
        self._puzzles: list[str] = df["Puzzles"].tolist()
        self.offset = problem_offset

    def __len__(self) -> int:
        return len(self._puzzles)

    def get_input(self, idx: int) -> str:
        return self._puzzles[self.offset + idx]

    def make_root(self, idx: int) -> ThoughtNode:
        puzzle = self.get_input(idx)
        numbers = [float(x) for x in puzzle.split()]
        return ThoughtNode(
            thought=puzzle,
            step_outputs={"remaining": numbers, "expression": ""},
        )

    # ------------------------------------------------------------------
    # Prompt builders (passed to ThoughtGenerator / ThoughtEvaluator)
    # ------------------------------------------------------------------

    def propose_prompt_fn(self, node: ThoughtNode) -> str:
        remaining = node.step_outputs.get("remaining", [])
        inp = " ".join(str(int(x) if x == int(x) else x) for x in remaining)
        return propose_prompt.format(input=inp)

    def value_prompt_fn(self, node: ThoughtNode) -> str:
        remaining = node.step_outputs.get("remaining", [])
        if len(remaining) == 1:
            # Last step: validate the final expression
            expr = node.step_outputs.get("expression", "")
            original = node.get_path()[0].thought
            return value_last_step_prompt.format(input=original, answer=expr)
        inp = " ".join(str(int(x) if x == int(x) else x) for x in remaining)
        return value_prompt.format(input=inp)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_propose(self, text: str) -> list[str]:
        """Parse numbered/line-delimited steps from propose output."""
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        steps = []
        for line in lines:
            # Strip leading number+dot or bullet
            line = re.sub(r"^\d+[.)]\s*", "", line)
            if _STEP_RE.search(line):
                steps.append(line)
        return steps

    def parse_propose_with_state(self, text: str, node: ThoughtNode) -> list[tuple[str, dict]]:
        """Return (thought, step_outputs) pairs for each valid proposed step."""
        text = _normalize_llm_output(text)
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        results = []
        for line in lines:
            line = re.sub(r"^\d+[.)]\s*", "", line)
            m = _STEP_RE.search(line)
            if m:
                remaining_str = m.group(4)
                remaining = _parse_remaining(remaining_str)
                prev_expr = node.step_outputs.get("expression", "")
                expr = (prev_expr + " | " + line) if prev_expr else line
                results.append((line, {"remaining": remaining, "expression": expr}))

        if not results:
            # Fallback: handle "a op b = c" on one line + "Remaining: x y" on the next
            for i, line in enumerate(lines):
                m = _ARITH_ONLY_RE.search(re.sub(r"^(?:\d+[.)]\s*|[-*]\s+)", "", line))
                if not m:
                    continue
                for j in range(i + 1, min(i + 4, len(lines))):
                    rm = _REMAINING_LABEL_RE.search(lines[j])
                    if rm:
                        remaining = _parse_remaining(rm.group(1))
                        if remaining:
                            rem_str = " ".join(
                                str(int(x) if x == int(x) else x) for x in remaining
                            )
                            thought = f"{m.group(1)} {m.group(2)} {m.group(3)} = {m.group(4)} (left: {rem_str})"
                            prev_expr = node.step_outputs.get("expression", "")
                            expr = (prev_expr + " | " + thought) if prev_expr else thought
                            results.append((thought, {"remaining": remaining, "expression": expr}))
                        break

        return results

    def is_terminal(self, node: ThoughtNode) -> bool:
        remaining = node.step_outputs.get("remaining", [])
        return len(remaining) == 1

    def is_success(self, node: ThoughtNode) -> bool:
        remaining = node.step_outputs.get("remaining", [])
        if not remaining:
            return False
        try:
            return abs(remaining[0] - 24) < 1e-6
        except (TypeError, ValueError):
            return False

    def success_from_expression(self, expr: str, puzzle: str | None = None) -> bool:
        """Validate a final arithmetic expression equals 24 using exactly the puzzle numbers."""
        try:
            clean = re.sub(r'\s*=\s*[\d.]+\s*$', '', expr.strip())
            if abs(_safe_eval(clean) - 24) >= 1e-6:
                return False
            if puzzle is not None:
                expected = sorted(float(x) for x in puzzle.split())
                used = sorted(float(x) for x in re.findall(r'\d+(?:\.\d+)?', clean))
                return used == expected
            return True
        except Exception:
            return False
