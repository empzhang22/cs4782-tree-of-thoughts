import re

from llm.base import LLMClient
from tasks.game24.prompts import standard_prompt

_ANSWER_RE = re.compile(r"Answer:\s*(.+)", re.IGNORECASE)


def _extract_answer(text: str) -> str:
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else text.strip().splitlines()[-1]


def run_io(llm: LLMClient, puzzle: str, temperature: float = 0.7, max_tokens: int = 128) -> dict:
    prompt = standard_prompt.format(input=puzzle)
    response = llm.complete(prompt, n=1, temperature=temperature, max_tokens=max_tokens)[0]
    return {"output": _extract_answer(response), "api_calls": 1}
