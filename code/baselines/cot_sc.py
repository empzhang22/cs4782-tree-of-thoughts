import re
from collections import Counter

from llm.base import LLMClient
from tasks.game24.prompts import cot_prompt

_ANSWER_RE = re.compile(r"Answer:\s*(.+)")


def _extract_answer(text: str) -> str:
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else text.strip().splitlines()[-1]


def run_cot_sc(
    llm: LLMClient,
    puzzle: str,
    n_samples: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> dict:
    """CoT self-consistency: sample n times, return majority-vote answer."""
    prompt = cot_prompt.format(input=puzzle)
    responses = llm.complete(prompt, n=n_samples, temperature=temperature, max_tokens=max_tokens)
    answers = [_extract_answer(r) for r in responses]
    majority = Counter(answers).most_common(1)[0][0]
    return {"output": majority, "all_outputs": answers, "api_calls": n_samples}
