from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMClient
from .cache import ResponseCache

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class OpenAIClient(LLMClient):
    def __init__(self, model: str, cache: ResponseCache | None = None):
        self.model = model
        self.cache = cache
        self.call_count: int = 0
        self._client = OpenAI()  # reads OPENAI_API_KEY from env

    def complete(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> list[str]:
        if self.cache is not None:
            cached = self.cache.get(self.model, prompt, n, temperature, max_tokens)
            if cached is not None:
                return cached

        results = self._single_complete(prompt, n, temperature, max_tokens)
        self.call_count += 1

        if self.cache is not None:
            self.cache.set(self.model, prompt, n, temperature, max_tokens, results)

        return results

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _single_complete(
        self, prompt: str, n: int, temperature: float, max_tokens: int
    ) -> list[str]:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
        )
        return [choice.message.content or "" for choice in response.choices]
