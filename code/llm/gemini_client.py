import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMClient
from .cache import ResponseCache

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_PAID_TIER_RPM = 1000


class GeminiClient(LLMClient):
    def __init__(self, model: str, cache: ResponseCache | None = None, rpm: int = _PAID_TIER_RPM):
        self.model = model
        self.cache = cache
        self.call_count: int = 0
        self._client = genai.Client()  # reads GOOGLE_API_KEY from env
        self._min_interval = 60.0 / rpm
        self._last_call_time = 0.0

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

        results = [self._one_call(prompt, temperature, max_tokens) for _ in range(n)]
        self.call_count += 1

        if self.cache is not None:
            self.cache.set(self.model, prompt, n, temperature, max_tokens, results)

        return results

    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=2, max=60))
    def _one_call(self, prompt: str, temperature: float, max_tokens: int) -> str:
        elapsed = time.time() - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_time = time.time()

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text or ""
