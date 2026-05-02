from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMClient
from .cache import ResponseCache

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class GeminiClient(LLMClient):
    def __init__(self, model: str, cache: ResponseCache | None = None):
        self.model = model
        self.cache = cache
        self.call_count: int = 0
        self._client = genai.Client()  # reads GOOGLE_API_KEY from env

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

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _one_call(self, prompt: str, temperature: float, max_tokens: int) -> str:
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text or ""
