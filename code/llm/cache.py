import hashlib
import json
import diskcache


class ResponseCache:
    def __init__(self, cache_dir: str):
        self._cache = diskcache.Cache(cache_dir)

    def _key(self, model: str, prompt: str, n: int, temperature: float, max_tokens: int) -> str:
        payload = json.dumps(
            {"model": model, "prompt": prompt, "n": n, "temperature": temperature, "max_tokens": max_tokens},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, model: str, prompt: str, n: int, temperature: float, max_tokens: int):
        return self._cache.get(self._key(model, prompt, n, temperature, max_tokens))

    def set(self, model: str, prompt: str, n: int, temperature: float, max_tokens: int, value):
        self._cache.set(self._key(model, prompt, n, temperature, max_tokens), value)

    def close(self):
        self._cache.close()
