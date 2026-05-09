from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def complete(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> list[str]:
        """Return n completion strings for the given prompt."""
        ...
