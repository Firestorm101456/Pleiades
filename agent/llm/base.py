from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    backend: str


class LanguageModel(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        """Generate text from a prompt."""


class StubLanguageModel(LanguageModel):
    def generate(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        lines = [line.strip() for line in prompt.splitlines() if line.strip()]
        focus = lines[-1] if lines else "the objective"
        return LLMResponse(
            text=(
                "Local deterministic reasoning stub. "
                f"Focus on: {focus[:220]}. "
                "Prefer gathering missing evidence before consequential action."
            ),
            model="stub",
            backend="local",
        )

