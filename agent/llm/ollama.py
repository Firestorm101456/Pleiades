from __future__ import annotations

import json
import urllib.error
import urllib.request

from agent.llm.base import LLMResponse, LanguageModel


class OllamaLanguageModel(LanguageModel):
    def __init__(self, base_url: str, model: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system

        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        return LLMResponse(
            text=str(data.get("response", "")).strip(),
            model=self.model,
            backend="ollama",
        )

