from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_path: Path = Path(os.getenv("PLEIADES_DB", "pleiades.sqlite3"))
    llm_backend: str = os.getenv("PLEIADES_LLM_BACKEND", "stub")
    ollama_url: str = os.getenv("PLEIADES_OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("PLEIADES_OLLAMA_MODEL", "llama3.1")
    uncertainty_stop_threshold: float = float(os.getenv("PLEIADES_UNCERTAINTY_STOP", "0.72"))
    max_auto_read_acquisitions: int = int(os.getenv("PLEIADES_MAX_AUTO_READS", "4"))


settings = Settings()
