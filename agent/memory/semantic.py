from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from agent.memory.database import MemoryDatabase


@dataclass(frozen=True)
class SemanticMemory:
    information: str
    source: str
    confidence: float
    tags: list[str] = field(default_factory=list)
    relationships: dict[str, list[str]] = field(default_factory=dict)
    timestamp: str = ""


class SemanticStore:
    def __init__(self, db: MemoryDatabase) -> None:
        self.db = db

    def add(self, memory: SemanticMemory) -> None:
        timestamp = memory.timestamp or datetime.now(UTC).isoformat()
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO semantic_memory
                (information, source, timestamp, confidence, tags, relationships)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.information,
                    memory.source,
                    timestamp,
                    memory.confidence,
                    self.db.dumps(memory.tags),
                    self.db.dumps(memory.relationships),
                ),
            )

