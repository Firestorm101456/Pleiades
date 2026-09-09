from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from agent.memory.database import MemoryDatabase


@dataclass(frozen=True)
class EpisodicMemory:
    event: str
    context: str
    result: str
    importance: float
    confidence: float
    timestamp: str = ""


class EpisodicStore:
    def __init__(self, db: MemoryDatabase) -> None:
        self.db = db

    def add(self, memory: EpisodicMemory) -> None:
        timestamp = memory.timestamp or datetime.now(UTC).isoformat()
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO episodic_memory
                (timestamp, event, context, result, importance, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (timestamp, memory.event, memory.context, memory.result, memory.importance, memory.confidence),
            )

