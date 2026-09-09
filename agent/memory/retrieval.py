from __future__ import annotations

from agent.memory.database import MemoryDatabase


class Retriever:
    def __init__(self, db: MemoryDatabase) -> None:
        self.db = db

    def relevant_knowledge(self, query: str, limit: int = 5) -> list[str]:
        terms = [term for term in query.lower().split() if len(term) > 3]
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT information, source FROM semantic_memory ORDER BY id DESC LIMIT 100"
            ).fetchall()
        scored: list[tuple[int, str]] = []
        for row in rows:
            text = f"{row['information']} source={row['source']}"
            score = sum(1 for term in terms if term in text.lower())
            if score:
                scored.append((score, text))
        return [text for _, text in sorted(scored, reverse=True)[:limit]]

