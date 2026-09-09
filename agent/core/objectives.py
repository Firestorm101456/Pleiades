from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Objective:
    description: str
    success_criteria: list[str]
    constraints: list[str]

    @classmethod
    def from_text(cls, text: str) -> "Objective":
        return cls(
            description=text.strip(),
            success_criteria=["recommend the highest expected utility next action"],
            constraints=["avoid consequential action without explicit authorization"],
        )

