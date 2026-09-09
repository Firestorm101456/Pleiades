from __future__ import annotations

from agent.analysis.information_value import InformationStatus, InformationValueAnalysis
from agent.core.state import State


class Planner:
    def propose_actions(self, state: State, information_value: InformationValueAnalysis | None = None) -> list[str]:
        if (
            information_value
            and information_value.status == InformationStatus.NEEDED
            and information_value.top_candidate
        ):
            return [
                f"Gather information: {information_value.top_candidate.information}",
                "Choose the best reversible next step using current evidence.",
                "Stop and request additional human input.",
            ]
        if information_value and information_value.status == InformationStatus.SUFFICIENT:
            return [
                "Choose the best reversible next step using current evidence.",
                "Stop and request additional human input.",
            ]
        return [
            "Gather one high-value missing fact before acting.",
            "Choose the best reversible next step using current evidence.",
            "Stop and request additional human input.",
        ]
