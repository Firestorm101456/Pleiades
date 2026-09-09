from __future__ import annotations

from dataclasses import dataclass

from agent.analysis.hypothesis import Hypothesis


@dataclass
class Prediction:
    action: str
    expected_outcome: str
    probability: float
    uncertainty: float
    basis: str = "estimated, not measured"


class Predictor:
    def predict(self, actions: list[str], hypotheses: list[Hypothesis]) -> list[Prediction]:
        best_confidence = max((h.confidence for h in hypotheses), default=0.5)
        uncertainty = max(0.0, min(1.0, 1.0 - best_confidence))
        return [
            Prediction(
                action=action,
                expected_outcome=f"Advances objective by: {action}",
                probability=max(0.05, min(0.95, best_confidence - index * 0.08)),
                uncertainty=min(1.0, uncertainty + index * 0.05),
            )
            for index, action in enumerate(actions)
        ]

