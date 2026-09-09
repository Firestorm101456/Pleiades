from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CandidateAction:
    description: str
    expected_benefit: float
    expected_cost: float
    probability_of_success: float
    major_risks: list[str] = field(default_factory=list)
    reversibility: float = 0.5
    required_resources: list[str] = field(default_factory=list)
    opportunity_cost: float = 0.0
    uncertainty: float = 0.5

    @property
    def expected_utility(self) -> float:
        upside = self.expected_benefit * self.probability_of_success
        downside = self.expected_cost + self.opportunity_cost + (self.uncertainty * 0.5)
        reversibility_credit = self.reversibility * 0.2
        return upside - downside + reversibility_credit


class DecisionEngine:
    def rank(self, actions: list[CandidateAction]) -> list[CandidateAction]:
        return sorted(actions, key=lambda action: action.expected_utility, reverse=True)

