from __future__ import annotations

from dataclasses import dataclass

from agent.analysis.hypothesis import Hypothesis
from agent.core.decision import CandidateAction
from agent.core.state import State


@dataclass
class Critique:
    weak_assumptions: list[str]
    contradictory_evidence: list[str]
    missing_information: list[str]
    alternative_explanations: list[str]
    worst_plausible_outcome: str
    safer_action: str | None
    should_change_recommendation: bool


class Critic:
    def review(
        self,
        state: State,
        hypotheses: list[Hypothesis],
        selected_action: CandidateAction,
    ) -> Critique:
        weak_assumptions = [
            assumption
            for hypothesis in hypotheses
            for assumption in hypothesis.assumptions
            if hypothesis.confidence < 0.65
        ]
        contradictory = [
            evidence
            for hypothesis in hypotheses
            for evidence in hypothesis.contradictory_evidence
        ]
        missing = state.unknown_information[:]
        should_change = selected_action.uncertainty > 0.65 or selected_action.probability_of_success < 0.45
        safer = "Gather one high-value missing fact before acting." if should_change else None
        return Critique(
            weak_assumptions=weak_assumptions,
            contradictory_evidence=contradictory,
            missing_information=missing,
            alternative_explanations=[h.description for h in hypotheses[1:]],
            worst_plausible_outcome="A confident action optimizes for the wrong objective or stale evidence.",
            safer_action=safer,
            should_change_recommendation=should_change,
        )

