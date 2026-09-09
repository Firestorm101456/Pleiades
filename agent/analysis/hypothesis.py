from __future__ import annotations

from dataclasses import dataclass, field

from agent.core.state import State
from agent.llm.base import LanguageModel


@dataclass
class Hypothesis:
    description: str
    supporting_evidence: list[str] = field(default_factory=list)
    contradictory_evidence: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    confidence: float = 0.5
    predicted_consequences: list[str] = field(default_factory=list)


class HypothesisGenerator:
    def __init__(self, llm: LanguageModel) -> None:
        self.llm = llm

    def generate(self, state: State, evidence: list[str]) -> list[Hypothesis]:
        prompt = (
            "Generate competing analytical hypotheses for this objective.\n"
            f"Objective: {state.current_objective}\nEvidence: {evidence}"
        )
        llm_summary = self.llm.generate(prompt).text
        return [
            Hypothesis(
                description="Proceed after targeted evidence gathering.",
                supporting_evidence=evidence[:3],
                assumptions=["The missing information can be acquired cheaply."],
                confidence=0.62,
                predicted_consequences=["Better decision quality with limited delay."],
            ),
            Hypothesis(
                description="Act now using current evidence.",
                supporting_evidence=state.known_facts[:3],
                contradictory_evidence=state.unknown_information[:3],
                assumptions=["Current evidence is representative enough."],
                confidence=0.42,
                predicted_consequences=["Faster progress but higher error risk."],
            ),
            Hypothesis(
                description=f"LLM alternate framing: {llm_summary}",
                supporting_evidence=evidence[:2],
                assumptions=["The language model framing is relevant but unverified."],
                confidence=0.5,
                predicted_consequences=["May expose overlooked considerations."],
            ),
        ]

