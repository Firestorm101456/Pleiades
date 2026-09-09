from __future__ import annotations

from dataclasses import dataclass, field

from agent.core.acquisition import AcquisitionAttempt
from agent.capabilities.models import CapabilityRequirement, MissingCapability
from agent.core.information import InformationRequirement


@dataclass
class State:
    current_objective: str
    known_facts: list[str] = field(default_factory=list)
    unknown_information: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    available_resources: list[str] = field(default_factory=list)
    current_conditions: list[str] = field(default_factory=list)
    active_risks: list[str] = field(default_factory=list)
    previous_actions: list[str] = field(default_factory=list)
    previous_outcomes: list[str] = field(default_factory=list)
    information_requirements: list[InformationRequirement] = field(default_factory=list)
    acquisition_attempts: list[AcquisitionAttempt] = field(default_factory=list)
    capability_requirements: list[CapabilityRequirement] = field(default_factory=list)
    missing_capabilities: list[MissingCapability] = field(default_factory=list)

    def update_from_observation(self, observation: str) -> None:
        if observation and observation not in self.known_facts:
            self.known_facts.append(observation)

    def mark_information_acquired(self, information: str) -> None:
        target = information.lower()
        self.unknown_information = [
            item
            for item in self.unknown_information
            if item.lower() != target and not all(word in " ".join(self.known_facts).lower() for word in target.split() if len(word) > 3)
        ]

    def upsert_information_requirement(self, requirement: InformationRequirement) -> None:
        for index, existing in enumerate(self.information_requirements):
            if existing.id == requirement.id:
                self.information_requirements[index] = requirement
                return
        self.information_requirements.append(requirement)

    def record_acquisition_attempt(self, attempt: AcquisitionAttempt) -> None:
        self.acquisition_attempts.append(attempt)

    def has_attempted_strategy(self, component_id: str, strategy_id: str) -> bool:
        return any(
            attempt.component_id == component_id and attempt.strategy_id == strategy_id
            for attempt in self.acquisition_attempts
        )

    def identify_unknowns(self) -> None:
        return
