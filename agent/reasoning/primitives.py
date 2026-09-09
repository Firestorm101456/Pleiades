from __future__ import annotations

import re
from dataclasses import dataclass

from agent.core.facts import ObjectiveModel
from agent.core.state import State
from agent.core.tasks import Task


@dataclass(frozen=True)
class DerivedNeed:
    name: str
    reason: str
    importance: float
    source_terms: list[str]


class TaskReasoningPrimitives:
    def identify_unknowns(self, task: Task, state: State) -> list[DerivedNeed]:
        if task.purpose not in {"research", "analysis", "planning"}:
            return []
        needs = self.determine_information_requirements(task, state)
        known = " ".join(state.known_facts + task.evidence + task.observations).lower()
        return [need for need in needs if not self._covered(need.name, known)]

    def determine_information_requirements(self, task: Task, state: State) -> list[DerivedNeed]:
        needs: list[DerivedNeed] = []
        for variable in state.unknown_information:
            name = variable.strip()
            if not name:
                continue
            needs.append(
                DerivedNeed(
                    name=name,
                    reason=f"Required because task '{task.title}' cannot satisfy its purpose while {name} is unavailable.",
                    importance=self._importance(name, task),
                    source_terms=[name],
                )
            )
        return self._dedupe(needs)

    def determine_required_capabilities(self, task: Task, state: State) -> list[DerivedNeed]:
        if task.purpose in {"understanding", "clarification", "result"}:
            return [DerivedNeed(cap, f"Task wording implies capability: {cap}.", 0.75, [cap]) for cap in task.required_capabilities]
        text = self._context(task, state)
        needs = [DerivedNeed(cap, f"Task wording implies capability: {cap}.", 0.75, [cap]) for cap in task.required_capabilities]
        if task.purpose == "research" and any(term in text for term in ("research", "gather", "current", "regulation", "price", "weather", "incentive", "requirements")):
            needs.append(DerivedNeed("perform_web_research", "The task requires information likely outside local state.", 0.8, ["research"]))
        if task.purpose == "research" and any(term in text for term in ("mac", "computer", "system", "slow", "diagnos")):
            needs.append(DerivedNeed("inspect_system", "The task requires observing local system state.", 0.85, ["system"]))
        if task.purpose == "analysis" and any(term in text for term in ("calculate", "financial", "cost", "income", "budget", "compare", "viable")):
            needs.append(DerivedNeed("calculate", "The task requires numeric or comparative analysis.", 0.75, ["calculate"]))
        if task.purpose == "execution" and any(term in text for term in ("fix", "modify", "install", "execute", "create")):
            needs.append(DerivedNeed("perform_authorized_action", "The task may change state and needs controlled actuation.", 0.9, ["action"]))
        return self._dedupe(needs)

    def determine_required_resources(self, task: Task, state: State) -> list[DerivedNeed]:
        text = self._context(task, state)
        resources = []
        if any(term in text for term in ("budget", "income", "$", "cost")):
            resources.append(DerivedNeed("financial inputs", "Financial terms affect feasibility or ranking.", 0.75, ["financial"]))
        if any(term in text for term in ("document", "file", "dataset")):
            resources.append(DerivedNeed("user-provided data", "The task references data that must be supplied or located.", 0.7, ["data"]))
        if any(term in text for term in ("mac", "computer", "system")):
            resources.append(DerivedNeed("local system observations", "The task depends on this machine's current state.", 0.8, ["system"]))
        return resources

    def determine_verification_requirements(self, task: Task, state: State) -> list[DerivedNeed]:
        criteria = task.completion_criteria or task.expected_outcomes
        return [
            DerivedNeed(
                self._shorten(criterion),
                f"Completion must be checked against: {criterion}",
                0.8,
                self._meaningful_terms(criterion),
            )
            for criterion in criteria
        ]

    def _context(self, task: Task, state: State) -> str:
        return " ".join([state.current_objective, task.title, task.description, task.purpose, *state.constraints]).lower()

    def _meaningful_terms(self, text: str) -> list[str]:
        return list(dict.fromkeys(term for term in re.findall(r"\$?\b[a-z0-9][a-z0-9$,-]*\b", text.lower()) if len(term) > 2))

    def _importance(self, anchor: str, task: Task) -> float:
        if anchor in {"deadline", "budget", "income", "cost", "risk", "safety", "slow", "fix", "viable"}:
            return 0.85
        return max(0.45, min(0.9, task.priority))

    def _covered(self, name: str, known: str) -> bool:
        terms = [term for term in name.lower().split() if len(term) > 3]
        return bool(terms) and all(term in known for term in terms)

    def _shorten(self, text: str) -> str:
        return " ".join(self._meaningful_terms(text)[:6]) or text[:60]

    def _dedupe(self, needs: list[DerivedNeed]) -> list[DerivedNeed]:
        seen: set[str] = set()
        output: list[DerivedNeed] = []
        for need in needs:
            key = need.name.lower()
            if key in seen:
                continue
            seen.add(key)
            output.append(need)
        return output
