from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from agent.core.information import ComponentStatus, InformationComponent, InformationRequirement
from agent.core.state import State


class InformationStatus(str, Enum):
    NEEDED = "NEEDED"
    SUFFICIENT = "SUFFICIENT"


class ValueLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True)
class InformationCandidate:
    information: str
    required_information_types: list[str]
    requirement: InformationRequirement
    why_it_matters: str
    potential_decision_impact: str
    decision_impact: float
    relevance: float
    uncertainty_reduction: float
    acquisition_cost: float
    time_cost: float
    candidate_elimination: float
    ranking_change: float
    acting_without_it_risk: float
    uncertainty: ValueLevel
    recommended: bool = False

    @property
    def score(self) -> float:
        return (
            self.decision_impact
            + self.relevance
            + self.uncertainty_reduction
            + self.candidate_elimination
            + self.ranking_change
            + self.acting_without_it_risk
            - self.acquisition_cost
            - self.time_cost
        )

    @property
    def value(self) -> ValueLevel:
        if self.score >= 4.0:
            return ValueLevel.HIGH
        if self.score >= 2.4:
            return ValueLevel.MEDIUM
        return ValueLevel.LOW

    def explain_score(self) -> str:
        return (
            "estimated score = impact + relevance + uncertainty reduction + elimination "
            "+ ranking change + acting-without-it risk - acquisition cost - time cost"
        )


@dataclass(frozen=True)
class InformationValueAnalysis:
    candidates: list[InformationCandidate]
    status: InformationStatus
    stop_reason: str

    @property
    def top_candidate(self) -> InformationCandidate | None:
        return self.candidates[0] if self.candidates else None


class InformationValueAnalyzer:
    def analyze(self, state: State, candidate_actions: list[str]) -> InformationValueAnalysis:
        candidates = self._generate_candidates(state, candidate_actions)
        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
        if ranked:
            ranked[0] = self._with_recommendation(ranked[0])

        top = ranked[0] if ranked else None
        status = InformationStatus.NEEDED
        stop_reason = "The top missing item has enough estimated value to justify gathering it."
        known_coverage = self._known_coverage(state)

        if not top:
            status = InformationStatus.SUFFICIENT
            stop_reason = "No decision-relevant missing information was identified."
        elif top.value == ValueLevel.LOW:
            status = InformationStatus.SUFFICIENT
            stop_reason = "Remaining unknowns have low estimated decision impact."
        elif (known_coverage >= 0.7 or len(state.known_facts) >= 4) and top.score < 4.4:
            status = InformationStatus.SUFFICIENT
            stop_reason = (
                "Major decision constraints appear known, and remaining information is unlikely "
                "to change the ranking for the current objective."
            )

        return InformationValueAnalysis(ranked, status, stop_reason)

    def _generate_candidates(self, state: State, candidate_actions: list[str]) -> list[InformationCandidate]:
        objective = state.current_objective.lower()
        base_variables = self._decision_variables(objective)
        state_unknowns = state.unknown_information[:]
        unknowns = [item for item in base_variables + state_unknowns if not self._is_known(item, state)]
        seen: set[str] = set()
        candidates: list[InformationCandidate] = []

        for variable in unknowns:
            normalized = variable.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            candidate = self._score_candidate(variable, state, candidate_actions)
            if candidate.requirement.status.value != "RESOLVED":
                candidates.append(candidate)

        return candidates

    def _decision_variables(self, objective: str) -> list[str]:
        domain_rules = [
            (("local llm", "local model", "ollama"), ["system RAM", "GPU model and VRAM", "primary workload", "required context length", "acceptable inference speed", "available storage", "existing local models"]),
            (("database", "data store"), ["expected data size", "query pattern", "consistency requirements", "latency requirements", "operational constraints"]),
            (("cybersecurity", "security tool"), ["target environment", "required detection capability", "deployment constraints", "integration requirements", "false-positive tolerance"]),
            (("laptop", "notebook"), ["workload", "budget", "portability requirements", "battery life requirement", "operating system constraints"]),
        ]
        variables: list[str] = []
        for keywords, additions in domain_rules:
            if any(keyword in objective for keyword in keywords):
                variables.extend(additions)
        if variables:
            return variables
        if any(word in objective for word in ("choose", "decide", "whether", "recommend")):
            return ["decision_criteria"]
        if any(word in objective for word in ("research", "gather", "investigate")):
            return ["research_scope"]
        if any(word in objective for word in ("fix", "diagnose", "slow", "broken")):
            return ["observed_symptoms"]
        return []

    def _score_candidate(self, information: str, state: State, candidate_actions: list[str]) -> InformationCandidate:
        text = information.lower()
        objective = state.current_objective.lower()
        constraints = " ".join(state.constraints + state.known_facts).lower()

        relevance = 0.9 if any(word in objective for word in text.split()) else 0.55
        decision_impact = 0.75
        uncertainty_reduction = 0.7
        acquisition_cost = 0.18
        time_cost = 0.18
        candidate_elimination = 0.35
        ranking_change = 0.45
        acting_without_it_risk = 0.45

        high_impact_terms = ("constraint", "requirement", "workload", "budget", "ram", "vram", "size", "consistency", "environment", "criteria")
        if any(term in text for term in high_impact_terms):
            decision_impact += 0.45
            candidate_elimination += 0.35
            ranking_change += 0.25

        if text.startswith("what ") or text.startswith("which "):
            acquisition_cost += 0.35
            time_cost += 0.25
            candidate_elimination -= 0.2
            ranking_change -= 0.2

        if any(term in text for term in ("available", "existing", "current")):
            acquisition_cost -= 0.08
            time_cost -= 0.06

        if "must" in constraints or "constraint" in text:
            acting_without_it_risk += 0.25

        if len(candidate_actions) <= 1:
            ranking_change -= 0.25

        uncertainty = ValueLevel.MEDIUM
        if any(term in text for term in high_impact_terms):
            uncertainty = ValueLevel.LOW
        elif "unknown" in text or "which evidence" in text:
            uncertainty = ValueLevel.HIGH

        requirement = self._build_requirement(information, state)
        unresolved_types = requirement.unresolved_information_types
        return InformationCandidate(
            information=information,
            required_information_types=unresolved_types or self._required_information_types(text),
            requirement=requirement,
            why_it_matters=f"It clarifies a decision variable for: {state.current_objective}",
            potential_decision_impact="Could change option ranking or eliminate unsuitable candidates.",
            decision_impact=round(decision_impact, 2),
            relevance=round(relevance, 2),
            uncertainty_reduction=round(uncertainty_reduction, 2),
            acquisition_cost=round(max(0.0, acquisition_cost), 2),
            time_cost=round(max(0.0, time_cost), 2),
            candidate_elimination=round(candidate_elimination, 2),
            ranking_change=round(ranking_change, 2),
            acting_without_it_risk=round(acting_without_it_risk, 2),
            uncertainty=uncertainty,
        )

    def _known_coverage(self, state: State) -> float:
        if not state.known_facts:
            return 0.0
        known = " ".join(state.known_facts).lower()
        core = ("workload", "budget", "constraint", "requirement", "ram", "vram", "size", "environment")
        return sum(1 for item in core if item in known) / len(core)

    def _is_known(self, information: str, state: State) -> bool:
        requirement = self._build_requirement(information, state)
        if requirement.status.value == "RESOLVED":
            return True
        text = information.lower()
        known = " ".join(state.known_facts + state.current_conditions + state.constraints).lower()
        words = [word for word in text.split() if len(word) > 3]
        return bool(words) and all(word in known for word in words)

    def _with_recommendation(self, candidate: InformationCandidate) -> InformationCandidate:
        return InformationCandidate(**{**candidate.__dict__, "recommended": True})

    def _build_requirement(self, information: str, state: State) -> InformationRequirement:
        components = [
            self._component_for(name, state)
            for name in self._component_names(information)
        ]
        return InformationRequirement(
            id=self._normalize_id(information),
            name=information,
            description=f"Information required to evaluate: {state.current_objective}",
            components=components,
            importance=0.7,
        )

    def _component_names(self, information: str) -> list[str]:
        text = information.lower()
        if "gpu" in text and ("vram" in text or "gpu memory" in text):
            return ["GPU model", "GPU VRAM"]
        if "cpu" in text and "core" in text:
            return ["CPU model", "CPU core count"]
        return [information]

    def _component_for(self, name: str, state: State) -> InformationComponent:
        value, status, provenance = self._known_component_value(name, state)
        return InformationComponent(
            id=self._normalize_id(name),
            name=name,
            description=f"Atomic information component: {name}",
            required_information_types=self._required_information_types(name.lower()),
            status=status,
            value=value,
            value_type=type(value).__name__ if value is not None else "unknown",
            provenance=provenance,
            confidence=0.95 if status == ComponentStatus.MEASURED else 0.75 if status == ComponentStatus.KNOWN else 0.0,
            importance=0.7,
        )

    def _known_component_value(self, name: str, state: State) -> tuple[str | None, ComponentStatus, str]:
        target = self._normalize_id(name)
        for fact in state.known_facts:
            label, value, status, source = self._parse_fact(fact)
            if label and self._component_alias_match(target, self._normalize_id(label)):
                return value, status, source
        return None, ComponentStatus.UNKNOWN, "state"

    def _parse_fact(self, fact: str) -> tuple[str | None, str | None, ComponentStatus, str]:
        if ":" not in fact:
            return None, None, ComponentStatus.KNOWN, "user"
        label, rest = fact.split(":", 1)
        value = rest.strip()
        source = "user"
        status = ComponentStatus.KNOWN
        if "(MEASURED" in value:
            status = ComponentStatus.MEASURED
        elif "(ESTIMATED" in value:
            status = ComponentStatus.ESTIMATED
        if "source=" in value:
            source = value.split("source=", 1)[1].split(")", 1)[0]
        value = value.split("(", 1)[0].strip()
        if value == "UNKNOWN":
            status = ComponentStatus.UNKNOWN
            value = None
        return label.strip(), value, status, source

    def _component_alias_match(self, wanted: str, observed: str) -> bool:
        aliases = {
            "gpu_vram": {"gpu_vram", "gpu_memory_vram", "gpu_memory"},
            "gpu_model": {"gpu_model"},
            "system_ram": {"system_ram", "ram"},
            "available_storage": {"available_storage", "storage_available"},
            "cpu_model": {"cpu_model"},
            "cpu_core_count": {"cpu_core_count", "cpu_cores"},
        }
        return observed in aliases.get(wanted, {wanted})

    def _normalize_id(self, text: str) -> str:
        return "_".join(part for part in text.lower().replace("/", " ").replace("-", " ").split() if part)

    def _required_information_types(self, text: str) -> list[str]:
        matched: list[str] = []
        if "vram" in text or "gpu memory" in text:
            matched.append("gpu_vram")
        elif "gpu" in text or "graphics" in text:
            matched.append("gpu")
        if "system ram" in text or text == "ram" or text.endswith(" ram"):
            matched.append("ram")
        if "cpu" in text or "core count" in text or "processor" in text:
            matched.append("cpu")
        if "storage" in text or "disk" in text:
            matched.append("storage")
        if "operating system" in text or text.startswith("os "):
            matched.append("operating_system")
        if "hardware" in text or "computer" in text or "machine" in text:
            matched.append("hardware")
        return matched or ["human_input"]
