from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from agent.core.acquisition import AcquisitionAttempt, AcquisitionPlan, AcquisitionResultStatus, AcquisitionStrategy
from agent.capabilities.discovery import EnvironmentDiscovery
from agent.capabilities.inventory import CapabilityInventory
from agent.analysis.critic import Critic, Critique
from agent.analysis.hypothesis import Hypothesis, HypothesisGenerator
from agent.analysis.information_value import InformationCandidate, InformationValueAnalysis, InformationValueAnalyzer, InformationStatus
from agent.analysis.prediction import Prediction, Predictor
from agent.analysis.risk import RiskAnalyzer
from agent.config import Settings
from agent.core.decision import CandidateAction, DecisionEngine
from agent.core.objectives import Objective
from agent.core.state import State
from agent.llm.base import LanguageModel, StubLanguageModel
from agent.llm.ollama import OllamaLanguageModel
from agent.memory.database import MemoryDatabase
from agent.memory.episodic import EpisodicMemory, EpisodicStore
from agent.memory.retrieval import Retriever
from agent.planning.planner import Planner
from agent.tools.registry import ToolRegistry, facts_from_tool_result_by_capabilities


@dataclass
class ToolObservation:
    information: str
    tool_name: str
    strategy_name: str
    acquisition_status: str
    acquisition_reason: str
    risk_level: str
    operation_type: str
    matched_capabilities: list[str]
    target_components: list[str]
    facts: list[str]
    raw_metadata: dict[str, Any] | None


@dataclass
class AnalysisResult:
    state: State
    evidence: list[str]
    information_value: InformationValueAnalysis
    tool_observations: list[ToolObservation]
    hypotheses: list[Hypothesis]
    predictions: list[Prediction]
    considered_actions: list[CandidateAction]
    selected_action: CandidateAction
    critique: Critique
    final_result: str


class AnalyticalAgent:
    def __init__(self, settings: Settings, llm: LanguageModel | None = None) -> None:
        self.settings = settings
        self.llm = llm or self._make_llm(settings)
        self.db = MemoryDatabase(settings.database_path)
        self.inventory = CapabilityInventory(self.db)
        self.discovery = EnvironmentDiscovery(self.inventory)
        self.discovery.verify_known_safe()
        self.retriever = Retriever(self.db)
        self.episodic = EpisodicStore(self.db)
        self.hypotheses = HypothesisGenerator(self.llm)
        self.information_value = InformationValueAnalyzer()
        self.planner = Planner()
        self.predictor = Predictor()
        self.risk = RiskAnalyzer()
        self.decisions = DecisionEngine()
        self.critic = Critic()
        self.tools = ToolRegistry()

    def analyze(self, objective_text: str, observations: list[str] | None = None) -> AnalysisResult:
        objective = Objective.from_text(objective_text)
        state = State(
            current_objective=objective.description,
            constraints=objective.constraints[:],
            available_resources=["local SQLite memory", "local model backend", "read-only inspection tools"]
            + [resource.name for resource in self.inventory.resources() if resource.state.value == "AVAILABLE"],
        )
        for observation in observations or []:
            state.update_from_observation(observation)
        state.identify_unknowns()

        evidence = self.retriever.relevant_knowledge(objective.description)
        evidence.extend(state.known_facts)
        initial_action_texts = self.planner.propose_actions(state)
        information_value = self.information_value.analyze(state, initial_action_texts)
        self._record_information_requirements(state, information_value)
        tool_observations: list[ToolObservation] = []
        acquisition_blocked_reason: str | None = None

        for _ in range(self.settings.max_auto_read_acquisitions):
            if not information_value.top_candidate or information_value.status != InformationStatus.NEEDED:
                break
            top = information_value.top_candidate
            if top.information not in state.unknown_information:
                state.unknown_information.insert(0, top.information)
            evidence.append(f"Highest estimated information value: {top.information}")
            acquisition_plan = self.tools.plan_acquisition(top, state)
            observation = self._try_acquire_information(top, state, acquisition_plan)
            if not observation:
                acquisition_blocked_reason = acquisition_plan.reason
                break
            tool_observations.append(observation)
            evidence.extend(observation.facts)
            if observation.acquisition_status in {"RESOLVED", "PARTIALLY_RESOLVED"}:
                state.mark_information_acquired(top.information)
            information_value = self.information_value.analyze(state, initial_action_texts)
            self._record_information_requirements(state, information_value)

        hypotheses = self.hypotheses.generate(state, evidence)
        action_texts = self.planner.propose_actions(state, information_value)
        if acquisition_blocked_reason and action_texts and action_texts[0].startswith("Gather information:"):
            action_texts = [
                f"Proceed under uncertainty: {acquisition_blocked_reason}",
                "Stop and request additional human input.",
            ]
        predictions = self.predictor.predict(action_texts, hypotheses)
        actions = self._candidate_actions(predictions)
        ranked = self.decisions.rank(actions)
        selected = ranked[0]
        critique = self.critic.review(state, hypotheses, selected)

        if critique.should_change_recommendation and critique.safer_action:
            selected = next(
                (action for action in actions if action.description == critique.safer_action),
                CandidateAction(
                    description=critique.safer_action,
                    expected_benefit=0.55,
                    expected_cost=0.15,
                    probability_of_success=0.7,
                    major_risks=["Delay or extra effort."],
                    reversibility=0.9,
                    opportunity_cost=0.1,
                    uncertainty=0.3,
                ),
            )

        final_result = self._final_result(selected)
        self._store_analysis(state, evidence, information_value, tool_observations, hypotheses, predictions, actions, selected, final_result)
        self.episodic.add(
            EpisodicMemory(
                event="analysis_completed",
                context=objective.description,
                result=final_result,
                importance=0.7,
                confidence=selected.probability_of_success,
            )
        )

        return AnalysisResult(state, evidence, information_value, tool_observations, hypotheses, predictions, actions, selected, critique, final_result)

    def _record_information_requirements(self, state: State, information_value: InformationValueAnalysis) -> None:
        for candidate in information_value.candidates:
            state.upsert_information_requirement(candidate.requirement)

    def _try_acquire_information(
        self,
        candidate: InformationCandidate,
        state: State,
        acquisition_plan: AcquisitionPlan,
    ) -> ToolObservation | None:
        strategy = acquisition_plan.selected_strategy
        if not strategy:
            return None
        tool = self.tools.get_tool(strategy.tool_name)
        if not tool:
            return None
        matched_capabilities = set(strategy.required_capabilities)
        result = tool.execute({}, authorized=False)
        if not result.ok:
            self._record_strategy_attempts(
                candidate,
                state,
                strategy,
                AcquisitionResultStatus.FAILED,
                result.output,
                "The tool returned a failure result.",
            )
            return ToolObservation(
                information=candidate.information,
                tool_name=tool.name,
                strategy_name=strategy.name,
                acquisition_status=AcquisitionResultStatus.FAILED.value,
                acquisition_reason="The tool returned a failure result.",
                risk_level=tool.risk_level.value,
                operation_type=tool.operation_type.value,
                matched_capabilities=sorted(matched_capabilities),
                target_components=strategy.target_component_ids,
                facts=[f"Tool {tool.name} failed: {result.output}"],
                raw_metadata=result.metadata,
            )
        facts = facts_from_tool_result_by_capabilities(tool.name, matched_capabilities, result.metadata)
        for fact in facts:
            if "UNKNOWN" not in fact:
                state.update_from_observation(fact)
        status = self._status_for_strategy(candidate, strategy, facts)
        self._record_strategy_attempts(
            candidate,
            state,
            strategy,
            status,
            "; ".join(facts) if facts else result.output,
            self._reason_for_strategy_status(status),
        )
        state.previous_actions.append(f"Executed read-only tool: {tool.name}")
        state.previous_outcomes.extend(facts)
        return ToolObservation(
            information=candidate.information,
            tool_name=tool.name,
            strategy_name=strategy.name,
            acquisition_status=status.value,
            acquisition_reason=self._reason_for_strategy_status(status),
            risk_level=tool.risk_level.value,
            operation_type=tool.operation_type.value,
            matched_capabilities=sorted(matched_capabilities),
            target_components=strategy.target_component_ids,
            facts=facts,
            raw_metadata=result.metadata,
        )

    def _status_for_strategy(
        self,
        candidate: InformationCandidate,
        strategy: AcquisitionStrategy,
        facts: list[str],
    ) -> AcquisitionResultStatus:
        if not facts:
            return AcquisitionResultStatus.UNKNOWN
        measured_component_ids = {
            component.id
            for component in candidate.requirement.unresolved_components
            if component.id in strategy.target_component_ids
            if self._component_was_measured(component.name, facts)
        }
        if len(measured_component_ids) == len(strategy.target_component_ids):
            return AcquisitionResultStatus.RESOLVED
        if measured_component_ids:
            return AcquisitionResultStatus.PARTIALLY_RESOLVED
        return AcquisitionResultStatus.UNKNOWN

    def _component_was_measured(self, component_name: str, facts: list[str]) -> bool:
        normalized = component_name.lower().replace("/", " ")
        aliases = {
            "gpu vram": ("GPU memory/VRAM:",),
            "gpu model": ("GPU model:",),
            "system ram": ("System RAM:",),
            "available storage": ("Available storage:",),
            "cpu model": ("CPU model:",),
            "cpu core count": ("CPU cores:",),
        }
        labels = aliases.get(normalized, (f"{component_name}:",))
        return any(any(fact.startswith(label) for label in labels) and "UNKNOWN" not in fact for fact in facts)

    def _record_strategy_attempts(
        self,
        candidate: InformationCandidate,
        state: State,
        strategy: AcquisitionStrategy,
        status: AcquisitionResultStatus,
        result: str,
        reason: str,
    ) -> None:
        for component in candidate.requirement.unresolved_components:
            if component.id not in strategy.target_component_ids:
                continue
            component_status = status
            if status == AcquisitionResultStatus.PARTIALLY_RESOLVED:
                component_status = (
                    AcquisitionResultStatus.RESOLVED
                    if self._component_was_measured(component.name, result.split("; "))
                    else AcquisitionResultStatus.UNKNOWN
                )
            state.record_acquisition_attempt(
                AcquisitionAttempt.record(
                    component_id=component.id,
                    component_name=component.name,
                    strategy=strategy,
                    status=component_status,
                    result=result,
                    reason=reason,
                )
            )

    def _reason_for_strategy_status(self, status: AcquisitionResultStatus) -> str:
        reasons = {
            AcquisitionResultStatus.RESOLVED: "The strategy produced measured information for the target component.",
            AcquisitionResultStatus.PARTIALLY_RESOLVED: "The strategy resolved some, but not all, target components.",
            AcquisitionResultStatus.UNKNOWN: "The observation method did not expose the target component.",
            AcquisitionResultStatus.FAILED: "The strategy failed before producing usable observations.",
            AcquisitionResultStatus.UNAVAILABLE: "No useful acquisition mechanism is currently available.",
        }
        return reasons[status]

    def _candidate_actions(self, predictions: list[Prediction]) -> list[CandidateAction]:
        actions: list[CandidateAction] = []
        for prediction in predictions:
            risks = self.risk.assess(prediction.action)
            benefit = 0.75 if "Gather" in prediction.action else 0.62
            cost = 0.2 if "Gather" in prediction.action else 0.35
            reversibility = 0.9 if "Gather" in prediction.action else 0.7
            uncertainty = prediction.uncertainty
            if "Choose the best reversible next step" in prediction.action:
                benefit = 0.78
                cost = 0.18
                uncertainty = max(0.15, prediction.uncertainty - 0.12)
            if "Proceed under uncertainty" in prediction.action:
                benefit = 0.8
                cost = 0.18
                reversibility = 0.75
                uncertainty = max(0.32, prediction.uncertainty - 0.08)
            if "Stop" in prediction.action:
                benefit, cost = 0.45, 0.1
                reversibility = 0.9
            actions.append(
                CandidateAction(
                    description=prediction.action,
                    expected_benefit=benefit,
                    expected_cost=cost,
                    probability_of_success=prediction.probability,
                    major_risks=risks,
                    reversibility=reversibility,
                    required_resources=["time", "attention"],
                    opportunity_cost=0.08,
                    uncertainty=uncertainty,
                )
            )
        return actions

    def _final_result(self, selected: CandidateAction) -> str:
        if selected.uncertainty >= self.settings.uncertainty_stop_threshold:
            return "Uncertainty is too high; request additional information before action."
        return f"Recommended action: {selected.description}"

    def _store_analysis(
        self,
        state: State,
        evidence: list[str],
        information_value: InformationValueAnalysis,
        tool_observations: list[ToolObservation],
        hypotheses: list[Hypothesis],
        predictions: list[Prediction],
        actions: list[CandidateAction],
        selected: CandidateAction,
        final_result: str,
    ) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO analysis_history
                (timestamp, objective, initial_state, evidence_used, information_value, hypotheses, predictions,
                 considered_actions, selected_action, confidence, final_result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    state.current_objective,
                    self.db.dumps(asdict(state)),
                    self.db.dumps(evidence),
                    self.db.dumps(
                        {
                            **asdict(information_value),
                            "tool_observations": [asdict(item) for item in tool_observations],
                        }
                    ),
                    self.db.dumps([asdict(item) for item in hypotheses]),
                    self.db.dumps([asdict(item) for item in predictions]),
                    self.db.dumps([asdict(item) for item in actions]),
                    self.db.dumps(asdict(selected)),
                    selected.probability_of_success,
                    final_result,
                ),
            )

    @staticmethod
    def _make_llm(settings: Settings) -> LanguageModel:
        if settings.llm_backend == "ollama":
            return OllamaLanguageModel(settings.ollama_url, settings.ollama_model)
        return StubLanguageModel()
