from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

from agent.analysis.information_value import InformationStatus, InformationValueAnalyzer
from agent.capabilities.acquisition import CapabilityAcquisitionManager, MockAcquisitionProvider
from agent.capabilities.discovery import EnvironmentDiscovery
from agent.capabilities.inventory import CapabilityInventory
from agent.capabilities.models import CapabilityRequirement, InventoryState, RiskClassification, utc_now
from agent.config import Settings
from agent.core.facts import ObjectiveModel
from agent.core.state import State
from agent.core.task_manager import TaskManager
from agent.core.tasks import RequirementCategory, Task, TaskRequirement, TaskStatus
from agent.memory.database import MemoryDatabase
from agent.reasoning.decomposer import TaskDecomposer
from agent.reasoning.goal import GoalInterpreter
from agent.reasoning.requirements import RequirementGenerator
from agent.security.gate import AuthorizationRequest, MockAuthorizationProvider, SecurityGate
from agent.tools.registry import ToolRegistry, facts_from_tool_result_by_capabilities


class EngineStatus(str, Enum):
    COMPLETE = "COMPLETE"
    CONTINUE = "CONTINUE"
    REPLAN = "REPLAN"
    BLOCK = "BLOCK"
    REQUEST_CLARIFICATION = "REQUEST_CLARIFICATION"
    PROCEED_UNDER_EXPLICIT_ASSUMPTION = "PROCEED_UNDER_EXPLICIT_ASSUMPTION"
    FAIL_SAFELY = "FAIL_SAFELY"
    PARTIALLY_COMPLETED = "PARTIALLY_COMPLETED"


@dataclass
class EngineCycle:
    cycle: int
    task_id: str | None
    task_title: str | None
    action: str
    observation: str
    status: EngineStatus
    capability_resolution: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    verification: str = ""


@dataclass
class EngineResult:
    objective_model: ObjectiveModel
    state: State
    tasks: list[Task]
    cycles: list[EngineCycle]
    status: EngineStatus
    final_result: str
    clarification_requests: list[str] = field(default_factory=list)


class CoreEngine:
    def __init__(
        self,
        settings: Settings,
        *,
        authorization_provider: MockAuthorizationProvider | None = None,
        max_cycles: int = 8,
    ) -> None:
        self.settings = settings
        self.db = MemoryDatabase(settings.database_path)
        self.inventory = CapabilityInventory(self.db)
        self.discovery = EnvironmentDiscovery(self.inventory)
        self.discovery.verify_known_safe()
        self.security = SecurityGate(self.db, authorization_provider or MockAuthorizationProvider(approve=False))
        self.capability_acquisition = CapabilityAcquisitionManager(self.inventory, self.security, [MockAcquisitionProvider()])
        self.goal_interpreter = GoalInterpreter()
        self.decomposer = TaskDecomposer()
        self.requirements = RequirementGenerator()
        self.task_manager = TaskManager(self.db)
        self.information_value = InformationValueAnalyzer()
        self.tools = ToolRegistry()
        self.max_cycles = max_cycles

    def run(self, objective_text: str, observations: list[str] | None = None) -> EngineResult:
        objective = self.goal_interpreter.interpret(objective_text, observations)
        self._persist_objective(objective)
        state = State(
            current_objective=objective.objective_text,
            known_facts=[fact.text for fact in objective.explicit_facts],
            assumptions=objective.assumptions[:],
            constraints=objective.constraints[:],
            unknown_information=objective.unknown_variables[:],
            available_resources=[resource.name for resource in self.inventory.resources() if resource.state == InventoryState.AVAILABLE],
        )
        state.known_facts.extend([fact.text for fact in objective.inferred_facts])
        tasks = self.decomposer.decompose(objective)
        self.task_manager.register(tasks)
        cycles: list[EngineCycle] = []
        clarification_requests: list[str] = []

        for cycle_index in range(1, self.max_cycles + 1):
            ready = self.task_manager.ready_tasks()
            if not ready:
                status = EngineStatus.COMPLETE if self.task_manager.objective_complete(objective.objective_id) else EngineStatus.BLOCK
                cycles.append(EngineCycle(cycle_index, None, None, "select_task", "No ready task is available.", status))
                break

            task = ready[0]
            self.task_manager.start(task.task_id)
            generated = self.requirements.generate(task, state)
            task.requirements = generated

            if self._needs_clarification(task, objective):
                question = self._clarification_for(task, objective)
                clarification_requests.append(question)
                self.task_manager.block(task.task_id, question)
                cycles.append(
                    EngineCycle(
                        cycle_index,
                        task.task_id,
                        task.title,
                        "request_clarification",
                        question,
                        EngineStatus.REQUEST_CLARIFICATION,
                        ["Clarification requirement handled before capability acquisition."],
                        [req.description for req in generated],
                        "Task blocked because ambiguity has high impact.",
                    )
                )
                break

            capability_resolution, capabilities_ready = self._resolve_capabilities(task, generated, state)
            if not capabilities_ready:
                blocker = "Mandatory capability requirement is unavailable or not authorized."
                self.task_manager.block(task.task_id, blocker)
                cycles.append(
                    EngineCycle(
                        cycle_index,
                        task.task_id,
                        task.title,
                        "resolve_capabilities",
                        blocker,
                        EngineStatus.BLOCK,
                        capability_resolution,
                        [self._requirement_label(req) for req in generated],
                        "Task blocked before execution because capability resolution did not verify required access.",
                    )
                )
                break

            action, observation = self._execute_task(task, generated, state)
            verified, verification = self._verify_task(task, observation)
            self.task_manager.record_attempt(task.task_id, action, "ACTION_SUCCEEDED" if observation else "UNKNOWN", observation)
            if verified:
                self.task_manager.complete(task.task_id, observation)
                status = EngineStatus.CONTINUE
            else:
                recovery = self._recovery_task(task, objective, verification)
                self.task_manager.add_task(recovery)
                self.task_manager.block(task.task_id, verification)
                status = EngineStatus.REPLAN
            cycles.append(
                EngineCycle(
                    cycle_index,
                    task.task_id,
                    task.title,
                    action,
                    observation,
                    status,
                    capability_resolution,
                    [self._requirement_label(req) for req in generated],
                    verification,
                )
            )
            if self.task_manager.objective_complete(objective.objective_id):
                break

        final_tasks = list(self.task_manager.tasks.values())
        final_status = self._final_status(objective.objective_id, clarification_requests)
        final_result = self._dynamic_result(objective, final_tasks, cycles, final_status, clarification_requests)
        self._persist_engine_run(objective, final_status, cycles, final_result)
        return EngineResult(objective, state, final_tasks, cycles, final_status, final_result, clarification_requests)

    def _resolve_capabilities(self, task: Task, requirements: list[TaskRequirement], state: State) -> tuple[list[str], bool]:
        notes: list[str] = []
        ready = True
        capability_requirements = [
            CapabilityRequirement(req.capability_id, req.reason)
            for req in requirements
            if req.category == RequirementCategory.CAPABILITY and req.capability_id and req.mandatory
        ]
        available = self.inventory.available_capability_ids()
        for requirement in capability_requirements:
            if requirement.capability_id in available:
                notes.append(f"{requirement.capability_id}: AVAILABLE")
        for missing in self.inventory.find_missing(capability_requirements):
            notes.append(f"{missing.requirement.capability_id}: {missing.current_state.value}")
            if self.inventory.can_compose(missing.requirement.capability_id):
                notes.append(f"{missing.requirement.capability_id}: composed from available capabilities")
                continue
            acquisition = self.capability_acquisition.acquire_best(missing.requirement)
            if acquisition and acquisition.ok:
                notes.append(f"{missing.requirement.capability_id}: acquired and verified")
            elif acquisition:
                notes.append(f"{missing.requirement.capability_id}: acquisition blocked or failed")
                ready = False
            else:
                notes.append(f"{missing.requirement.capability_id}: no acquisition option")
                ready = False
        for capability_id in self.inventory.available_capability_ids():
            if capability_id not in state.available_resources:
                pass
        return notes or ["No task-derived capability requirement."], ready

    def _needs_clarification(self, task: Task, objective: ObjectiveModel) -> bool:
        return task.purpose == "clarification" and any(item.impact == "HIGH" for item in objective.ambiguities)

    def _clarification_for(self, task: Task, objective: ObjectiveModel) -> str:
        high = next(item for item in objective.ambiguities if item.impact == "HIGH")
        return high.clarification_question

    def _execute_task(self, task: Task, requirements: list[TaskRequirement], state: State) -> tuple[str, str]:
        if task.purpose == "understanding":
            return "interpret_objective", f"Objective model for '{state.current_objective}' records supplied facts, inferred facts, constraints, and ambiguities."
        if task.purpose == "research":
            return self._execute_research_task(task, state)
        if task.purpose == "analysis":
            return "analyze_evidence", f"Analysis step used {len(state.known_facts)} recorded fact(s); unresolved values remain labeled."
        if task.purpose == "planning":
            return "construct_plan", f"Planning step represented dependencies and blockers for {task.title}."
        if task.purpose == "execution":
            request = AuthorizationRequest(
                action=f"Execute task {task.task_id}",
                risk=task.risk,
                reason="Task may change local state.",
                scope=task.task_id,
            )
            decision = self.security.check(request)
            if decision.authorization_required and not decision.approved:
                return "authorize_action", f"Action not executed: {decision.reason}"
            return "execute_authorized_action", f"Authorized action for {task.title} executed; completion still depends on verification."
        if task.purpose == "result":
            return "generate_result", f"Result generated from {len(self.task_manager.tasks)} task record(s) and current blockers."
        return "advance_task", "Task advanced using currently available state."

    def _execute_research_task(self, task: Task, state: State) -> tuple[str, str]:
        requirement_unknowns = [
            req.description
            for req in task.requirements
            if req.category == RequirementCategory.INFORMATION and req.status != "RESOLVED"
        ]
        scoped_state = State(
            current_objective=state.current_objective,
            known_facts=state.known_facts[:],
            unknown_information=requirement_unknowns or state.unknown_information[:],
            constraints=state.constraints[:],
            acquisition_attempts=state.acquisition_attempts[:],
        )
        candidates = self.information_value.analyze(scoped_state, [task.description])
        if candidates.status == InformationStatus.SUFFICIENT or not candidates.top_candidate:
            return "evaluate_information_value", candidates.stop_reason
        plan = self.tools.plan_acquisition(candidates.top_candidate, state)
        if not plan.selected_strategy:
            return "plan_information_acquisition", plan.reason
        tool = self.tools.get_tool(plan.selected_strategy.tool_name)
        if not tool:
            return "select_sensor", "Planned sensor is unavailable."
        result = tool.execute({}, authorized=False)
        facts = facts_from_tool_result_by_capabilities(tool.name, set(plan.selected_strategy.required_capabilities), result.metadata)
        for fact in facts:
            state.update_from_observation(fact)
        return f"execute_sensor:{tool.name}", "; ".join(facts) if facts else result.output

    def _verify_task(self, task: Task, observation: str) -> tuple[bool, str]:
        if task.purpose == "execution" and observation.startswith("Action not executed"):
            return False, "Execution was blocked; task completion criteria were not satisfied."
        if observation.startswith("Mandatory capability") or "No untried safe" in observation or "unavailable" in observation.lower():
            return False, "Required evidence or capability was not acquired, so completion criteria are unsatisfied."
        if not observation:
            return False, "No observation was produced, so completion cannot be verified."
        if task.purpose == "research" and not any(":" in fact and "UNKNOWN" not in fact for fact in task.observations + [observation]):
            return False, "Research task did not acquire observed evidence."
        return True, "Task completion criteria satisfied by observation and explicit verification step."

    def _recovery_task(self, task: Task, objective: ObjectiveModel, reason: str) -> Task:
        return Task(
            task_id=f"{task.task_id}:recovery:{len(self.task_manager.tasks)}",
            objective_id=objective.objective_id,
            title=f"Recover from blocked task: {task.title}",
            description=f"Find an alternate path because: {reason}",
            purpose="replanning",
            priority=max(0.4, task.priority - 0.05),
            dependencies=[],
            expected_outcomes=["A safer alternate task or explicit blocker is recorded."],
            completion_criteria=["Recovery path is documented."],
            uncertainty=min(1.0, task.uncertainty + 0.1),
        )

    def _final_status(self, objective_id: str, clarification_requests: list[str]) -> EngineStatus:
        if clarification_requests:
            return EngineStatus.REQUEST_CLARIFICATION
        if self.task_manager.objective_complete(objective_id):
            return EngineStatus.COMPLETE
        if self.task_manager.blocked_tasks():
            return EngineStatus.PARTIALLY_COMPLETED
        return EngineStatus.CONTINUE

    def _dynamic_result(
        self,
        objective: ObjectiveModel,
        tasks: list[Task],
        cycles: list[EngineCycle],
        status: EngineStatus,
        clarification_requests: list[str],
    ) -> str:
        completed = [task for task in tasks if task.status == TaskStatus.COMPLETED]
        blocked = [task for task in tasks if task.status == TaskStatus.BLOCKED]
        lines = [f"Status: {status.value}", f"Objective: {objective.objective_text}"]
        if objective.desired_outcome in {"diagnosis_and_resolution", "research_synthesis"}:
            lines.append("Findings:")
            lines.extend(f"- {cycle.observation}" for cycle in cycles if cycle.observation and cycle.task_title)
            lines.append("Remaining issues:")
            lines.extend(f"- {task.title}: {', '.join(task.failure_information) or task.status.value}" for task in blocked)
        elif objective.desired_outcome == "plan":
            lines.append("Plan:")
            lines.extend(f"- {task.title}: {task.status.value}" for task in tasks)
            lines.append("Dependencies:")
            lines.extend(f"- {task.title} depends on {', '.join(task.dependencies) or 'no prior task'}" for task in tasks)
        elif objective.desired_outcome == "decision":
            lines.append("Decision Basis:")
            lines.extend(f"- {task.title}: {task.status.value}" for task in completed)
            lines.append("Uncertainty:")
            lines.extend(f"- {item}" for item in objective.unknown_variables or ["None identified at this stage."])
        else:
            lines.append("Work Performed:")
            lines.extend(f"- {task.title}: {task.status.value}" for task in tasks)
        if clarification_requests:
            lines.append("Clarification Needed:")
            lines.extend(f"- {item}" for item in clarification_requests)
        return "\n".join(lines)

    def _requirement_label(self, requirement: TaskRequirement) -> str:
        return (
            f"{requirement.category.value}: {requirement.description} "
            f"[status={requirement.status}, source={requirement.source}, importance={requirement.importance:.2f}]"
        )

    def _persist_objective(self, objective: ObjectiveModel) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO objective_models
                (objective_id, objective_text, desired_outcome, payload, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (objective.objective_id, objective.objective_text, objective.desired_outcome, self.db.dumps(asdict(objective)), utc_now(), utc_now()),
            )

    def _persist_engine_run(self, objective: ObjectiveModel, status: EngineStatus, cycles: list[EngineCycle], result: str) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO engine_runs
                (timestamp, objective_id, objective_text, status, cycles, result, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now(),
                    objective.objective_id,
                    objective.objective_text,
                    status.value,
                    len(cycles),
                    result,
                    self.db.dumps({"cycles": [asdict(cycle) for cycle in cycles]}),
                ),
            )
