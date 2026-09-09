from __future__ import annotations

from agent.core.tasks import RequirementCategory, Task, TaskRequirement
from agent.core.state import State
from agent.reasoning.primitives import TaskReasoningPrimitives


class RequirementGenerator:
    def __init__(self, primitives: TaskReasoningPrimitives | None = None) -> None:
        self.primitives = primitives or TaskReasoningPrimitives()

    def generate(self, task: Task, state: State | None = None) -> list[TaskRequirement]:
        state = state or State(current_objective=task.description)
        requirements: list[TaskRequirement] = []

        for need in self.primitives.determine_required_capabilities(task, state):
            requirements.append(
                TaskRequirement(
                    id=f"{task.task_id}:capability:{need.name}",
                    task_id=task.task_id,
                    category=RequirementCategory.CAPABILITY,
                    description=need.name,
                    capability_id=need.name,
                    reason=need.reason,
                    importance=need.importance,
                    source="derived_from_task_capability_primitives",
                    confidence=0.7,
                )
            )

        for need in self.primitives.identify_unknowns(task, state):
            requirements.append(
                TaskRequirement(
                    id=f"{task.task_id}:information:{self._slug(need.name)}",
                    task_id=task.task_id,
                    category=RequirementCategory.INFORMATION,
                    description=need.name,
                    information=need.name,
                    reason=need.reason,
                    importance=need.importance,
                    source="derived_from_task_unknowns",
                    confidence=0.65,
                )
            )

        for need in self.primitives.determine_required_resources(task, state):
            requirements.append(
                TaskRequirement(
                    id=f"{task.task_id}:resource:{self._slug(need.name)}",
                    task_id=task.task_id,
                    category=RequirementCategory.RESOURCE,
                    description=need.name,
                    resource_id=need.name,
                    reason=need.reason,
                    importance=need.importance,
                    mandatory=False,
                    source="derived_from_task_resource_primitives",
                    confidence=0.6,
                )
            )

        if task.purpose in {"analysis", "planning", "execution", "replanning"}:
            requirements.append(
                TaskRequirement(
                    id=f"{task.task_id}:action:{task.purpose}",
                    task_id=task.task_id,
                    category=RequirementCategory.ACTION,
                    description=f"perform {task.purpose} step for {task.title}",
                    reason=f"Task purpose is {task.purpose}, so progress requires an action matching that purpose.",
                    importance=task.priority,
                    source="derived_from_task_purpose",
                    confidence=0.75,
                )
            )

        if task.risk.value != "LOW":
            requirements.append(
                TaskRequirement(
                    id=f"{task.task_id}:authorization:{task.risk.value.lower()}",
                    task_id=task.task_id,
                    category=RequirementCategory.AUTHORIZATION,
                    description=f"authorization for {task.risk.value.lower()} risk task",
                    reason=f"Task risk is {task.risk.value}; security policy requires scoped authorization.",
                    importance=0.9,
                    source="derived_from_task_risk",
                    confidence=0.9,
                )
            )

        for need in self.primitives.determine_verification_requirements(task, state):
            requirements.append(
                TaskRequirement(
                    id=f"{task.task_id}:verification:{self._slug(need.name)}",
                    task_id=task.task_id,
                    category=RequirementCategory.VERIFICATION,
                    description=need.name,
                    reason=need.reason,
                    importance=need.importance,
                    source="derived_from_completion_criteria",
                    confidence=0.75,
                )
            )

        return requirements

    def _slug(self, text: str) -> str:
        return "_".join(part for part in text.lower().replace("/", " ").replace("-", " ").split() if part)[:80] or "requirement"
