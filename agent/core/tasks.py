from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agent.capabilities.models import RiskClassification, utc_now


class TaskStatus(str, Enum):
    CREATED = "CREATED"
    READY = "READY"
    IN_PROGRESS = "IN_PROGRESS"
    BLOCKED = "BLOCKED"
    WAITING = "WAITING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class RequirementCategory(str, Enum):
    INFORMATION = "INFORMATION"
    CAPABILITY = "CAPABILITY"
    RESOURCE = "RESOURCE"
    ACTION = "ACTION"
    VERIFICATION = "VERIFICATION"
    AUTHORIZATION = "AUTHORIZATION"
    CLARIFICATION = "CLARIFICATION"


@dataclass
class TaskRequirement:
    id: str
    task_id: str
    category: RequirementCategory
    description: str
    reason: str
    importance: float = 0.5
    mandatory: bool = True
    status: str = "UNRESOLVED"
    source: str = "task_derived"
    confidence: float = 0.6
    dependencies: list[str] = field(default_factory=list)
    capability_id: str | None = None
    resource_id: str | None = None
    information: str | None = None
    evidence: list[str] = field(default_factory=list)


@dataclass
class TaskAttempt:
    timestamp: str
    action: str
    status: str
    observation: str


@dataclass
class Task:
    task_id: str
    objective_id: str
    title: str
    description: str
    purpose: str
    parent_task_id: str | None = None
    status: TaskStatus = TaskStatus.CREATED
    priority: float = 0.5
    dependencies: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    requirements: list[TaskRequirement] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    required_resources: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    expected_outcomes: list[str] = field(default_factory=list)
    verification_criteria: list[str] = field(default_factory=list)
    completion_criteria: list[str] = field(default_factory=list)
    deadline: str | None = None
    estimated_cost: float = 0.1
    estimated_duration: float = 0.1
    risk: RiskClassification = RiskClassification.LOW
    uncertainty: float = 0.5
    attempts: list[TaskAttempt] = field(default_factory=list)
    failure_information: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def add_observation(self, observation: str) -> None:
        if observation:
            self.observations.append(observation)
            self.evidence.append(observation)
            self.updated_at = utc_now()

    @property
    def terminal(self) -> bool:
        return self.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "objective_id": self.objective_id,
            "parent_task_id": self.parent_task_id,
            "title": self.title,
            "description": self.description,
            "purpose": self.purpose,
            "status": self.status.value,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "prerequisites": self.prerequisites,
            "requirements": [requirement.__dict__ | {"category": requirement.category.value} for requirement in self.requirements],
            "required_capabilities": self.required_capabilities,
            "required_resources": self.required_resources,
            "actions": self.actions,
            "expected_outcomes": self.expected_outcomes,
            "verification_criteria": self.verification_criteria,
            "completion_criteria": self.completion_criteria,
            "deadline": self.deadline,
            "estimated_cost": self.estimated_cost,
            "estimated_duration": self.estimated_duration,
            "risk": self.risk.value,
            "uncertainty": self.uncertainty,
            "attempts": [attempt.__dict__ for attempt in self.attempts],
            "failure_information": self.failure_information,
            "evidence": self.evidence,
            "observations": self.observations,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "Task":
        requirements = [
            TaskRequirement(**{**item, "category": RequirementCategory(item["category"])})
            for item in record.get("requirements", [])
        ]
        attempts = [TaskAttempt(**item) for item in record.get("attempts", [])]
        return cls(
            task_id=record["task_id"],
            objective_id=record["objective_id"],
            parent_task_id=record.get("parent_task_id"),
            title=record["title"],
            description=record["description"],
            purpose=record["purpose"],
            status=TaskStatus(record["status"]),
            priority=record["priority"],
            dependencies=record.get("dependencies", []),
            prerequisites=record.get("prerequisites", []),
            requirements=requirements,
            required_capabilities=record.get("required_capabilities", []),
            required_resources=record.get("required_resources", []),
            actions=record.get("actions", []),
            expected_outcomes=record.get("expected_outcomes", []),
            verification_criteria=record.get("verification_criteria", []),
            completion_criteria=record.get("completion_criteria", []),
            deadline=record.get("deadline"),
            estimated_cost=record.get("estimated_cost", 0.1),
            estimated_duration=record.get("estimated_duration", 0.1),
            risk=RiskClassification(record.get("risk", RiskClassification.LOW.value)),
            uncertainty=record.get("uncertainty", 0.5),
            attempts=attempts,
            failure_information=record.get("failure_information", []),
            evidence=record.get("evidence", []),
            observations=record.get("observations", []),
            created_at=record.get("created_at", utc_now()),
            updated_at=record.get("updated_at", utc_now()),
        )
