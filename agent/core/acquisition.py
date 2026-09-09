from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class AcquisitionResultStatus(str, Enum):
    RESOLVED = "RESOLVED"
    PARTIALLY_RESOLVED = "PARTIALLY_RESOLVED"
    UNKNOWN = "UNKNOWN"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class AcquisitionStrategy:
    id: str
    name: str
    description: str
    target_component_ids: list[str]
    required_capabilities: list[str]
    tool_name: str
    estimated_information_gain: float
    estimated_success_probability: float
    acquisition_cost: float
    time_cost: float
    risk_cost: float
    risk_level: str

    @property
    def score(self) -> float:
        return (
            self.estimated_information_gain
            + self.estimated_success_probability
            - self.acquisition_cost
            - self.time_cost
            - self.risk_cost
        )

    def explain_score(self) -> str:
        return (
            "estimated score = information gain + success probability "
            "- acquisition cost - time cost - risk cost"
        )


@dataclass(frozen=True)
class AcquisitionAttempt:
    component_id: str
    component_name: str
    strategy_id: str
    strategy_name: str
    tool_name: str
    timestamp: str
    status: AcquisitionResultStatus
    result: str
    reason: str

    @classmethod
    def record(
        cls,
        *,
        component_id: str,
        component_name: str,
        strategy: AcquisitionStrategy,
        status: AcquisitionResultStatus,
        result: str,
        reason: str,
    ) -> "AcquisitionAttempt":
        return cls(
            component_id=component_id,
            component_name=component_name,
            strategy_id=strategy.id,
            strategy_name=strategy.name,
            tool_name=strategy.tool_name,
            timestamp=datetime.now(UTC).isoformat(),
            status=status,
            result=result,
            reason=reason,
        )


@dataclass(frozen=True)
class AcquisitionPlan:
    strategies: list[AcquisitionStrategy] = field(default_factory=list)
    selected_strategy: AcquisitionStrategy | None = None
    status: str = "NO_STRATEGY"
    reason: str = "No acquisition strategy was selected."

