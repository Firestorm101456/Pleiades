from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class ComponentStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    KNOWN = "KNOWN"
    MEASURED = "MEASURED"
    ESTIMATED = "ESTIMATED"
    CONTRADICTED = "CONTRADICTED"
    UNAVAILABLE = "UNAVAILABLE"


class RequirementStatus(str, Enum):
    RESOLVED = "RESOLVED"
    PARTIALLY_RESOLVED = "PARTIALLY_RESOLVED"
    UNRESOLVED = "UNRESOLVED"
    UNAVAILABLE = "UNAVAILABLE"


RESOLVED_COMPONENT_STATUSES = {
    ComponentStatus.KNOWN,
    ComponentStatus.MEASURED,
    ComponentStatus.ESTIMATED,
}


@dataclass
class InformationComponent:
    id: str
    name: str
    description: str
    required_information_types: list[str]
    status: ComponentStatus = ComponentStatus.UNKNOWN
    value: str | int | float | None = None
    value_type: str = "unknown"
    provenance: str = "state"
    confidence: float = 0.0
    required: bool = True
    importance: float = 0.5
    acquisition_options: list[str] = field(default_factory=list)

    @property
    def is_resolved(self) -> bool:
        return self.status in RESOLVED_COMPONENT_STATUSES and self.value not in (None, "")


@dataclass
class InformationRequirement:
    id: str
    name: str
    description: str
    components: list[InformationComponent]
    importance: float = 0.5
    parent_requirement_id: str | None = None
    source: str = "information_value"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def status(self) -> RequirementStatus:
        if not self.components:
            return RequirementStatus.UNRESOLVED
        required = [component for component in self.components if component.required]
        if required and all(component.status == ComponentStatus.UNAVAILABLE for component in required):
            return RequirementStatus.UNAVAILABLE
        resolved = [component for component in required if component.is_resolved]
        if required and len(resolved) == len(required):
            return RequirementStatus.RESOLVED
        if resolved:
            return RequirementStatus.PARTIALLY_RESOLVED
        return RequirementStatus.UNRESOLVED

    @property
    def unresolved_components(self) -> list[InformationComponent]:
        return [component for component in self.components if component.required and not component.is_resolved]

    @property
    def unresolved_information_types(self) -> list[str]:
        capabilities: list[str] = []
        for component in self.unresolved_components:
            capabilities.extend(component.required_information_types)
        return list(dict.fromkeys(capabilities))

