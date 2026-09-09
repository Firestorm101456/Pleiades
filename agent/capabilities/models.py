from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class InventoryState(str, Enum):
    UNKNOWN = "UNKNOWN"
    DISCOVERED = "DISCOVERED"
    KNOWN_BUT_NOT_ACQUIRED = "KNOWN_BUT_NOT_ACQUIRED"
    AVAILABLE = "AVAILABLE"
    UNVERIFIED = "UNVERIFIED"
    BROKEN = "BROKEN"
    UNAVAILABLE = "UNAVAILABLE"


class CapabilityType(str, Enum):
    SENSOR = "SENSOR"
    ACTUATOR = "ACTUATOR"
    ANALYSIS = "ANALYSIS"
    RESOURCE = "RESOURCE"


class RiskClassification(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class Evidence:
    kind: str
    detail: str
    source: str
    timestamp: str = field(default_factory=utc_now)


@dataclass
class ResourceRecord:
    id: str
    name: str
    type: str
    state: InventoryState = InventoryState.UNKNOWN
    description: str = ""
    provider: str = ""
    version: str | None = None
    location: str | None = None
    discovery_source: str | None = None
    acquisition_method: str | None = None
    provenance: list[Evidence] = field(default_factory=list)
    acquisition_history: list[dict[str, Any]] = field(default_factory=list)
    verification_results: list[dict[str, Any]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    confidence: float = 0.0
    last_verified_at: str | None = None
    last_verified_status: str | None = None

    def add_evidence(self, kind: str, detail: str, source: str) -> None:
        self.provenance.append(Evidence(kind=kind, detail=detail, source=source))


@dataclass
class CapabilityRecord:
    id: str
    name: str
    description: str
    type: CapabilityType
    state: InventoryState = InventoryState.UNKNOWN
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)
    risk_level: RiskClassification = RiskClassification.LOW
    authorization_required: bool = False
    reversibility: str = "unknown"
    verification_method: str = ""
    provenance: list[Evidence] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)
    learned_uses: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    confidence: float = 0.0
    last_verified_at: str | None = None
    last_verified_status: str | None = None

    def add_evidence(self, kind: str, detail: str, source: str) -> None:
        self.provenance.append(Evidence(kind=kind, detail=detail, source=source))


@dataclass(frozen=True)
class CapabilityRequirement:
    capability_id: str
    reason: str
    acceptable_substitutes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MissingCapability:
    requirement: CapabilityRequirement
    current_state: InventoryState
    available_alternatives: list[str]
    acquisition_options: list[str]
    reason: str


@dataclass(frozen=True)
class CapabilityComposition:
    capability_id: str
    required_capabilities: list[str]
    description: str


@dataclass(frozen=True)
class AcquisitionOption:
    id: str
    acquires_resource_id: str
    provides_capability_ids: list[str]
    source: str
    method: str
    prerequisites: list[str]
    estimated_cost: float
    estimated_time: float
    reliability: float
    risk: RiskClassification
    reversibility: str
    authorization_required: bool
    verification_method: str
    rollback_method: str | None = None

    @property
    def score(self) -> float:
        return self.reliability - self.estimated_cost - self.estimated_time


@dataclass(frozen=True)
class AcquisitionExecutionResult:
    option_id: str
    ok: bool
    output: str
    resource: ResourceRecord | None = None
    capabilities: list[CapabilityRecord] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)

