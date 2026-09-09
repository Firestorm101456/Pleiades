from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FactSource(str, Enum):
    USER_PROVIDED = "USER_PROVIDED"
    OBSERVED = "OBSERVED"
    RETRIEVED = "RETRIEVED"
    INFERRED = "INFERRED"
    ASSUMED = "ASSUMED"
    ESTIMATED = "ESTIMATED"
    UNKNOWN = "UNKNOWN"


class SemanticRole(str, Enum):
    ACTION = "ACTION"
    ACTOR = "ACTOR"
    ENTITY = "ENTITY"
    OBJECT = "OBJECT"
    ATTRIBUTE = "ATTRIBUTE"
    QUANTITY = "QUANTITY"
    TIME = "TIME"
    LOCATION = "LOCATION"
    CONSTRAINT = "CONSTRAINT"
    RESOURCE = "RESOURCE"
    PREFERENCE = "PREFERENCE"
    DECISION_CRITERION = "DECISION_CRITERION"
    SUCCESS_CONDITION = "SUCCESS_CONDITION"
    DEPENDENCY = "DEPENDENCY"
    REQUESTED_OUTPUT = "REQUESTED_OUTPUT"
    UNKNOWN_VARIABLE = "UNKNOWN_VARIABLE"
    ASSUMPTION = "ASSUMPTION"


@dataclass(frozen=True)
class StructuredFact:
    text: str
    source: FactSource
    confidence: float = 1.0
    derived_from: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SemanticElement:
    role: SemanticRole
    value: str
    source: FactSource = FactSource.INFERRED
    confidence: float = 0.7
    derived_from: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Ambiguity:
    text: str
    interpretations: list[str]
    impact: str
    clarification_question: str


@dataclass(frozen=True)
class ObjectiveModel:
    objective_id: str
    objective_text: str
    desired_outcome: str
    semantic_elements: list[SemanticElement] = field(default_factory=list)
    actors: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)
    location_constraints: list[str] = field(default_factory=list)
    financial_constraints: list[str] = field(default_factory=list)
    requested_outputs: list[str] = field(default_factory=list)
    unknown_variables: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    explicit_facts: list[StructuredFact] = field(default_factory=list)
    inferred_facts: list[StructuredFact] = field(default_factory=list)
    time_constraints: list[str] = field(default_factory=list)
    deadlines: list[str] = field(default_factory=list)
    quantities: list[str] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)
    preferences: list[str] = field(default_factory=list)
    success_conditions: list[str] = field(default_factory=list)
    implicit_dependencies: list[str] = field(default_factory=list)
    ambiguities: list[Ambiguity] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    conflicting_requirements: list[str] = field(default_factory=list)
