from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from agent.capabilities.models import (
    CapabilityType,
    CapabilityComposition,
    CapabilityRecord,
    CapabilityRequirement,
    Evidence,
    InventoryState,
    MissingCapability,
    RiskClassification,
    ResourceRecord,
    utc_now,
)
from agent.memory.database import MemoryDatabase


class CapabilityInventory:
    def __init__(self, db: MemoryDatabase) -> None:
        self.db = db

    def upsert_resource(self, resource: ResourceRecord) -> None:
        existing_resource = self.get_resource(resource.id)
        if existing_resource and self._state_rank(existing_resource.state) > self._state_rank(resource.state):
            resource.state = existing_resource.state
            resource.version = resource.version or existing_resource.version
            resource.last_verified_at = resource.last_verified_at or existing_resource.last_verified_at
            resource.last_verified_status = resource.last_verified_status or existing_resource.last_verified_status
            resource.confidence = max(resource.confidence, existing_resource.confidence)
        payload = self._resource_payload(resource)
        with self.db.connect() as conn:
            existing = conn.execute("SELECT id FROM resources WHERE id = ?", (resource.id,)).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE resources
                    SET name=?, type=?, state=?, description=?, provider=?, version=?, location=?,
                        discovery_source=?, acquisition_method=?, provenance=?, acquisition_history=?,
                        verification_results=?, limitations=?, confidence=?, last_verified_at=?,
                        last_verified_status=?, updated_at=?
                    WHERE id=?
                    """,
                    (*payload[1:], utc_now(), resource.id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO resources
                    (id, name, type, state, description, provider, version, location, discovery_source,
                     acquisition_method, provenance, acquisition_history, verification_results, limitations,
                     confidence, last_verified_at, last_verified_status, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (*payload, utc_now()),
                )

    def upsert_capability(self, capability: CapabilityRecord) -> None:
        existing_capability = self.get_capability(capability.id)
        if existing_capability and self._state_rank(existing_capability.state) > self._state_rank(capability.state):
            capability.state = existing_capability.state
            capability.last_verified_at = capability.last_verified_at or existing_capability.last_verified_at
            capability.last_verified_status = capability.last_verified_status or existing_capability.last_verified_status
            capability.confidence = max(capability.confidence, existing_capability.confidence)
            capability.learned_uses = list(dict.fromkeys(existing_capability.learned_uses + capability.learned_uses))
            capability.resources = list(dict.fromkeys(existing_capability.resources + capability.resources))
        payload = self._capability_payload(capability)
        with self.db.connect() as conn:
            existing = conn.execute("SELECT id FROM capabilities WHERE id = ?", (capability.id,)).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE capabilities
                    SET name=?, description=?, type=?, state=?, inputs=?, outputs=?, prerequisites=?,
                        effects=?, risk_level=?, authorization_required=?, reversibility=?, verification_method=?,
                        provenance=?, learned_uses=?, limitations=?, confidence=?, last_verified_at=?,
                        last_verified_status=?, updated_at=?
                    WHERE id=?
                    """,
                    (*payload[1:-1], utc_now(), capability.id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO capabilities
                    (id, name, description, type, state, inputs, outputs, prerequisites, effects, risk_level,
                     authorization_required, reversibility, verification_method, provenance, learned_uses,
                     limitations, confidence, last_verified_at, last_verified_status, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (*payload[:-1], utc_now()),
                )
            for resource_id in capability.resources:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO resource_capabilities (resource_id, capability_id, provenance)
                    VALUES (?, ?, ?)
                    """,
                    (resource_id, capability.id, self.db.dumps({"source": "inventory_upsert", "timestamp": utc_now()})),
                )

    def add_composition(self, composition: CapabilityComposition) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO capability_compositions
                (capability_id, required_capabilities, description, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (composition.capability_id, self.db.dumps(composition.required_capabilities), composition.description, utc_now()),
            )

    def resources(self) -> list[ResourceRecord]:
        with self.db.connect() as conn:
            return [self._resource_from_row(row) for row in conn.execute("SELECT * FROM resources ORDER BY id").fetchall()]

    def capabilities(self) -> list[CapabilityRecord]:
        with self.db.connect() as conn:
            return [self._capability_from_row(row) for row in conn.execute("SELECT * FROM capabilities ORDER BY id").fetchall()]

    def get_resource(self, resource_id: str) -> ResourceRecord | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM resources WHERE id = ?", (resource_id,)).fetchone()
            return self._resource_from_row(row) if row else None

    def get_capability(self, capability_id: str) -> CapabilityRecord | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM capabilities WHERE id = ?", (capability_id,)).fetchone()
            return self._capability_from_row(row) if row else None

    def available_capability_ids(self) -> set[str]:
        return {cap.id for cap in self.capabilities() if cap.state == InventoryState.AVAILABLE}

    def find_missing(self, requirements: list[CapabilityRequirement]) -> list[MissingCapability]:
        missing: list[MissingCapability] = []
        available = self.available_capability_ids()
        for requirement in requirements:
            if requirement.capability_id in available:
                continue
            substitutes = [item for item in requirement.acceptable_substitutes if item in available]
            if substitutes:
                continue
            current = self.get_capability(requirement.capability_id)
            missing.append(
                MissingCapability(
                    requirement=requirement,
                    current_state=current.state if current else InventoryState.UNKNOWN,
                    available_alternatives=[],
                    acquisition_options=[],
                    reason="No verified local capability or acceptable substitute is available.",
                )
            )
        return missing

    def can_compose(self, capability_id: str) -> bool:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT required_capabilities FROM capability_compositions WHERE capability_id = ?",
                (capability_id,),
            ).fetchone()
        if not row:
            return False
        required = set(json.loads(row["required_capabilities"]))
        return required <= self.available_capability_ids()

    def mark_capability_used(self, capability_id: str, use: str) -> None:
        capability = self.get_capability(capability_id)
        if not capability:
            return
        if use not in capability.learned_uses:
            capability.learned_uses.append(use)
        self.upsert_capability(capability)

    def _state_rank(self, state: InventoryState) -> int:
        ranks = {
            InventoryState.UNKNOWN: 0,
            InventoryState.KNOWN_BUT_NOT_ACQUIRED: 1,
            InventoryState.DISCOVERED: 2,
            InventoryState.UNVERIFIED: 3,
            InventoryState.BROKEN: 3,
            InventoryState.UNAVAILABLE: 3,
            InventoryState.AVAILABLE: 4,
        }
        return ranks[state]

    def _resource_payload(self, resource: ResourceRecord) -> tuple[Any, ...]:
        return (
            resource.id,
            resource.name,
            resource.type,
            resource.state.value,
            resource.description,
            resource.provider,
            resource.version,
            resource.location,
            resource.discovery_source,
            resource.acquisition_method,
            self.db.dumps([asdict(item) for item in resource.provenance]),
            self.db.dumps(resource.acquisition_history),
            self.db.dumps(resource.verification_results),
            self.db.dumps(resource.limitations),
            resource.confidence,
            resource.last_verified_at,
            resource.last_verified_status,
        )

    def _capability_payload(self, capability: CapabilityRecord) -> tuple[Any, ...]:
        return (
            capability.id,
            capability.name,
            capability.description,
            capability.type.value,
            capability.state.value,
            self.db.dumps(capability.inputs),
            self.db.dumps(capability.outputs),
            self.db.dumps(capability.prerequisites),
            self.db.dumps(capability.effects),
            capability.risk_level.value,
            int(capability.authorization_required),
            capability.reversibility,
            capability.verification_method,
            self.db.dumps([asdict(item) for item in capability.provenance]),
            self.db.dumps(capability.learned_uses),
            self.db.dumps(capability.limitations),
            capability.confidence,
            capability.last_verified_at,
            capability.last_verified_status,
            capability.resources,
        )

    def _resource_from_row(self, row: Any) -> ResourceRecord:
        return ResourceRecord(
            id=row["id"],
            name=row["name"],
            type=row["type"],
            state=InventoryState(row["state"]),
            description=row["description"],
            provider=row["provider"],
            version=row["version"],
            location=row["location"],
            discovery_source=row["discovery_source"],
            acquisition_method=row["acquisition_method"],
            provenance=[Evidence(**item) for item in json.loads(row["provenance"])],
            acquisition_history=json.loads(row["acquisition_history"]),
            verification_results=json.loads(row["verification_results"]),
            limitations=json.loads(row["limitations"]),
            confidence=row["confidence"],
            last_verified_at=row["last_verified_at"],
            last_verified_status=row["last_verified_status"],
        )

    def _capability_from_row(self, row: Any) -> CapabilityRecord:
        with self.db.connect() as conn:
            resources = [
                rel["resource_id"]
                for rel in conn.execute(
                    "SELECT resource_id FROM resource_capabilities WHERE capability_id = ? ORDER BY resource_id",
                    (row["id"],),
                ).fetchall()
            ]
        return CapabilityRecord(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            type=CapabilityType(row["type"]),
            state=InventoryState(row["state"]),
            inputs=json.loads(row["inputs"]),
            outputs=json.loads(row["outputs"]),
            prerequisites=json.loads(row["prerequisites"]),
            effects=json.loads(row["effects"]),
            risk_level=RiskClassification(row["risk_level"]),
            authorization_required=bool(row["authorization_required"]),
            reversibility=row["reversibility"],
            verification_method=row["verification_method"],
            provenance=[Evidence(**item) for item in json.loads(row["provenance"])],
            resources=resources,
            learned_uses=json.loads(row["learned_uses"]),
            limitations=json.loads(row["limitations"]),
            confidence=row["confidence"],
            last_verified_at=row["last_verified_at"],
            last_verified_status=row["last_verified_status"],
        )
