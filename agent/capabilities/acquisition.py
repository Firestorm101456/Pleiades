from __future__ import annotations

from typing import Protocol

from agent.capabilities.inventory import CapabilityInventory
from agent.capabilities.models import (
    AcquisitionExecutionResult,
    AcquisitionOption,
    CapabilityRecord,
    CapabilityRequirement,
    CapabilityType,
    InventoryState,
    ResourceRecord,
    RiskClassification,
    utc_now,
)
from agent.security.gate import AuthorizationRequest, SecurityGate


class AcquisitionProvider(Protocol):
    def options_for(self, requirement: CapabilityRequirement) -> list[AcquisitionOption]:
        ...

    def acquire(self, option: AcquisitionOption) -> AcquisitionExecutionResult:
        ...


class MockAcquisitionProvider:
    def __init__(self, fail: bool = False, fail_verification: bool = False) -> None:
        self.fail = fail
        self.fail_verification = fail_verification

    def options_for(self, requirement: CapabilityRequirement) -> list[AcquisitionOption]:
        return [
            AcquisitionOption(
                id=f"mock_user_resource:{requirement.capability_id}",
                acquires_resource_id=f"mock_resource:{requirement.capability_id}",
                provides_capability_ids=[requirement.capability_id],
                source="user-provided mock resource",
                method="user_provided_resource",
                prerequisites=[],
                estimated_cost=0.05,
                estimated_time=0.05,
                reliability=0.9,
                risk=RiskClassification.MODERATE,
                reversibility="removable_inventory_record",
                authorization_required=True,
                verification_method="mock safe self-test",
                rollback_method="remove inventory record",
            )
        ]

    def acquire(self, option: AcquisitionOption) -> AcquisitionExecutionResult:
        if self.fail:
            return AcquisitionExecutionResult(option.id, False, "Mock acquisition failed before verification.")
        verified = not self.fail_verification
        state = InventoryState.AVAILABLE if verified else InventoryState.BROKEN
        resource = ResourceRecord(
            id=option.acquires_resource_id,
            name=option.acquires_resource_id,
            type="mock_resource",
            state=state,
            description="Mock acquired resource used for controlled tests and demos.",
            acquisition_method=option.method,
            confidence=0.9 if verified else 0.2,
            last_verified_at=utc_now(),
            last_verified_status="mock_verification_succeeded" if verified else "mock_verification_failed",
        )
        resource.add_evidence("ACQUIRED_FROM", option.source, "mock_acquisition_provider")
        resource.add_evidence("VERIFIED_BY", option.verification_method, "mock_acquisition_provider")
        capability_state = InventoryState.AVAILABLE if verified else InventoryState.BROKEN
        capabilities = [
            CapabilityRecord(
                id=capability_id,
                name=capability_id.replace("_", " ").title(),
                description=f"Capability acquired through controlled mock provider: {capability_id}.",
                type=CapabilityType.ANALYSIS,
                state=capability_state,
                resources=[resource.id],
                risk_level=option.risk,
                authorization_required=option.authorization_required,
                reversibility=option.reversibility,
                verification_method=option.verification_method,
                learned_uses=[f"Acquired to provide {capability_id}."] if verified else [],
                confidence=0.85 if verified else 0.1,
                last_verified_at=resource.last_verified_at,
                last_verified_status=resource.last_verified_status,
            )
            for capability_id in option.provides_capability_ids
        ]
        return AcquisitionExecutionResult(
            option.id,
            verified,
            "Mock acquisition and verification succeeded." if verified else "Mock verification failed.",
            resource,
            capabilities,
            {"verified": verified, "method": option.verification_method},
        )


class CapabilityAcquisitionManager:
    def __init__(self, inventory: CapabilityInventory, security: SecurityGate, providers: list[AcquisitionProvider] | None = None) -> None:
        self.inventory = inventory
        self.security = security
        self.providers = providers or [MockAcquisitionProvider()]

    def options_for(self, requirement: CapabilityRequirement) -> list[AcquisitionOption]:
        options: list[AcquisitionOption] = []
        for provider in self.providers:
            options.extend(provider.options_for(requirement))
        return sorted(options, key=lambda item: item.score, reverse=True)

    def acquire_best(self, requirement: CapabilityRequirement) -> AcquisitionExecutionResult | None:
        options = self.options_for(requirement)
        if not options:
            return None
        option = options[0]
        request = AuthorizationRequest(
            action=f"Acquire {option.acquires_resource_id}",
            risk=option.risk,
            reason=f"Capability {requirement.capability_id} is missing.",
            scope=option.id,
        )
        decision = self.security.check(request)
        if decision.authorization_required and not decision.approved:
            return AcquisitionExecutionResult(option.id, False, "Authorization denied.")
        result = self.providers[0].acquire(option)
        with self.inventory.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO capability_acquisition_attempts
                (timestamp, option_id, capability_id, resource_id, status, result, state_before, state_after, verification_result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now(),
                    option.id,
                    requirement.capability_id,
                    option.acquires_resource_id,
                    "SUCCEEDED" if result.ok else "FAILED",
                    result.output,
                    InventoryState.UNKNOWN.value,
                    result.resource.state.value if result.resource else InventoryState.UNVERIFIED.value,
                    self.inventory.db.dumps(result.verification),
                ),
            )
        if result.resource:
            self.inventory.upsert_resource(result.resource)
        for capability in result.capabilities:
            self.inventory.upsert_capability(capability)
        return result
