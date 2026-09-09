from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from agent.capabilities.models import RiskClassification
from agent.memory.database import MemoryDatabase


@dataclass(frozen=True)
class AuthorizationRequest:
    action: str
    risk: RiskClassification
    reason: str
    scope: str


@dataclass(frozen=True)
class AuthorizationDecision:
    approved: bool
    authorization_required: bool
    reason: str


class AuthorizationProvider(Protocol):
    def authorize(self, request: AuthorizationRequest) -> AuthorizationDecision:
        ...


class MockAuthorizationProvider:
    def __init__(self, approve: bool = True, secret: str = "MASTER-TEST-SECRET") -> None:
        self.approve = approve
        self.secret = secret

    def authorize(self, request: AuthorizationRequest) -> AuthorizationDecision:
        return AuthorizationDecision(
            approved=self.approve,
            authorization_required=True,
            reason="Approved by mock provider." if self.approve else "Denied by mock provider.",
        )


class SecurityGate:
    def __init__(self, db: MemoryDatabase, provider: AuthorizationProvider | None = None) -> None:
        self.db = db
        self.provider = provider or MockAuthorizationProvider(approve=False)

    def check(self, request: AuthorizationRequest) -> AuthorizationDecision:
        required = request.risk in {RiskClassification.MODERATE, RiskClassification.HIGH, RiskClassification.CRITICAL}
        if not required:
            decision = AuthorizationDecision(False, False, "No authorization required for low-risk action.")
            self.record(request, decision)
            return decision
        decision = self.provider.authorize(request)
        self.record(request, decision)
        return decision

    def record(self, request: AuthorizationRequest, decision: AuthorizationDecision, execution: str = "", verification: str = "") -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO authorization_events
                (timestamp, requested_action, risk_classification, reason, requested_scope,
                 authorization_required, approved, result_reason, resulting_execution, verification_result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    request.action,
                    request.risk.value,
                    request.reason,
                    request.scope,
                    int(decision.authorization_required),
                    int(decision.approved),
                    decision.reason,
                    execution,
                    verification,
                ),
            )

