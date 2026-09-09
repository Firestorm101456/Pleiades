from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class OperationType(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str
    metadata: dict[str, Any] | None = None


class Tool(ABC):
    name: str
    description: str
    input_schema: dict[str, Any]
    capabilities: set[str] = set()
    risk_level: RiskLevel
    operation_type: OperationType

    @abstractmethod
    def execute(self, arguments: dict[str, Any], *, authorized: bool = False) -> ToolResult:
        """Run the tool. Consequential operations must check authorization."""

    def require_authorization(self, authorized: bool) -> None:
        if self.operation_type != OperationType.READ and not authorized:
            raise PermissionError(f"{self.name} requires explicit authorization.")
