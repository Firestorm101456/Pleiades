from __future__ import annotations

from typing import Any

from agent.tools.base import OperationType, RiskLevel, Tool, ToolResult


class WebResearchStub(Tool):
    name = "web_research"
    description = "Stub interface for future web research. Does not access the network."
    input_schema = {"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}}
    risk_level = RiskLevel.MEDIUM
    operation_type = OperationType.READ

    def execute(self, arguments: dict[str, Any], *, authorized: bool = False) -> ToolResult:
        return ToolResult(
            True,
            f"Web research stub recorded query: {arguments['query']}",
            {"network_access": False},
        )

