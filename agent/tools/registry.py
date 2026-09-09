from __future__ import annotations

from dataclasses import dataclass

from agent.analysis.information_value import InformationCandidate
from agent.core.acquisition import AcquisitionPlan, AcquisitionStrategy
from agent.core.information import InformationComponent
from agent.core.state import State
from agent.tools.base import OperationType, RiskLevel, Tool
from agent.tools.filesystem import FilesystemReadTool
from agent.tools.system_info import ObservedValue, SystemInfo, SystemInfoTool
from agent.tools.web import WebResearchStub


@dataclass(frozen=True)
class ToolMatch:
    tool: Tool
    matched_capabilities: set[str]


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self.tools = tools or [SystemInfoTool(), FilesystemReadTool(), WebResearchStub()]

    def find_safe_read_tool(self, candidate: InformationCandidate) -> ToolMatch | None:
        plan = self.plan_acquisition(candidate, State(current_objective=""))
        if not plan.selected_strategy:
            return None
        tool = self.get_tool(plan.selected_strategy.tool_name)
        if not tool:
            return None
        return ToolMatch(tool, set(plan.selected_strategy.required_capabilities))

    def get_tool(self, name: str) -> Tool | None:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def plan_acquisition(self, candidate: InformationCandidate, state: State) -> AcquisitionPlan:
        strategies: list[AcquisitionStrategy] = []
        for component in candidate.requirement.unresolved_components:
            strategies.extend(self._strategies_for_component(component, state))
        ranked = sorted(strategies, key=lambda item: item.score, reverse=True)
        if ranked:
            return AcquisitionPlan(
                strategies=ranked,
                selected_strategy=ranked[0],
                status="CONTINUE_ACQUISITION",
                reason="A safe read-only strategy has positive estimated value and has not been exhausted.",
            )
        if candidate.requirement.unresolved_components:
            return AcquisitionPlan(
                strategies=[],
                selected_strategy=None,
                status="PROCEED_UNDER_UNCERTAINTY",
                reason=(
                    "No safe read-only strategy remains for: "
                    + ", ".join(component.name for component in candidate.requirement.unresolved_components)
                ),
            )
        return AcquisitionPlan(
            strategies=[],
            selected_strategy=None,
            status="INFORMATION_RESOLVED",
            reason="All required components are already resolved.",
        )

    def _strategies_for_component(self, component: InformationComponent, state: State) -> list[AcquisitionStrategy]:
        strategies: list[AcquisitionStrategy] = []
        required = set(component.required_information_types)
        for tool in self.tools:
            matched = required & tool.capabilities
            if not matched or tool.operation_type != OperationType.READ:
                continue
            risk_cost = {
                RiskLevel.LOW: 0.05,
                RiskLevel.MEDIUM: 0.25,
                RiskLevel.MODERATE: 0.25,
                RiskLevel.HIGH: 0.8,
                RiskLevel.CRITICAL: 1.0,
            }[tool.risk_level]
            strategy = AcquisitionStrategy(
                id=f"{tool.name}:{component.id}",
                name=f"{tool.name} for {component.name}",
                description=f"Use {tool.name} to observe {component.name}.",
                target_component_ids=[component.id],
                required_capabilities=sorted(matched),
                tool_name=tool.name,
                estimated_information_gain=component.importance,
                estimated_success_probability=0.72 if tool.risk_level == RiskLevel.LOW else 0.45,
                acquisition_cost=0.12,
                time_cost=0.1,
                risk_cost=risk_cost,
                risk_level=tool.risk_level.value,
            )
            if tool.risk_level == RiskLevel.LOW and not state.has_attempted_strategy(component.id, strategy.id):
                strategies.append(strategy)
        return strategies


def facts_from_tool_result(match: ToolMatch, metadata: dict | None) -> list[str]:
    return facts_from_tool_result_by_capabilities(match.tool.name, match.matched_capabilities, metadata)


def facts_from_tool_result_by_capabilities(
    tool_name: str,
    capabilities: set[str],
    metadata: dict | None,
) -> list[str]:
    if tool_name != "system_info" or not metadata:
        return []
    info = SystemInfo(
        **{
            key: ObservedValue(**value) if isinstance(value, dict) else value
            for key, value in metadata.items()
        }
    )
    return info.facts_for_capabilities(capabilities)
