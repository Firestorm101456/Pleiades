from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.capabilities.acquisition import CapabilityAcquisitionManager, MockAcquisitionProvider
from agent.capabilities.inventory import CapabilityInventory
from agent.capabilities.models import CapabilityRequirement
from agent.config import Settings, settings
from agent.core.engine import CoreEngine
from agent.memory.database import MemoryDatabase
from agent.security.gate import MockAuthorizationProvider, SecurityGate
from agent.core.agent import AnalyticalAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Pleiades analytical loop.")
    parser.add_argument("objective", help="Objective or decision to analyze.")
    parser.add_argument(
        "--observation",
        action="append",
        default=[],
        help="Observation to seed the current state. Can be passed multiple times.",
    )
    parser.add_argument("--demo-capabilities", action="store_true", help="Run the Milestone 6 capability inventory demo.")
    parser.add_argument("--engine", action="store_true", help="Run the Milestone 7 core engine.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.demo_capabilities:
        run_capability_demo()
        return
    if args.engine:
        run_engine(args.objective, args.observation)
        return
    agent = AnalyticalAgent(settings)
    result = agent.analyze(args.objective, args.observation)

    print(f"Objective: {result.state.current_objective}")
    print("\nCurrent knowledge:")
    for fact in result.state.known_facts or ["No observations supplied."]:
        print(f"- {fact}")
    print(f"\nInformation status: {result.information_value.status.value}")
    print(f"Reason: {result.information_value.stop_reason}")
    if result.tool_observations:
        print("\nTool observations:")
        for observation in result.tool_observations:
            print(f"- Tool selected: {observation.tool_name}")
            print(f"  Strategy: {observation.strategy_name}")
            print(f"  Status: {observation.acquisition_status}")
            print(f"  Reason: {observation.acquisition_reason}")
            print(f"  Need: {observation.information}")
            print(f"  Risk: {observation.operation_type.upper()} / {observation.risk_level.upper()}")
            print(f"  Capabilities: {', '.join(observation.matched_capabilities)}")
            for fact in observation.facts:
                print(f"  Observation: {fact}")
    if result.state.acquisition_attempts:
        print("\nAcquisition attempts:")
        for attempt in result.state.acquisition_attempts:
            print(f"- {attempt.component_name} via {attempt.strategy_name}: {attempt.status.value}")
            print(f"  Reason: {attempt.reason}")
    print("\nRanked information value:")
    for candidate in result.information_value.candidates[:5]:
        marker = " YES" if candidate.recommended else " NO"
        print(f"- {candidate.information}")
        print(f"  Value: {candidate.value.value}; uncertainty: {candidate.uncertainty.value}; recommended:{marker}")
        print(f"  Why: {candidate.why_it_matters}")
        print(f"  Impact: {candidate.potential_decision_impact}")
        print(f"  Score: {candidate.score:.2f} estimated, not measured")
        print(f"  Requirement status: {candidate.requirement.status.value}")
        for component in candidate.requirement.components:
            value = component.value if component.value is not None else "UNKNOWN"
            print(f"    Component: {component.name} = {value} [{component.status.value}]")
    print(f"Selected action: {result.selected_action.description}")
    print(f"Expected utility: {result.selected_action.expected_utility:.3f}")
    print(f"Confidence: {result.selected_action.probability_of_success:.2f} estimated, not measured")
    print(f"Final result: {result.final_result}")
    print("\nTop risks:")
    for risk in result.selected_action.major_risks:
        print(f"- {risk}")
    print("\nMissing information:")
    for item in result.critique.missing_information:
        print(f"- {item}")


def run_capability_demo() -> None:
    with TemporaryDirectory() as directory:
        demo_settings = Settings(database_path=Path(directory) / "capability_demo.sqlite3")
        _run_capability_demo(demo_settings)


def run_engine(objective: str, observations: list[str]) -> None:
    engine = CoreEngine(settings)
    result = engine.run(objective, observations)

    print(f"Objective: {result.objective_model.objective_text}")
    print(f"Desired outcome: {result.objective_model.desired_outcome}")
    print("\nObjective interpretation:")
    for element in result.objective_model.semantic_elements:
        print(f"- {element.role.value}: {element.value}")
    for fact in result.objective_model.explicit_facts:
        print(f"- USER_PROVIDED: {fact.text}")
    for fact in result.objective_model.inferred_facts:
        print(f"- INFERRED: {fact.text}")
    for ambiguity in result.objective_model.ambiguities:
        print(f"- AMBIGUITY ({ambiguity.impact}): {ambiguity.text}")
    if result.objective_model.unknown_variables:
        print("Unknown variables:")
        for variable in result.objective_model.unknown_variables:
            print(f"- {variable}")
    else:
        print("Unknown variables: None identified at this stage.")

    print("\nGenerated tasks:")
    for task in result.tasks:
        deps = ", ".join(task.dependencies) or "none"
        print(f"- {task.title}: {task.status.value}; deps={deps}")

    print("\nExecution cycles:")
    for cycle in result.cycles:
        print(f"- Cycle {cycle.cycle}: {cycle.task_title or 'none'}")
        print(f"  Action: {cycle.action}")
        print(f"  Status: {cycle.status.value}")
        if cycle.requirements:
            print(f"  Requirements: {', '.join(cycle.requirements)}")
        if cycle.capability_resolution:
            print(f"  Capability resolution: {', '.join(cycle.capability_resolution)}")
        print(f"  Observation: {cycle.observation}")
        print(f"  Verification: {cycle.verification}")

    print("\nFinal result:")
    print(result.final_result)


def _run_capability_demo(demo_settings: Settings) -> None:
    agent = AnalyticalAgent(demo_settings)
    inventory = agent.inventory

    print("OBJECTIVE")
    print("- Demonstrate dynamic capability, resource, acquisition, and security inventory.")

    print("\nDISCOVERY")
    result = agent.discovery.discover()
    for resource in result.resources:
        print(f"- {resource.id}: {resource.state.value} ({resource.discovery_source})")

    print("\nINVENTORY CHECK")
    for resource in inventory.resources():
        print(f"- Resource {resource.id}: {resource.state.value}")

    print("\nKNOWN CAPABILITIES")
    for capability in inventory.capabilities():
        print(f"- {capability.id}: {capability.state.value} via {', '.join(capability.resources) or 'no verified resource'}")

    requirement = CapabilityRequirement("summarize_tabular_data", "Generic task requires summarizing structured data.")
    missing = inventory.find_missing([requirement])

    print("\nCAPABILITY REQUIRED")
    print(f"- {requirement.capability_id}: {requirement.reason}")

    print("\nMISSING CAPABILITY")
    for item in missing:
        print(f"- {item.requirement.capability_id}: {item.current_state.value}; {item.reason}")

    db = MemoryDatabase(demo_settings.database_path)
    security = SecurityGate(db, MockAuthorizationProvider(approve=True))
    manager = CapabilityAcquisitionManager(inventory, security, [MockAcquisitionProvider()])
    options = manager.options_for(requirement)

    print("\nACQUISITION OPTIONS")
    for option in options:
        print(f"- {option.id}: method={option.method}; risk={option.risk.value}; authorization_required={option.authorization_required}")

    selected = options[0]
    print("\nRISK ASSESSMENT")
    print(f"- Selected {selected.id} with scoped authorization: {selected.authorization_required}")

    print("\nAUTHORIZATION")
    acquisition = manager.acquire_best(requirement)
    print(f"- Result: {'approved and executed' if acquisition and acquisition.ok else 'not executed'}")

    print("\nVERIFICATION")
    if acquisition:
        print(f"- {acquisition.output}")

    print("\nINVENTORY UPDATE")
    restarted_inventory = CapabilityInventory(MemoryDatabase(demo_settings.database_path))
    acquired = restarted_inventory.get_capability(requirement.capability_id)
    print(f"- {requirement.capability_id}: {acquired.state.value if acquired else 'UNKNOWN'}")

    print("\nLEARNED CAPABILITIES")
    if acquired:
        for use in acquired.learned_uses:
            print(f"- {use}")

    print("\nNEXT ACTION")
    second_missing = restarted_inventory.find_missing([requirement])
    print("- Second task reuses learned capability." if not second_missing else "- Capability is still missing.")


if __name__ == "__main__":
    main()
