from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path

from agent.analysis.information_value import InformationValueAnalyzer
from agent.capabilities.acquisition import CapabilityAcquisitionManager, MockAcquisitionProvider
from agent.capabilities.discovery import EnvironmentDiscovery
from agent.capabilities.inventory import CapabilityInventory
from agent.capabilities.models import (
    CapabilityComposition,
    CapabilityRecord,
    CapabilityRequirement,
    CapabilityType,
    InventoryState,
    ResourceRecord,
    RiskClassification,
)
from agent.config import Settings
from agent.core.engine import CoreEngine, EngineStatus
from agent.core.facts import FactSource
from agent.core.facts import SemanticRole
from agent.core.acquisition import AcquisitionAttempt, AcquisitionResultStatus
from agent.core.information import ComponentStatus, InformationComponent, InformationRequirement, RequirementStatus
from agent.core.agent import AnalyticalAgent
from agent.core.state import State
from agent.core.task_manager import TaskManager
from agent.core.tasks import RequirementCategory, Task, TaskStatus
from agent.reasoning.decomposer import TaskDecomposer
from agent.reasoning.goal import GoalInterpreter
from agent.reasoning.requirements import RequirementGenerator
from agent.tools.python_tool import PythonExecutionTool
from agent.tools.registry import ToolRegistry
from agent.memory.database import MemoryDatabase
from agent.security.gate import MockAuthorizationProvider, SecurityGate


class AgentTests(unittest.TestCase):
    def test_agent_runs_and_persists_analysis(self):
        with self.subTest("analysis is persisted"):
            from tempfile import TemporaryDirectory
            from pathlib import Path

            with TemporaryDirectory() as directory:
                settings = Settings(database_path=Path(directory) / "memory.sqlite3")
                agent = AnalyticalAgent(settings)

                result = agent.analyze("Choose the safest next step for a research task", ["deadline is tomorrow"])

                self.assertTrue(result.selected_action.description)
                self.assertIn("Recommended action", result.final_result)
                self.assertTrue(result.information_value.candidates)

                with sqlite3.connect(settings.database_path) as conn:
                    count = conn.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0]

                self.assertEqual(count, 1)

    def test_information_value_prioritizes_decision_relevant_unknowns(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            settings = Settings(database_path=Path(directory) / "memory.sqlite3")
            agent = AnalyticalAgent(settings)

            result = agent.analyze("Choose the best local LLM for my computer", ["must run locally"])
            ranked = [item.information for item in result.information_value.candidates[:5]]

            self.assertIn(result.information_value.status.value, {"NEEDED", "SUFFICIENT"})
            self.assertIn(result.information_value.top_candidate.information, ranked)
            self.assertTrue(any("RAM" in item or "VRAM" in item or "workload" in item for item in ranked))
            self.assertTrue(result.information_value.top_candidate.recommended)

    def test_agent_acquires_safe_system_info_automatically(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            settings = Settings(database_path=Path(directory) / "memory.sqlite3")
            agent = AnalyticalAgent(settings)

            result = agent.analyze("Choose the best local LLM for my computer", ["must run locally"])

            self.assertTrue(result.tool_observations)
            observation = result.tool_observations[0]
            self.assertEqual(observation.tool_name, "system_info")
            self.assertEqual(observation.operation_type, "read")
            self.assertEqual(observation.risk_level, "low")
            self.assertTrue(observation.facts)
            self.assertTrue(any("System RAM" in fact or "GPU" in fact for fact in result.state.known_facts))

    def test_compound_requirement_can_be_partially_resolved(self):
        requirement = InformationRequirement(
            id="gpu_model_and_vram",
            name="GPU model and VRAM",
            description="Hardware requirement",
            components=[
                InformationComponent(
                    id="gpu_model",
                    name="GPU model",
                    description="GPU model",
                    required_information_types=["gpu"],
                    status=ComponentStatus.MEASURED,
                    value="Intel Iris Plus Graphics 655",
                ),
                InformationComponent(
                    id="gpu_vram",
                    name="GPU VRAM",
                    description="GPU VRAM",
                    required_information_types=["gpu_vram"],
                    status=ComponentStatus.UNKNOWN,
                ),
            ],
        )

        self.assertEqual(requirement.status, RequirementStatus.PARTIALLY_RESOLVED)
        self.assertEqual([component.name for component in requirement.unresolved_components], ["GPU VRAM"])

    def test_known_component_is_not_requested_again(self):
        state = State(
            current_objective="Choose the best local LLM for my computer",
            known_facts=["GPU model: Intel Iris Plus Graphics 655 (MEASURED, source=system_profiler)"],
        )
        analyzer = InformationValueAnalyzer()

        candidate = analyzer._score_candidate("GPU model and VRAM", state, ["Gather information"])

        self.assertEqual(candidate.requirement.status, RequirementStatus.PARTIALLY_RESOLVED)
        self.assertEqual(candidate.required_information_types, ["gpu_vram"])

    def test_acquisition_strategy_attempt_prevents_identical_retry(self):
        state = State(
            current_objective="Choose the best local LLM for my computer",
            known_facts=["GPU model: Intel Iris Plus Graphics 655 (MEASURED, source=system_profiler)"],
        )
        analyzer = InformationValueAnalyzer()
        registry = ToolRegistry()
        candidate = analyzer._score_candidate("GPU model and VRAM", state, ["Gather information"])

        plan = registry.plan_acquisition(candidate, state)
        self.assertEqual(plan.status, "CONTINUE_ACQUISITION")
        self.assertIsNotNone(plan.selected_strategy)

        strategy = plan.selected_strategy
        state.record_acquisition_attempt(
            AcquisitionAttempt.record(
                component_id="gpu_vram",
                component_name="GPU VRAM",
                strategy=strategy,
                status=AcquisitionResultStatus.UNKNOWN,
                result="GPU memory/VRAM: UNKNOWN (source=system_profiler)",
                reason="The observation method did not expose the target component.",
            )
        )

        retry_plan = registry.plan_acquisition(candidate, state)
        self.assertEqual(retry_plan.status, "PROCEED_UNDER_UNCERTAINTY")
        self.assertIsNone(retry_plan.selected_strategy)

    def test_agent_records_component_acquisition_attempts(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            settings = Settings(database_path=Path(directory) / "memory.sqlite3")
            agent = AnalyticalAgent(settings)

            result = agent.analyze("Choose the best local LLM for my computer", ["must run locally"])

            self.assertTrue(result.state.acquisition_attempts)
            self.assertTrue(all(attempt.strategy_name for attempt in result.state.acquisition_attempts))
            self.assertTrue(any(attempt.component_name == "GPU VRAM" for attempt in result.state.acquisition_attempts))
            self.assertIn("Proceed under uncertainty", result.selected_action.description)

    def test_information_value_can_be_sufficient(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        observations = [
            "system RAM is 32GB",
            "GPU model is M3 Max with 36GB VRAM",
            "primary workload is code assistance and summarization",
            "required context length is medium",
        ]
        with TemporaryDirectory() as directory:
            settings = Settings(database_path=Path(directory) / "memory.sqlite3")
            agent = AnalyticalAgent(settings)

            result = agent.analyze("Choose the best local LLM for my computer", observations)

            self.assertEqual(result.information_value.status.value, "SUFFICIENT")
            self.assertTrue(result.information_value.stop_reason)

    def test_python_tool_requires_authorization(self):
        tool = PythonExecutionTool()

        with self.assertRaises(PermissionError):
            tool.execute({"code": "print(1)"})

        result = tool.execute({"code": "print(1)"}, authorized=True)

        self.assertTrue(result.ok)
        self.assertEqual(result.output.strip(), "1")

    def test_resource_discovery_and_verification_are_separate(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            inventory = CapabilityInventory(MemoryDatabase(Path(directory) / "memory.sqlite3"))
            discovery = EnvironmentDiscovery(inventory)

            discovered = discovery.discover()
            python_resource = next(item for item in discovered.resources if item.id == "executable:python")
            self.assertEqual(python_resource.state, InventoryState.DISCOVERED)

            verified = discovery.verify_known_safe()
            verified_python = next(item for item in verified.resources if item.id == "executable:python")
            self.assertEqual(verified_python.state, InventoryState.AVAILABLE)
            self.assertTrue(verified_python.version)

    def test_resource_deduplication_and_relationship_shapes(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            inventory = CapabilityInventory(MemoryDatabase(Path(directory) / "memory.sqlite3"))
            resource = ResourceRecord("tool:one", "one", "executable", InventoryState.AVAILABLE)
            capability_a = CapabilityRecord("cap_a", "A", "A", CapabilityType.SENSOR, InventoryState.AVAILABLE, resources=["tool:one"])
            capability_b = CapabilityRecord("cap_b", "B", "B", CapabilityType.ANALYSIS, InventoryState.AVAILABLE, resources=["tool:one"])
            inventory.upsert_resource(resource)
            inventory.upsert_resource(resource)
            inventory.upsert_capability(capability_a)
            inventory.upsert_capability(capability_b)

            self.assertEqual(len(inventory.resources()), 1)
            self.assertEqual({cap.id for cap in inventory.capabilities()}, {"cap_a", "cap_b"})
            self.assertEqual(inventory.get_capability("cap_a").resources, ["tool:one"])

    def test_multiple_resources_can_provide_one_capability(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            inventory = CapabilityInventory(MemoryDatabase(Path(directory) / "memory.sqlite3"))
            inventory.upsert_resource(ResourceRecord("tool:a", "a", "executable", InventoryState.AVAILABLE))
            inventory.upsert_resource(ResourceRecord("tool:b", "b", "python_module", InventoryState.AVAILABLE))
            inventory.upsert_capability(
                CapabilityRecord("extract_text", "Extract", "Extract", CapabilityType.ANALYSIS, InventoryState.AVAILABLE, resources=["tool:a", "tool:b"])
            )

            self.assertEqual(inventory.get_capability("extract_text").resources, ["tool:a", "tool:b"])

    def test_missing_capability_detection_and_composition(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            inventory = CapabilityInventory(MemoryDatabase(Path(directory) / "memory.sqlite3"))
            for cap_id in ["read_files", "parse_json", "calculate"]:
                inventory.upsert_capability(CapabilityRecord(cap_id, cap_id, cap_id, CapabilityType.ANALYSIS, InventoryState.AVAILABLE))
            inventory.add_composition(CapabilityComposition("transform_json_data", ["read_files", "parse_json", "calculate"], "Compose basic capabilities."))

            missing = inventory.find_missing([CapabilityRequirement("extract_pdf_text", "Need PDF extraction.")])

            self.assertEqual(missing[0].current_state, InventoryState.UNKNOWN)
            self.assertTrue(inventory.can_compose("transform_json_data"))

    def test_authorized_acquisition_persists_and_reuses_capability(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            db = MemoryDatabase(Path(directory) / "memory.sqlite3")
            inventory = CapabilityInventory(db)
            gate = SecurityGate(db, MockAuthorizationProvider(approve=True, secret="SHOULD_NOT_LEAK"))
            manager = CapabilityAcquisitionManager(inventory, gate, [MockAcquisitionProvider()])
            requirement = CapabilityRequirement("summarize_tabular_data", "Needed for generic task.")

            result = manager.acquire_best(requirement)
            restarted = CapabilityInventory(MemoryDatabase(Path(directory) / "memory.sqlite3"))

            self.assertTrue(result.ok)
            self.assertEqual(restarted.get_capability("summarize_tabular_data").state, InventoryState.AVAILABLE)
            self.assertEqual(restarted.find_missing([requirement]), [])
            with db.connect() as conn:
                auth_row = conn.execute("SELECT * FROM authorization_events").fetchone()
                serialized = "\n".join(str(value) for value in auth_row)
            self.assertNotIn("SHOULD_NOT_LEAK", serialized)

    def test_authorization_denied_and_failed_verification_do_not_mark_available(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            db = MemoryDatabase(Path(directory) / "memory.sqlite3")
            requirement = CapabilityRequirement("new_capability", "Needed.")
            denied_manager = CapabilityAcquisitionManager(
                CapabilityInventory(db),
                SecurityGate(db, MockAuthorizationProvider(approve=False)),
                [MockAcquisitionProvider()],
            )
            denied = denied_manager.acquire_best(requirement)
            self.assertFalse(denied.ok)
            self.assertIsNone(CapabilityInventory(db).get_capability("new_capability"))

            failed_manager = CapabilityAcquisitionManager(
                CapabilityInventory(db),
                SecurityGate(db, MockAuthorizationProvider(approve=True)),
                [MockAcquisitionProvider(fail_verification=True)],
            )
            failed = failed_manager.acquire_best(requirement)
            self.assertFalse(failed.ok)
            self.assertEqual(CapabilityInventory(db).get_capability("new_capability").state, InventoryState.BROKEN)

    def test_objective_survives_capability_inventory_initialization(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            objective = "Choose the best local LLM for my computer"
            agent = AnalyticalAgent(Settings(database_path=Path(directory) / "memory.sqlite3"))
            result = agent.analyze(objective, ["must run locally"])

            self.assertEqual(result.state.current_objective, objective)

    def test_known_but_not_acquired_is_not_available(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            inventory = CapabilityInventory(MemoryDatabase(Path(directory) / "memory.sqlite3"))
            inventory.upsert_resource(ResourceRecord("known:ffmpeg", "ffmpeg", "executable", InventoryState.KNOWN_BUT_NOT_ACQUIRED))
            inventory.upsert_capability(
                CapabilityRecord("convert_media", "Convert", "Known capability", CapabilityType.ACTUATOR, InventoryState.KNOWN_BUT_NOT_ACQUIRED)
            )

            self.assertNotIn("convert_media", inventory.available_capability_ids())

    def test_goal_understanding_preserves_facts_inferences_and_ambiguity(self):
        model = GoalInterpreter().interpret(
            "Research whether this works. My income is $100,000 and my maiden voyage is in two weeks.",
            ["must avoid high risk"],
        )

        self.assertTrue(any(fact.source == FactSource.USER_PROVIDED and "income" in fact.text.lower() for fact in model.explicit_facts))
        self.assertTrue(any(fact.source == FactSource.INFERRED and "approximately 2 weeks" in fact.text for fact in model.inferred_facts))
        self.assertTrue(any(ambiguity.impact == "HIGH" for ambiguity in model.ambiguities))

    def test_decomposer_generates_different_domain_general_task_structures(self):
        interpreter = GoalInterpreter()
        decomposer = TaskDecomposer()

        research = decomposer.decompose(interpreter.interpret("Research whether buying an electric car makes financial sense for me."))
        repair = decomposer.decompose(interpreter.interpret("My Mac is running slowly. Find out why and fix it."))

        self.assertNotEqual([task.purpose for task in research], [task.purpose for task in repair])
        self.assertTrue(any(task.purpose == "research" for task in research))
        self.assertTrue(any(task.purpose == "execution" for task in repair))
        source = Path(__file__).parents[1] / "agent" / "reasoning" / "decomposer.py"
        text = source.read_text()
        self.assertNotIn("create_fishing_tasks", text)
        self.assertNotIn("create_farming_tasks", text)

    def test_task_manager_dependencies_ready_blocked_and_completion(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            manager = TaskManager(MemoryDatabase(Path(directory) / "memory.sqlite3"))
            first = Task("obj:first", "obj", "First", "First", "research", priority=0.5)
            second = Task("obj:second", "obj", "Second", "Second", "analysis", priority=0.9, dependencies=["obj:first"])
            manager.register([first, second])

            self.assertEqual([task.task_id for task in manager.ready_tasks()], ["obj:first"])
            self.assertEqual(manager.tasks["obj:second"].status, TaskStatus.BLOCKED)
            manager.start("obj:first")
            manager.complete("obj:first", "done")
            self.assertEqual([task.task_id for task in manager.ready_tasks()], ["obj:second"])

    def test_requirement_generation_is_task_specific(self):
        generator = RequirementGenerator()
        task = Task("t1", "obj", "Analyze options", "Compare tradeoffs from evidence.", "analysis", required_capabilities=["calculate"])

        requirements = generator.generate(task, State(current_objective="Compare repair cost against replacement cost."))
        categories = {requirement.category for requirement in requirements}

        self.assertIn(RequirementCategory.CAPABILITY, categories)
        self.assertIn(RequirementCategory.ACTION, categories)
        self.assertNotIn("hard constraints", " ".join(requirement.description.lower() for requirement in requirements))
        self.assertTrue(all(requirement.source for requirement in requirements))
        self.assertTrue(all(requirement.reason for requirement in requirements))

    def test_engine_preserves_objective_executes_and_generates_dynamic_result(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            objective = "Research whether buying an electric car makes financial sense for me."
            engine = CoreEngine(Settings(database_path=Path(directory) / "memory.sqlite3"), max_cycles=8)

            result = engine.run(objective, ["budget is $20,000"])

            self.assertEqual(result.state.current_objective, objective)
            self.assertTrue(result.cycles)
            self.assertTrue(any(cycle.observation for cycle in result.cycles))
            self.assertNotEqual(result.status, EngineStatus.COMPLETE)
            self.assertTrue(any(task.status == TaskStatus.BLOCKED for task in result.tasks))
            self.assertIn("Status: PARTIALLY_COMPLETED", result.final_result)
            self.assertNotIn("COMPLETED research from external sources", result.final_result)

    def test_engine_requests_specific_clarification_for_high_impact_ambiguity(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            engine = CoreEngine(Settings(database_path=Path(directory) / "memory.sqlite3"), max_cycles=4)
            result = engine.run("Research viability with income = $100,000.")

            self.assertEqual(result.status, EngineStatus.REQUEST_CLARIFICATION)
            self.assertTrue(result.clarification_requests)
            self.assertIn("income", result.clarification_requests[0].lower())

    def test_deadlines_influence_ready_task_priority(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            manager = TaskManager(MemoryDatabase(Path(directory) / "memory.sqlite3"))
            normal = Task("normal", "obj", "Normal", "Normal", "research", priority=0.6)
            urgent = Task("urgent", "obj", "Urgent", "Urgent", "research", priority=0.61, deadline="tomorrow")
            manager.register([normal, urgent])

            self.assertEqual(manager.ready_tasks()[0].task_id, "urgent")

    def test_unauthorized_corrective_action_blocks_objective(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            engine = CoreEngine(Settings(database_path=Path(directory) / "memory.sqlite3"), max_cycles=4)
            result = engine.run("My Mac is slow. Fix it.")

            self.assertNotEqual(result.status, EngineStatus.COMPLETE)
            self.assertTrue(any(task.status == TaskStatus.BLOCKED for task in result.tasks))

    def test_unrelated_objectives_do_not_share_identical_requirements(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        objectives = [
            "Research whether buying an electric car makes financial sense for me.",
            "Gather information on farming and decide what I should do.",
            "As a new captain who has just bought a new longliner, gather the requirements needed for my maiden voyage. The voyage starts in two weeks and will last approximately three weeks in the Atlantic. Income = $100,000.",
            "My Mac has become very slow. Determine what is causing it.",
        ]
        requirement_sets = []
        with TemporaryDirectory() as directory:
            for index, objective in enumerate(objectives):
                engine = CoreEngine(Settings(database_path=Path(directory) / f"memory-{index}.sqlite3"), max_cycles=3)
                result = engine.run(objective)
                requirement_sets.append(
                    {
                        requirement.description
                        for task in result.tasks
                        for requirement in task.requirements
                    }
                )

        self.assertEqual(len({tuple(sorted(items)) for items in requirement_sets}), len(objectives))
        self.assertFalse(requirement_sets[0] & {"vessel readiness", "crew", "fuel"})
        self.assertFalse(requirement_sets[3] & {"candidate vehicle prices", "fishing authorization"})

    def test_no_universal_missing_information_phrases_remain_in_runtime_code(self):
        root = Path(__file__).parents[1] / "agent"
        runtime_text = "\n".join(path.read_text() for path in root.rglob("*.py"))

        forbidden = [
            "Which evidence would materially change the recommendation?",
            "What constraints or resources are missing from the state?",
            "What downside cases have not been observed yet?",
            "hard constraints",
            "available options",
            "evaluation criteria",
            "No missing required capability.",
        ]
        for phrase in forbidden:
            self.assertNotIn(phrase, runtime_text)

    def test_research_without_verified_capability_blocks_not_completes(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            engine = CoreEngine(Settings(database_path=Path(directory) / "memory.sqlite3"), max_cycles=5)
            result = engine.run("Gather information on farming and decide what I should do.")

            self.assertNotEqual(result.status, EngineStatus.COMPLETE)
            blocked = [task for task in result.tasks if task.status == TaskStatus.BLOCKED]
            self.assertTrue(blocked)
            self.assertTrue(any("capability" in detail.lower() for task in blocked for detail in task.failure_information))
            self.assertNotIn("Status: COMPLETE", result.final_result)

    def test_ev_objective_does_not_turn_language_tokens_into_unknowns(self):
        model = GoalInterpreter().interpret(
            "Research whether buying an electric car makes financial sense for me.",
            ["budget = $20,000"],
        )

        unknowns = {item.lower() for item in model.unknown_variables}
        self.assertFalse({"buying", "electric", "makes", "financial"} & unknowns)
        self.assertIn("annual_mileage", unknowns)
        self.assertTrue(any(element.role == SemanticRole.OBJECT and "electric car" in element.value for element in model.semantic_elements))
        self.assertTrue(any(element.role == SemanticRole.DECISION_CRITERION for element in model.semantic_elements))

    def test_fishing_objective_semantic_roles_not_token_unknowns(self):
        model = GoalInterpreter().interpret(
            "As a new captain who has bought a longliner, prepare for a three-week Atlantic maiden voyage."
        )

        unknowns = {item.lower() for item in model.unknown_variables}
        self.assertFalse({"captain", "bought", "longliner", "atlantic"} & unknowns)
        self.assertIn("new captain", model.actors)
        self.assertTrue(any("longliner" in value for value in model.attributes + model.objects))
        self.assertTrue(any(element.role == SemanticRole.TIME for element in model.semantic_elements))

    def test_task_derived_unknowns_do_not_come_from_words(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            engine = CoreEngine(Settings(database_path=Path(directory) / "memory.sqlite3"), max_cycles=5)
            result = engine.run(
                "Research whether buying an electric car makes financial sense for me.",
                ["budget = $20,000"],
            )

            requirements = [
                requirement.description
                for task in result.tasks
                for requirement in task.requirements
                if requirement.category == RequirementCategory.INFORMATION
            ]
            self.assertIn("annual_mileage", requirements)
            self.assertNotIn("electric", requirements)
            self.assertNotIn("buying", requirements)

    def test_required_regression_objectives_have_distinct_semantic_requirements(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        objectives = [
            ("ev", "Research whether buying an electric car makes financial sense for me."),
            (
                "fishing",
                "As a new captain who has just bought a new fishing vessel (longliner), gather all the requirements needed for my maiden voyage which will take place 2 weeks from now for about 3 weeks in the Atlantic and don't forget to name your vessel (income = $100,000).",
            ),
            ("mac", "My Mac has become very slow. Determine what is causing it."),
        ]
        requirement_sets: dict[str, set[str]] = {}
        with TemporaryDirectory() as directory:
            for name, objective in objectives:
                engine = CoreEngine(Settings(database_path=Path(directory) / f"{name}.sqlite3"), max_cycles=5)
                result = engine.run(objective)
                requirement_sets[name] = {
                    requirement.description
                    for task in result.tasks
                    for requirement in task.requirements
                }

        self.assertNotEqual(requirement_sets["ev"], requirement_sets["fishing"])
        self.assertNotEqual(requirement_sets["ev"], requirement_sets["mac"])
        self.assertTrue(any("annual_mileage" in item for item in requirement_sets["ev"]))
        self.assertTrue(any("root_cause" in item for item in requirement_sets["mac"]))

    def test_no_token_based_unknown_generation_remains(self):
        root = Path(__file__).parents[1] / "agent"
        runtime_text = "\n".join(path.read_text() for path in root.rglob("*.py"))

        self.assertNotIn("Unresolved objective term", runtime_text)
        self.assertNotIn("meaning of objective term", runtime_text)
        self.assertNotIn("_anchors", runtime_text)


if __name__ == "__main__":
    unittest.main()
