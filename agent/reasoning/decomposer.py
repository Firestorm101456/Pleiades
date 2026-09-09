from __future__ import annotations

import re

from agent.capabilities.models import RiskClassification
from agent.core.facts import ObjectiveModel
from agent.core.tasks import Task


class TaskDecomposer:
    def decompose(self, objective: ObjectiveModel) -> list[Task]:
        tasks: list[Task] = []
        root_id = objective.objective_id
        lowered = objective.objective_text.lower()
        subject = self._subject(objective.objective_text)

        tasks.append(
            Task(
                task_id=f"{root_id}:understand",
                objective_id=root_id,
                title=f"Model requested outcome for {subject}",
                description=f"Represent facts, constraints, ambiguities, and success conditions found in: {objective.objective_text}",
                purpose="understanding",
                priority=0.95,
                expected_outcomes=[f"Structured model exists for {objective.desired_outcome}."],
                completion_criteria=["Explicit and inferred facts are labeled in the objective model."],
            )
        )
        previous = tasks[-1].task_id

        if objective.ambiguities:
            high = [item for item in objective.ambiguities if item.impact == "HIGH"]
            tasks.append(
                Task(
                    task_id=f"{root_id}:clarify",
                    objective_id=root_id,
                    title=f"Resolve ambiguity affecting {subject}",
                    description="; ".join(item.clarification_question for item in objective.ambiguities),
                    purpose="clarification",
                    priority=0.9 if high else 0.55,
                    dependencies=[previous],
                    expected_outcomes=["Material ambiguity is clarified or represented as scenarios."],
                    completion_criteria=["High-impact ambiguity has an explicit handling path."],
                    uncertainty=0.8 if high else 0.45,
                )
            )
            previous = tasks[-1].task_id

        if any(word in lowered for word in ("research", "gather", "find out", "whether", "decide", "choose", "why", "diagnose", "determine", "causing", "cause", "requirements")):
            tasks.append(
                Task(
                    task_id=f"{root_id}:{self._slug('evidence ' + subject)}",
                    objective_id=root_id,
                    title=f"Establish evidence needed for {subject}",
                    description=f"Identify and acquire evidence relevant to {objective.desired_outcome}: {objective.objective_text}",
                    purpose="research",
                    priority=0.82,
                    dependencies=[previous],
                    required_capabilities=self._research_capabilities(lowered),
                    expected_outcomes=[f"Evidence related to {subject} is available or explicitly blocked."],
                    completion_criteria=[f"At least one evidence item for {subject} is observed or an acquisition blocker is recorded."],
                )
            )
            previous = tasks[-1].task_id

        if any(word in lowered for word in ("calculate", "financial", "viable", "$", "cost", "budget", "compare", "analyz")):
            tasks.append(
                Task(
                    task_id=f"{root_id}:analysis",
                    objective_id=root_id,
                    title=f"Analyze tradeoffs for {subject}",
                    description=f"Compare available evidence, quantities, risks, and uncertainty for: {objective.objective_text}",
                    purpose="analysis",
                    priority=0.75,
                    dependencies=[previous],
                    required_capabilities=["calculate"],
                    expected_outcomes=[f"Analysis supports, rejects, or blocks candidate paths for {subject}."],
                    completion_criteria=["Conclusion is based on observed facts or explicitly labeled assumptions."],
                )
            )
            previous = tasks[-1].task_id

        if any(word in lowered for word in ("prepare", "plan", "everything necessary")):
            tasks.append(
                Task(
                    task_id=f"{root_id}:plan",
                    objective_id=root_id,
                    title=f"Plan required work for {subject}",
                    description=f"Build ordered work, dependencies, resources, risks, and checks for: {objective.objective_text}",
                    purpose="planning",
                    priority=0.78,
                    dependencies=[previous],
                    expected_outcomes=[f"A task sequence exists for {subject}."],
                    completion_criteria=["Plan is grounded in generated tasks and unresolved blockers."],
                )
            )
            previous = tasks[-1].task_id

        if any(word in lowered for word in ("fix", "modify", "create", "install", "execute")):
            tasks.append(
                Task(
                    task_id=f"{root_id}:safe-action",
                    objective_id=root_id,
                    title=f"Select safe action for {subject}",
                    description=f"Choose an authorized action only if evidence supports changing state for: {objective.objective_text}",
                    purpose="execution",
                    priority=0.72,
                    dependencies=[previous],
                    required_capabilities=["run_python"] if "python" in lowered else [],
                    risk=RiskClassification.MODERATE,
                    expected_outcomes=[f"Action for {subject} is executed or blocked with evidence."],
                    completion_criteria=["Action outcome is observed and verified separately from execution."],
                )
            )
            previous = tasks[-1].task_id

        tasks.append(
            Task(
                task_id=f"{root_id}:result",
                objective_id=root_id,
                title=f"Report actual progress on {subject}",
                description=f"Produce a result from completed tasks, evidence, uncertainty, and blockers for: {objective.objective_text}",
                purpose="result",
                priority=0.65,
                dependencies=[previous],
                expected_outcomes=["Final result reflects actual work performed."],
                completion_criteria=["Result does not claim unperformed research or actions."],
            )
        )
        return tasks

    def _research_capabilities(self, lowered: str) -> list[str]:
        capabilities: list[str] = []
        if any(word in lowered for word in ("mac", "computer", "machine", "slow", "system")):
            capabilities.append("inspect_system")
        if any(word in lowered for word in ("research", "gather", "requirements", "price", "regulation", "weather", "incentive", "atlantic")):
            capabilities.append("perform_web_research")
        return capabilities

    def _subject(self, text: str) -> str:
        terms = [
            term
            for term in re.findall(r"\b[a-zA-Z][a-zA-Z0-9-]{3,}\b", text)
            if term.lower() not in {"research", "gather", "determine", "whether", "what", "should", "prepare", "requirements", "needed", "very", "become"}
        ]
        return " ".join(terms[:4]) if terms else "objective"

    def _slug(self, text: str) -> str:
        return "-".join(re.findall(r"[a-z0-9]+", text.lower()))[:48] or "task"
