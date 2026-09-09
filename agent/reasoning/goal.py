from __future__ import annotations

import re
from hashlib import sha1

from agent.core.facts import Ambiguity, FactSource, ObjectiveModel, SemanticElement, SemanticRole, StructuredFact


class GoalInterpreter:
    def interpret(self, objective_text: str, observations: list[str] | None = None) -> ObjectiveModel:
        text = objective_text.strip()
        supplied = [StructuredFact(item, FactSource.USER_PROVIDED) for item in observations or []]
        supplied.extend(self._facts_from_objective(text))
        inferred = self._inferences(text, supplied)
        semantic_elements = self._semantic_elements(text, observations or [], supplied, inferred)
        return ObjectiveModel(
            objective_id=sha1(text.encode("utf-8")).hexdigest()[:12],
            objective_text=text,
            desired_outcome=self._desired_outcome(text),
            semantic_elements=semantic_elements,
            actors=self._values_for(semantic_elements, SemanticRole.ACTOR),
            actions=self._values_for(semantic_elements, SemanticRole.ACTION),
            objects=self._values_for(semantic_elements, SemanticRole.OBJECT),
            attributes=self._values_for(semantic_elements, SemanticRole.ATTRIBUTE),
            location_constraints=self._values_for(semantic_elements, SemanticRole.LOCATION),
            financial_constraints=self._values_for(semantic_elements, SemanticRole.DECISION_CRITERION)
            + self._values_for(semantic_elements, SemanticRole.RESOURCE),
            requested_outputs=self._values_for(semantic_elements, SemanticRole.REQUESTED_OUTPUT),
            unknown_variables=self._unknown_variables(text, observations or [], semantic_elements),
            entities=self._entities(text),
            constraints=self._constraints(text, supplied),
            explicit_facts=supplied,
            inferred_facts=inferred,
            time_constraints=self._time_constraints(text),
            deadlines=[fact.text for fact in inferred if "deadline" in fact.text.lower()],
            quantities=self._quantities(text, observations or []),
            resources=self._resources(text, observations or []),
            preferences=self._preferences(text),
            success_conditions=self._success_conditions(text),
            implicit_dependencies=self._dependencies(text),
            ambiguities=self._ambiguities(text, observations or []),
            unknowns=self._unknown_variables(text, observations or [], semantic_elements),
            assumptions=[],
            conflicting_requirements=self._conflicts(text),
        )

    def _semantic_elements(
        self,
        text: str,
        observations: list[str],
        facts: list[StructuredFact],
        inferred: list[StructuredFact],
    ) -> list[SemanticElement]:
        elements: list[SemanticElement] = []
        lowered = text.lower()
        source = [text] + observations
        if re.search(r"\b(new\s+)?captain\b", lowered):
            elements.append(SemanticElement(SemanticRole.ACTOR, "new captain" if "new captain" in lowered else "captain", FactSource.USER_PROVIDED, 0.9, source))
        if re.search(r"\b(for me|my|i\s+|i$)\b", lowered):
            elements.append(SemanticElement(SemanticRole.ACTOR, "user", FactSource.USER_PROVIDED, 0.8, source))
        actions = []
        if re.search(r"\b(research|gather|find out|determine)\b", lowered):
            actions.append("determine")
        if re.search(r"\b(decide|whether|choose)\b", lowered):
            actions.append("evaluate decision")
        if re.search(r"\b(prepare|plan)\b", lowered):
            actions.append("prepare")
        if re.search(r"\b(fix|resolve)\b", lowered):
            actions.append("change state")
        for action in dict.fromkeys(actions):
            elements.append(SemanticElement(SemanticRole.ACTION, action, FactSource.INFERRED, 0.75, source))
        object_patterns = [
            r"(buying an? [a-z ]+?)(?: makes| is|\.|$)",
            r"(maiden voyage)",
            r"((?:fishing\s+)?vessel\s*\([^)]+\)|(?:fishing\s+)?vessel)",
            r"(mac)",
            r"(farming)",
        ]
        for pattern in object_patterns:
            match = re.search(pattern, lowered)
            if match:
                elements.append(SemanticElement(SemanticRole.OBJECT, match.group(1).strip(), FactSource.USER_PROVIDED, 0.8, source))
        vessel_type = re.search(r"\(([^)]+)\)", text)
        if vessel_type:
            elements.append(SemanticElement(SemanticRole.ATTRIBUTE, f"type = {vessel_type.group(1)}", FactSource.USER_PROVIDED, 0.9, source))
        bought_type = re.search(r"\bbought\s+(?:a\s+|an\s+)?([A-Za-z-]+)", text, flags=re.IGNORECASE)
        if bought_type:
            elements.append(SemanticElement(SemanticRole.ATTRIBUTE, f"acquired object/type = {bought_type.group(1)}", FactSource.USER_PROVIDED, 0.75, source))
        for quantity in self._quantities(text, observations):
            role = SemanticRole.RESOURCE if quantity.startswith("$") else SemanticRole.QUANTITY
            elements.append(SemanticElement(role, quantity, FactSource.USER_PROVIDED, 0.9, source))
        for time_item in self._time_constraints(text):
            elements.append(SemanticElement(SemanticRole.TIME, time_item, FactSource.USER_PROVIDED, 0.85, source))
        location = re.search(r"\bin\s+the\s+([A-Z][A-Za-z ]+)\b", text)
        if location:
            elements.append(SemanticElement(SemanticRole.LOCATION, location.group(1).strip(), FactSource.USER_PROVIDED, 0.8, source))
        if "financial sense" in lowered or any(item.startswith("$") for item in self._quantities(text, observations)):
            elements.append(SemanticElement(SemanticRole.DECISION_CRITERION, "financial viability", FactSource.INFERRED, 0.75, source))
        if "requirements" in lowered:
            elements.append(SemanticElement(SemanticRole.REQUESTED_OUTPUT, "requirements list", FactSource.USER_PROVIDED, 0.85, source))
        if "name your vessel" in lowered:
            elements.append(SemanticElement(SemanticRole.REQUESTED_OUTPUT, "vessel name", FactSource.USER_PROVIDED, 0.9, source))
        for fact in facts:
            elements.append(SemanticElement(SemanticRole.CONSTRAINT, fact.text, fact.source, fact.confidence, fact.derived_from))
        for fact in inferred:
            elements.append(SemanticElement(SemanticRole.DEPENDENCY, fact.text, fact.source, fact.confidence, fact.derived_from))
        return self._dedupe_elements(elements)

    def _facts_from_objective(self, text: str) -> list[StructuredFact]:
        facts: list[StructuredFact] = []
        for match in re.finditer(r"\b(?:I have|my|we have|deadline is|budget is)\b[^.。;]*", text, flags=re.IGNORECASE):
            facts.append(StructuredFact(match.group(0).strip(), FactSource.USER_PROVIDED))
        return facts

    def _inferences(self, text: str, facts: list[StructuredFact]) -> list[StructuredFact]:
        inferred: list[StructuredFact] = []
        match = re.search(r"\b(?:is\s+)?(in|within)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve)\s+(day|days|week|weeks|month|months)\b", text, flags=re.IGNORECASE)
        if match:
            amount = self._number_text(match.group(2))
            inferred.append(
                StructuredFact(
                    f"Preparation deadline is approximately {amount} {match.group(3)} from now.",
                    FactSource.INFERRED,
                    confidence=0.8,
                    derived_from=[fact.text for fact in facts] or [text],
                )
            )
        if any(word in text.lower() for word in ("fix", "prepare", "create", "modify", "install")):
            inferred.append(
                StructuredFact(
                    "Objective may require actuator capabilities and authorization before consequential actions.",
                    FactSource.INFERRED,
                    confidence=0.75,
                    derived_from=[text],
                )
            )
        return inferred

    def _number_text(self, value: str) -> str:
        numbers = {
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "twelve": "12",
        }
        return numbers.get(value.lower(), value)

    def _desired_outcome(self, text: str) -> str:
        lowered = text.lower()
        if any(word in lowered for word in ("decide", "choose", "whether")):
            return "decision"
        if any(word in lowered for word in ("prepare", "plan")):
            return "plan"
        if any(word in lowered for word in ("find out why", "diagnose", "fix")):
            return "diagnosis_and_resolution"
        if any(word in lowered for word in ("research", "gather information")):
            return "research_synthesis"
        return "answer_or_next_action"

    def _entities(self, text: str) -> list[str]:
        entities = re.findall(r"\b[A-Z][A-Za-z0-9_$-]*(?:\s+[A-Z][A-Za-z0-9_$-]*)*\b", text)
        return list(dict.fromkeys(item for item in entities if item.lower() not in {"i"}))

    def _constraints(self, text: str, facts: list[StructuredFact]) -> list[str]:
        constraints = [fact.text for fact in facts if any(word in fact.text.lower() for word in ("must", "cannot", "deadline", "budget", "have"))]
        return list(dict.fromkeys(constraints))

    def _time_constraints(self, text: str) -> list[str]:
        patterns = re.findall(
            r"\b(?:today|tomorrow|in \d+ \w+|within \d+ \w+|\d+ weeks? from now|for about \d+ weeks?|(?:one|two|three|four|five|six|seven|eight|nine|ten|twelve)-week|before \w+|by \w+)\b",
            text,
            flags=re.IGNORECASE,
        )
        return list(dict.fromkeys(patterns))

    def _quantities(self, text: str, observations: list[str]) -> list[str]:
        joined = " ".join([text] + observations)
        return list(dict.fromkeys(re.findall(r"\$?\b\d[\d,]*(?:\.\d+)?\s*(?:days?|weeks?|months?|years?|gb|tb|%)?\b", joined, flags=re.IGNORECASE)))

    def _resources(self, text: str, observations: list[str]) -> list[str]:
        joined = " ".join([text] + observations).lower()
        resources = []
        if "$" in joined or "budget" in joined:
            resources.append("money")
        if any(word in joined for word in ("computer", "mac", "laptop", "machine")):
            resources.append("local computer")
        if any(word in joined for word in ("file", "document", "dataset")):
            resources.append("user-provided data")
        return resources

    def _preferences(self, text: str) -> list[str]:
        return re.findall(r"\b(?:prefer|want|avoid|safest|best|local)\b[^.。;]*", text, flags=re.IGNORECASE)

    def _success_conditions(self, text: str) -> list[str]:
        lowered = text.lower()
        conditions = []
        if "research" in lowered or "gather information" in lowered:
            conditions.append("Relevant evidence has been gathered and synthesized.")
        if any(word in lowered for word in ("decide", "choose", "whether")):
            conditions.append("A recommendation is supported by explicit evidence and uncertainty labels.")
        if any(word in lowered for word in ("fix", "prepare", "create")):
            conditions.append("Required action is completed or safely blocked with a specific reason.")
        return conditions or ["The response satisfies the requested outcome."]

    def _dependencies(self, text: str) -> list[str]:
        lowered = text.lower()
        deps = []
        if any(word in lowered for word in ("decide", "choose", "whether", "viable")):
            deps.append("compare options or scenarios before recommending")
        if any(word in lowered for word in ("fix", "modify", "install")):
            deps.append("verify safe authorization before consequential action")
        if any(word in lowered for word in ("research", "find out", "diagnose")):
            deps.append("collect evidence before conclusion")
        return deps

    def _ambiguities(self, text: str, observations: list[str]) -> list[Ambiguity]:
        joined = " ".join([text] + observations)
        ambiguities: list[Ambiguity] = []
        if re.search(r"\bincome\s*(?:=|is)?\s*\$?\d", joined, flags=re.IGNORECASE):
            ambiguities.append(
                Ambiguity(
                    "income amount",
                    ["gross revenue", "profit", "available capital", "personal income"],
                    "HIGH",
                    "I need to know what the income figure represents because it changes viability and risk calculations.",
                )
            )
        if re.search(r"\bbest\b", joined, flags=re.IGNORECASE):
            ambiguities.append(
                Ambiguity(
                    "best",
                    ["lowest cost", "highest quality", "lowest risk", "fastest"],
                    "MEDIUM",
                    "I need to know how to rank 'best' if the tradeoff materially changes the recommendation.",
                )
            )
        return ambiguities

    def _unknown_variables(self, text: str, observations: list[str], elements: list[SemanticElement]) -> list[str]:
        lowered = text.lower()
        known = " ".join([text] + observations + [element.value for element in elements]).lower()
        unknowns: list[str] = []
        objects = self._values_for(elements, SemanticRole.OBJECT)
        if "financial sense" in lowered or "financial" in lowered:
            if any("car" in item or "vehicle" in item for item in objects) and "mileage" not in known:
                unknowns.append("annual_mileage")
            if "ownership" not in known:
                unknowns.append("ownership_period")
            if "electricity" not in known:
                unknowns.append("electricity_cost")
        if any(word in lowered for word in ("fix", "diagnose", "find out why", "causing", "cause")):
            unknowns.append("root_cause")
        return unknowns

    def _values_for(self, elements: list[SemanticElement], role: SemanticRole) -> list[str]:
        return list(dict.fromkeys(element.value for element in elements if element.role == role))

    def _dedupe_elements(self, elements: list[SemanticElement]) -> list[SemanticElement]:
        seen: set[tuple[str, str]] = set()
        output: list[SemanticElement] = []
        for element in elements:
            key = (element.role.value, element.value.lower())
            if key in seen:
                continue
            seen.add(key)
            output.append(element)
        return output

    def _conflicts(self, text: str) -> list[str]:
        lowered = text.lower()
        if "fast" in lowered and "safest" in lowered:
            return ["Speed and safety may conflict."]
        return []
