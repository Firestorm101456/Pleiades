from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass

from agent.capabilities.inventory import CapabilityInventory
from agent.capabilities.models import CapabilityRecord, CapabilityType, InventoryState, ResourceRecord, RiskClassification, utc_now


@dataclass(frozen=True)
class DiscoveryResult:
    resources: list[ResourceRecord]
    capabilities: list[CapabilityRecord]


class EnvironmentDiscovery:
    def __init__(self, inventory: CapabilityInventory) -> None:
        self.inventory = inventory

    def discover(self) -> DiscoveryResult:
        resources: list[ResourceRecord] = []
        capabilities: list[CapabilityRecord] = []
        for resource, caps in [
            self._executable("python", sys.executable, ["run_python", "parse_json", "calculate"]),
            self._path_executable("system_profiler", ["inspect_system"]),
            self._path_executable("sqlite3", ["data_storage"]),
        ]:
            resources.append(resource)
            capabilities.extend(caps)
        json_resource, json_caps = self._python_module("json", ["parse_json"])
        resources.append(json_resource)
        capabilities.extend(json_caps)
        for resource in resources:
            self.inventory.upsert_resource(resource)
        for capability in capabilities:
            self.inventory.upsert_capability(capability)
        return DiscoveryResult(resources, capabilities)

    def verify_known_safe(self) -> DiscoveryResult:
        result = self.discover()
        verified_resources: list[ResourceRecord] = []
        verified_capabilities: list[CapabilityRecord] = []
        for resource in result.resources:
            if resource.state == InventoryState.DISCOVERED and resource.location and resource.type == "executable":
                version = self._safe_version(resource.location)
                if version is not None:
                    resource.state = InventoryState.AVAILABLE
                    resource.version = version[:200]
                    resource.last_verified_at = utc_now()
                    resource.last_verified_status = "version_check_succeeded"
                    resource.confidence = 0.95
                    resource.add_evidence("VERIFIED_BY", "Executable responded to a safe version query.", "environment_discovery")
                    verified_resources.append(resource)
                    self.inventory.upsert_resource(resource)
            elif resource.state == InventoryState.DISCOVERED and resource.type == "python_module":
                resource.state = InventoryState.AVAILABLE
                resource.last_verified_at = utc_now()
                resource.last_verified_status = "module_import_spec_found"
                resource.confidence = 0.9
                resource.add_evidence("VERIFIED_BY", "Python import spec found.", "environment_discovery")
                verified_resources.append(resource)
                self.inventory.upsert_resource(resource)
        available_resource_ids = {resource.id for resource in verified_resources}
        for capability in result.capabilities:
            if set(capability.resources) & available_resource_ids:
                capability.state = InventoryState.AVAILABLE
                capability.last_verified_at = utc_now()
                capability.last_verified_status = "provider_verified"
                capability.confidence = 0.85
                capability.add_evidence("VERIFIED_BY", "At least one provider resource was verified.", "environment_discovery")
                verified_capabilities.append(capability)
                self.inventory.upsert_capability(capability)
        return DiscoveryResult(verified_resources, verified_capabilities)

    def _executable(self, name: str, location: str, capability_ids: list[str]) -> tuple[ResourceRecord, list[CapabilityRecord]]:
        resource = ResourceRecord(
            id=f"executable:{name}",
            name=name,
            type="executable",
            state=InventoryState.DISCOVERED,
            location=location,
            discovery_source="python_runtime",
            description=f"Executable discovered at {location}.",
            confidence=0.55,
        )
        resource.add_evidence("DISCOVERED_FROM", f"Executable path: {location}", "environment_discovery")
        return resource, [self._capability(capability_id, resource.id) for capability_id in capability_ids]

    def _path_executable(self, name: str, capability_ids: list[str]) -> tuple[ResourceRecord, list[CapabilityRecord]]:
        location = shutil.which(name)
        state = InventoryState.DISCOVERED if location else InventoryState.KNOWN_BUT_NOT_ACQUIRED
        resource = ResourceRecord(
            id=f"executable:{name}",
            name=name,
            type="executable",
            state=state,
            location=location,
            discovery_source="PATH",
            description=f"Executable {name}.",
            confidence=0.55 if location else 0.2,
        )
        detail = f"Executable found at {location}" if location else "Executable is known, but not found on PATH."
        resource.add_evidence("DISCOVERED_FROM" if location else "KNOWN_FROM", detail, "environment_discovery")
        return resource, [self._capability(capability_id, resource.id, state) for capability_id in capability_ids]

    def _python_module(self, name: str, capability_ids: list[str]) -> tuple[ResourceRecord, list[CapabilityRecord]]:
        spec = importlib.util.find_spec(name)
        state = InventoryState.DISCOVERED if spec else InventoryState.KNOWN_BUT_NOT_ACQUIRED
        resource = ResourceRecord(
            id=f"python_module:{name}",
            name=name,
            type="python_module",
            state=state,
            location=getattr(spec, "origin", None) if spec else None,
            discovery_source="python_environment",
            description=f"Python module {name}.",
            confidence=0.55 if spec else 0.2,
        )
        resource.add_evidence("DISCOVERED_FROM" if spec else "KNOWN_FROM", "Python import spec inspection.", "environment_discovery")
        return resource, [self._capability(capability_id, resource.id, state) for capability_id in capability_ids]

    def _capability(self, capability_id: str, resource_id: str, state: InventoryState = InventoryState.DISCOVERED) -> CapabilityRecord:
        types = {
            "inspect_system": CapabilityType.SENSOR,
            "run_python": CapabilityType.ACTUATOR,
            "parse_json": CapabilityType.ANALYSIS,
            "calculate": CapabilityType.ANALYSIS,
            "data_storage": CapabilityType.RESOURCE,
        }
        risk = RiskClassification.HIGH if capability_id == "run_python" else RiskClassification.LOW
        cap = CapabilityRecord(
            id=capability_id,
            name=capability_id.replace("_", " ").title(),
            description=f"General capability: {capability_id}.",
            type=types.get(capability_id, CapabilityType.RESOURCE),
            state=state,
            resources=[resource_id],
            risk_level=risk,
            authorization_required=risk != RiskClassification.LOW,
            reversibility="depends_on_action" if risk != RiskClassification.LOW else "read_only",
            verification_method="provider resource verification",
            confidence=0.45,
        )
        cap.add_evidence("LEARNED_FROM", f"Mapped from discovered provider {resource_id}.", "environment_discovery")
        return cap

    def _safe_version(self, executable: str) -> str | None:
        for arg in ("--version", "-V", "-h"):
            try:
                completed = subprocess.run([executable, arg], check=False, capture_output=True, text=True, timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                continue
            output = (completed.stdout or completed.stderr).strip()
            if output:
                return output.splitlines()[0]
        return None

