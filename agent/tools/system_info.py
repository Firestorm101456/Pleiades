from __future__ import annotations

import os
import platform
import plistlib
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

from agent.tools.base import OperationType, RiskLevel, Tool, ToolResult


@dataclass(frozen=True)
class ObservedValue:
    value: str | int | float | None
    status: str
    source: str


@dataclass(frozen=True)
class SystemInfo:
    os: ObservedValue
    os_version: ObservedValue
    architecture: ObservedValue
    cpu_model: ObservedValue
    cpu_cores: ObservedValue
    ram_gb: ObservedValue
    gpu_model: ObservedValue
    gpu_vram_gb: ObservedValue
    storage_available_gb: ObservedValue

    def facts_for_capabilities(self, capabilities: set[str]) -> list[str]:
        fields = {
            "operating_system": [("Operating system", self.os), ("OS version", self.os_version), ("Architecture", self.architecture)],
            "cpu": [("CPU model", self.cpu_model), ("CPU cores", self.cpu_cores)],
            "ram": [("System RAM", self.ram_gb)],
            "gpu": [("GPU model", self.gpu_model)],
            "gpu_vram": [("GPU memory/VRAM", self.gpu_vram_gb)],
            "storage": [("Available storage", self.storage_available_gb)],
            "hardware": [
                ("CPU model", self.cpu_model),
                ("CPU cores", self.cpu_cores),
                ("System RAM", self.ram_gb),
                ("GPU model", self.gpu_model),
                ("GPU memory/VRAM", self.gpu_vram_gb),
                ("Available storage", self.storage_available_gb),
            ],
        }
        facts: list[str] = []
        for capability in capabilities:
            for label, observed in fields.get(capability, []):
                if observed.status == "MEASURED" and observed.value is not None:
                    suffix = " GB" if label in {"System RAM", "GPU memory/VRAM", "Available storage"} else ""
                    facts.append(f"{label}: {observed.value}{suffix} ({observed.status}, source={observed.source})")
                else:
                    facts.append(f"{label}: UNKNOWN (source={observed.source})")
        return list(dict.fromkeys(facts))


class SystemInfoTool(Tool):
    name = "system_info"
    description = "Read-only inspection of local operating system and hardware information."
    input_schema = {"type": "object", "properties": {}}
    capabilities = {"hardware", "cpu", "gpu", "gpu_vram", "ram", "storage", "operating_system"}
    risk_level = RiskLevel.LOW
    operation_type = OperationType.READ

    def execute(self, arguments: dict[str, Any] | None = None, *, authorized: bool = False) -> ToolResult:
        info = collect_system_info()
        return ToolResult(True, "Collected read-only system information.", asdict(info))


def collect_system_info() -> SystemInfo:
    gpu_model, gpu_vram = _gpu_info()
    return SystemInfo(
        os=_measured(platform.system() or None, "platform"),
        os_version=_measured(platform.version() or platform.release() or None, "platform"),
        architecture=_measured(platform.machine() or None, "platform"),
        cpu_model=_measured(_cpu_model(), "sysctl/platform"),
        cpu_cores=_measured(os.cpu_count(), "os"),
        ram_gb=_measured(_ram_gb(), "sysctl/os"),
        gpu_model=_measured(gpu_model, "system_profiler"),
        gpu_vram_gb=_observed(gpu_vram, "system_profiler", measured=gpu_vram is not None),
        storage_available_gb=_measured(round(shutil.disk_usage("/").free / (1024**3), 2), "shutil"),
    )


def _measured(value: str | int | float | None, source: str) -> ObservedValue:
    return _observed(value, source, measured=True)


def _observed(value: str | int | float | None, source: str, measured: bool = False) -> ObservedValue:
    status = "MEASURED" if measured and value not in (None, "") else "UNKNOWN"
    return ObservedValue(value if value not in ("", None) else None, status, source)


def _cpu_model() -> str | None:
    if platform.system() == "Darwin":
        return _fixed_command(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor() or None
    return platform.processor() or platform.machine() or None


def _ram_gb() -> float | None:
    if platform.system() == "Darwin":
        raw = _fixed_command(["sysctl", "-n", "hw.memsize"])
        if raw and raw.isdigit():
            return round(int(raw) / (1024**3), 2)
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    return round((pages * page_size) / (1024**3), 2)


def _gpu_info() -> tuple[str | None, float | None]:
    if platform.system() != "Darwin":
        return None, None
    raw = _fixed_command(["system_profiler", "SPDisplaysDataType", "-xml"], timeout=8)
    if not raw:
        return None, None
    try:
        data = plistlib.loads(raw.encode("utf-8"))
    except plistlib.InvalidFileException:
        return None, None
    displays = data[0].get("_items", []) if data else []
    if not displays:
        return None, None
    names: list[str] = []
    vram_values: list[float] = []
    for display in displays:
        name = display.get("sppci_model") or display.get("sppci_device_type")
        if name:
            names.append(str(name))
        vram = _parse_vram_gb(str(display.get("spdisplays_vram", "")))
        if vram is not None:
            vram_values.append(vram)
    return ", ".join(names) if names else None, max(vram_values) if vram_values else None


def _parse_vram_gb(text: str) -> float | None:
    match = re.search(r"([\d.]+)\s*(GB|MB)", text, flags=re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).upper()
    return round(value if unit == "GB" else value / 1024, 2)


def _fixed_command(command: list[str], timeout: float = 3) -> str | None:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None
