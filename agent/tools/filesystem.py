from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.tools.base import OperationType, RiskLevel, Tool, ToolResult


class FilesystemReadTool(Tool):
    name = "filesystem_read"
    description = "Read a local text document for inspection."
    input_schema = {"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}}
    capabilities = {"read_files"}
    risk_level = RiskLevel.LOW
    operation_type = OperationType.READ

    def execute(self, arguments: dict[str, Any], *, authorized: bool = False) -> ToolResult:
        path = Path(str(arguments["path"])).expanduser()
        if not path.exists() or not path.is_file():
            return ToolResult(False, f"File not found: {path}")
        return ToolResult(True, path.read_text(encoding="utf-8", errors="replace"), {"path": str(path)})
