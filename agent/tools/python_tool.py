from __future__ import annotations

import subprocess
import sys
from typing import Any

from agent.tools.base import OperationType, RiskLevel, Tool, ToolResult


class PythonExecutionTool(Tool):
    name = "python_execute"
    description = "Execute Python code in a subprocess after explicit authorization."
    input_schema = {"type": "object", "required": ["code"], "properties": {"code": {"type": "string"}}}
    capabilities = {"run_python", "calculate"}
    risk_level = RiskLevel.HIGH
    operation_type = OperationType.EXECUTE

    def execute(self, arguments: dict[str, Any], *, authorized: bool = False) -> ToolResult:
        self.require_authorization(authorized)
        completed = subprocess.run(
            [sys.executable, "-c", str(arguments["code"])],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = completed.stdout + completed.stderr
        return ToolResult(completed.returncode == 0, output, {"returncode": completed.returncode})
