"""CalculatorTool — deterministic arithmetic via Python's math library."""

import math
import time

from tools.base import BaseTool, ToolResult

# Safe names available inside eval expressions
_SAFE_GLOBALS: dict = {
    "__builtins__": {},
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    **{name: getattr(math, name) for name in dir(math) if not name.startswith("_")},
}


class CalculatorTool(BaseTool):
    name = "calculator"
    description = (
        "Evaluate a mathematical expression or perform a named computation "
        "(e.g. moving_average, percentage_change). Returns a numeric result."
    )
    input_schema = {
        "expression": "str — Python-compatible math expression, e.g. '(25000 - 24000) / 24000 * 100'",
        "operation": "str — named op: 'moving_average' | 'percentage_change' | 'expression' (default)",
        "values": "list[float] — input values for named operations",
    }
    output_schema = {
        "result": "float — computed numeric result",
    }
    cost: float = 0.02

    def run(self, params: dict) -> ToolResult:
        t0 = time.time()
        operation: str = str(params.get("operation", "expression")).lower()

        try:
            if operation == "moving_average":
                values = [float(v) for v in params.get("values", [])]
                if not values:
                    raise ValueError("values list is empty for moving_average")
                result = sum(values) / len(values)

            elif operation == "percentage_change":
                values = [float(v) for v in params.get("values", [])]
                if len(values) < 2:
                    raise ValueError("percentage_change requires at least 2 values")
                prev, curr = values[0], values[1]
                if prev == 0:
                    raise ValueError("Division by zero: previous value is 0")
                result = (curr - prev) / prev * 100.0

            else:
                # Default: evaluate a math expression
                expression: str = str(params.get("expression", ""))
                if not expression:
                    raise ValueError("expression parameter is required")
                result = float(eval(expression, _SAFE_GLOBALS, {}))  # noqa: S307

            return ToolResult(
                success=True,
                output=round(result, 6),
                latency_ms=(time.time() - t0) * 1000,
            )

        except Exception as exc:
            return ToolResult(
                success=False,
                output=None,
                error=str(exc),
                latency_ms=(time.time() - t0) * 1000,
            )
