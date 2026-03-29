"""OpenCesium tools package."""

from tools.base import BaseTool, ToolResult
from tools.stock import StockTool
from tools.search import SearchTool
from tools.calculator import CalculatorTool
from tools.email_tool import EmailTool

TOOL_REGISTRY: dict[str, BaseTool] = {
    "stock": StockTool(),
    "search": SearchTool(),
    "calculator": CalculatorTool(),
    "email": EmailTool(),
}

__all__ = [
    "BaseTool",
    "ToolResult",
    "StockTool",
    "SearchTool",
    "CalculatorTool",
    "EmailTool",
    "TOOL_REGISTRY",
]
