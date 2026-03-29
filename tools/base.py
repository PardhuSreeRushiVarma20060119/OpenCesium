"""BaseTool abstract interface and ToolResult model."""

from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any


class ToolResult(BaseModel):
    success: bool
    output: Any
    error: str = ""
    latency_ms: float = 0.0


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    input_schema: dict = {}
    output_schema: dict = {}
    cost: float = 0.0  # Used in reward cost model

    @abstractmethod
    def run(self, params: dict) -> ToolResult:
        """Execute the tool with given parameters and return a ToolResult."""
        ...
