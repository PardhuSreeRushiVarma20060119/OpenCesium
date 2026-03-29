"""Pydantic v2 data models for OpenCesium.

All models are JSON-serialisable and satisfy the OpenEnv v1.0 typed-model
requirement.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Agent configuration (inner worker agent state)
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel):
    """Configuration vector θ = (prompt, k_max, strategy) plus tool subset U_A."""

    tools: List[str] = Field(default_factory=list)
    prompt: str = ""
    strategy: Literal["react", "plan_execute", "direct"] = "react"
    k_max: int = 10


# ---------------------------------------------------------------------------
# OpenEnv v1.0 typed observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    task_id: str
    task_description: str
    available_tools: List[str]
    agent_config: Dict[str, Any]
    last_score: float = 0.0
    last_reward: float = 0.0
    step_index: int = 0
    max_steps: int
    history: List[Dict[str, Any]] = Field(default_factory=list)
    tool_execution_log: List[str] = Field(default_factory=list)
    error_log: List[str] = Field(default_factory=list)
    done: bool = False


# ---------------------------------------------------------------------------
# OpenEnv v1.0 typed action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    action_type: Literal[
        "ADD_TOOL",
        "REMOVE_TOOL",
        "SET_PROMPT",
        "SET_STRATEGY",
        "EVALUATE",
        "NOOP",
    ]
    payload: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# OpenEnv v1.0 typed reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    total: float
    components: Dict[str, float]


# ---------------------------------------------------------------------------
# Internal environment state (not directly exposed via the API)
# ---------------------------------------------------------------------------

class EnvState(BaseModel):
    """Serialisable snapshot of the full environment state."""

    task_id: str
    task_description: str
    available_tools: List[str]
    agent_config: AgentConfig = Field(default_factory=AgentConfig)

    last_score: float = 0.0
    last_reward: float = 0.0
    step_index: int = 0
    max_steps: int = 10

    history: List[Dict[str, Any]] = Field(default_factory=list)
    tool_execution_log: List[str] = Field(default_factory=list)
    error_log: List[str] = Field(default_factory=list)
    done: bool = False

    # For loop-penalty detection: last successfully dispatched action
    last_action_type: Optional[str] = None
    last_action_payload: Optional[Dict[str, Any]] = None

    # Latest worker-execution output (set after each EVALUATE)
    worker_output: Optional[Dict[str, Any]] = None
