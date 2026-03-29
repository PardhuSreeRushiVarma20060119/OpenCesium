"""FastAPI route definitions for OpenCesium HTTP mode."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from env.core import OpenCesiumEnv
from env.models import Action, Observation, Reward
from tasks.registry import TASK_REGISTRY

router = APIRouter()

# ---------------------------------------------------------------------------
# Per-task environment singletons (one per task_id).
# HTTP clients are expected to drive a single task at a time.
# ---------------------------------------------------------------------------
_envs: dict[str, OpenCesiumEnv] = {
    task_id: OpenCesiumEnv(task_id=task_id) for task_id in TASK_REGISTRY
}


# ---------------------------------------------------------------------------
# Request / response bodies
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task_easy_stock_alert"


class StepRequest(BaseModel):
    task_id: str = "task_easy_stock_alert"
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any]


class StateRequest(BaseModel):
    task_id: str = "task_easy_stock_alert"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/health")
def health() -> dict:
    """Health check endpoint required by Hugging Face Spaces."""
    return {"status": "ok", "environment": "opencesium-v1"}


@router.post("/reset", response_model=Observation)
def reset(body: ResetRequest) -> Observation:
    """Reset the environment for the specified task."""
    if body.task_id not in _envs:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {body.task_id!r}")
    return _envs[body.task_id].reset()


@router.post("/step", response_model=StepResponse)
def step(body: StepRequest) -> StepResponse:
    """Apply an action to the specified task environment."""
    if body.task_id not in _envs:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {body.task_id!r}")
    env = _envs[body.task_id]
    try:
        obs, reward, done, info = env.step(body.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@router.post("/state")
def state(body: StateRequest) -> dict:
    """Return a serialisable snapshot of the current environment state."""
    if body.task_id not in _envs:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {body.task_id!r}")
    return _envs[body.task_id].state()
