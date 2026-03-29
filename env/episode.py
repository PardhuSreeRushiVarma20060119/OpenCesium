"""Episode tracking utilities — history management and termination logic."""

from __future__ import annotations

from typing import Any

from env.models import EnvState, Action, Reward


def record_step(
    state: EnvState,
    action: Action,
    reward: Reward,
) -> None:
    """Append one (action, reward) pair to the episode history."""
    state.history.append(
        {
            "step": state.step_index,
            "action_type": action.action_type,
            "payload": action.payload,
            "reward": reward.total,
            "components": reward.components,
            "score": state.last_score,
        }
    )


def is_terminal(state: EnvState, passing_threshold: float) -> bool:
    """Return True when the episode should terminate.

    Termination conditions (spec §2.4):
        τ = min{t | t ≥ H  OR  (Evaluate ∈ a_t AND Φ(A_t, T) ≥ θ_pass)}
    """
    if state.step_index >= state.max_steps:
        return True
    return False  # Early-pass termination is handled in transitions.py


def build_observation_dict(state: EnvState) -> dict[str, Any]:
    """Collapse EnvState into the flat Observation schema."""
    return {
        "task_id": state.task_id,
        "task_description": state.task_description,
        "available_tools": state.available_tools,
        "agent_config": state.agent_config.model_dump(),
        "last_score": state.last_score,
        "last_reward": state.last_reward,
        "step_index": state.step_index,
        "max_steps": state.max_steps,
        "history": state.history,
        "tool_execution_log": state.tool_execution_log,
        "error_log": state.error_log,
        "done": state.done,
    }
