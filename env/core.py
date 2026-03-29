"""OpenCesiumEnv — main environment class implementing the OpenEnv v1.0 API."""

from __future__ import annotations

from typing import Any

from env.models import Action, AgentConfig, EnvState, Observation, Reward
from env.episode import build_observation_dict, record_step
from env.transitions import dispatch
from tasks.registry import TASK_REGISTRY
from tools import TOOL_REGISTRY
from tools.email_tool import clear_dev_inbox

_AVAILABLE_TOOLS = list(TOOL_REGISTRY.keys())


class OpenCesiumEnv:
    """OpenEnv v1.0 compliant environment.

    Usage
    -----
    Direct Python mode::

        env = OpenCesiumEnv(task_id="task_easy_stock_alert")
        obs = env.reset()
        obs, reward, done, info = env.step(action)

    HTTP mode (via FastAPI server): POST /reset, POST /step, POST /state.
    """

    def __init__(
        self,
        task_id: str = "task_easy_stock_alert",
        max_steps: int | None = None,
    ) -> None:
        if task_id not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task_id {task_id!r}. "
                f"Available: {list(TASK_REGISTRY.keys())}"
            )
        self._task_id = task_id
        self._max_steps_override = max_steps
        self._state: EnvState | None = None
        # Resolve task definition and grader once at construction
        self._task_def, self._grader = TASK_REGISTRY[task_id]

    # ------------------------------------------------------------------
    # OpenEnv v1.0 API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Return a clean initial observation.  Zero step count, empty config."""
        clear_dev_inbox()

        max_steps = (
            self._max_steps_override
            if self._max_steps_override is not None
            else self._task_def.max_steps
        )

        self._state = EnvState(
            task_id=self._task_id,
            task_description=self._task_def.description,
            available_tools=_AVAILABLE_TOOLS,
            agent_config=AgentConfig(k_max=max_steps),
            max_steps=max_steps,
        )

        return Observation(**build_observation_dict(self._state))

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """Apply action and return (obs, reward, done, info).

        Parameters
        ----------
        action : Action model (validated by Pydantic).

        Returns
        -------
        obs    : updated Observation
        reward : Reward with component breakdown
        done   : episode termination flag
        info   : auxiliary diagnostics dict
        """
        if self._state is None:
            raise RuntimeError("reset() must be called before step()")
        if self._state.done:
            raise RuntimeError("Episode is already done; call reset() to start a new episode")

        reward, done = dispatch(
            action=action,
            state=self._state,
            task_def=self._task_def,
            grader=self._grader,
            tool_registry=TOOL_REGISTRY,
        )

        record_step(self._state, action, reward)

        obs = Observation(**build_observation_dict(self._state))
        info: dict[str, Any] = {
            "task_id": self._task_id,
            "step_index": self._state.step_index,
            "score": self._state.last_score,
            "worker_output": self._state.worker_output,
        }
        return obs, reward, done, info

    def state(self) -> dict:
        """Return a serialisable snapshot of the current EnvState."""
        if self._state is None:
            return {}
        return self._state.model_dump()
