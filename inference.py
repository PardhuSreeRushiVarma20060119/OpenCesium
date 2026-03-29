"""inference.py — Baseline meta-agent loop for OpenCesium.

Usage
-----
    python inference.py

Environment variables (required)
---------------------------------
    API_BASE_URL  : OpenAI-compatible API base URL
    MODEL_NAME    : Model identifier, e.g. "gpt-4o-mini"
    HF_TOKEN      : API key / Hugging Face token

Constraints (spec §8.1)
-----------------------
    - Uses OpenAI API client exclusively.
    - All credentials read from environment variables.
    - Named inference.py at project root.
    - Completes all three tasks within 20 minutes on 2-vCPU / 8 GB.
    - Deterministic outputs (temperature=0.0, seed=42).
"""

import json
import os
import time

from openai import OpenAI

from env import Action, OpenCesiumEnv

# ---------------------------------------------------------------------------
# Credentials (injected via environment)
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ---------------------------------------------------------------------------
# System prompt for the meta-agent
# ---------------------------------------------------------------------------
SYSTEM = """You are an AI workflow engineer.
Given a task description and available tools, issue actions to configure
and run a worker agent. Respond ONLY with valid JSON matching the Action schema:

{"action_type": "ADD_TOOL|REMOVE_TOOL|SET_PROMPT|SET_STRATEGY|EVALUATE|NOOP",
 "payload": {<optional key-value params>}}

Action semantics:
  ADD_TOOL      : {"tool": "<name>"}  — add a tool to the agent
  REMOVE_TOOL   : {"tool": "<name>"}  — remove a tool from the agent
  SET_PROMPT    : {"prompt": "<str>"} — set the agent system prompt
  SET_STRATEGY  : {"strategy": "react|plan_execute|direct"}
  EVALUATE      : {}                  — run the agent and score it
  NOOP          : {}                  — no operation

Strategy:
  1. Read the task description carefully.
  2. Add ALL required tools before calling EVALUATE.
  3. Call EVALUATE once the agent is fully configured.
  4. Do NOT repeat the same action consecutively.

Do not output any prose, only the JSON object."""


def run_task(task_id: str, max_steps: int = 20) -> dict:
    """Run a single task with the baseline meta-agent loop.

    Returns a summary dict with task_id, steps, cumulative_reward, final_score.
    """
    env = OpenCesiumEnv(task_id=task_id, max_steps=max_steps)
    obs = env.reset()

    msgs: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": obs.model_dump_json()},
    ]

    total_reward = 0.0
    final_score = 0.0

    for step in range(max_steps):
        # Query the meta-agent
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=msgs,
            temperature=0.0,
            max_tokens=256,
            seed=42,
        )
        raw = resp.choices[0].message.content.strip()

        # Parse action — fall back to NOOP on parse error
        try:
            action = Action(**json.loads(raw))
        except Exception:
            action = Action(action_type="NOOP", payload={})

        obs, reward, done, info = env.step(action)
        total_reward += reward.total
        final_score = obs.last_score

        msgs += [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": obs.model_dump_json()},
        ]

        if done:
            break

    return {
        "task_id": task_id,
        "steps": step + 1,
        "cumulative_reward": round(total_reward, 4),
        "final_score": round(final_score, 4),
    }


if __name__ == "__main__":
    tasks = [
        ("task_easy_stock_alert", 10),
        ("task_medium_gdp_growth", 15),
        ("task_hard_structured_report", 20),
    ]

    overall_start = time.time()
    results = []

    for tid, ms in tasks:
        t0 = time.time()
        r = run_task(tid, ms)
        elapsed = time.time() - t0
        print(
            f"[{tid}] score={r['final_score']:.4f} "
            f"cumR={r['cumulative_reward']:.4f} "
            f"steps={r['steps']} "
            f"time={elapsed:.1f}s"
        )
        results.append(r)

    total_time = time.time() - overall_start
    print(f"\nTotal runtime: {total_time:.1f}s")
    print(
        f"Average score: "
        f"{sum(r['final_score'] for r in results) / len(results):.4f}"
    )
