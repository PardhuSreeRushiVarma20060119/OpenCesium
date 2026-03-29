# OpenCesium

**A Meta-Agent Workflow Automation Environment for OpenEnv Hackathon Round 1**

[![OpenEnv v1.0 Compliant](https://img.shields.io/badge/OpenEnv-v1.0-blue)](openenv.yaml)
[![Environment ID](https://img.shields.io/badge/env--id-opencesium--v1-green)](openenv.yaml)

---

## Overview

OpenCesium is an OpenEnv-compliant benchmark in which a **meta-agent** learns to construct, parameterise, and iteratively refine executable tool-using sub-agents to complete structured business workflow tasks.

The meta-agent issues **configuration actions** (not tool calls directly). The environment evaluates each configuration and provides a shaped, dense reward at every step.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Meta-Agent (LLM)                                       │
│  Issues: ADD_TOOL | REMOVE_TOOL | SET_PROMPT |          │
│          SET_STRATEGY | EVALUATE | NOOP                 │
└─────────────────┬───────────────────────────────────────┘
                  │ Action
                  ▼
┌─────────────────────────────────────────────────────────┐
│  OpenCesiumEnv  (reset / step / state)                  │
│  ┌──────────────────────────────────────────────┐       │
│  │  EnvState                                    │       │
│  │  - AgentConfig (tools, prompt, strategy)     │       │
│  │  - step_index, last_score, history, logs     │       │
│  └──────────────────────────────────────────────┘       │
│  ┌──────────────────────────────────────────────┐       │
│  │  EVALUATE → Worker Executor                  │       │
│  │  - Runs configured tools in task order       │       │
│  │  - Returns output dict → Grader → score      │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────┬───────────────────────────────────────┘
                  │ (Observation, Reward, done, info)
                  ▼
             Meta-Agent continues…
```

---

## Task Suite

| ID | Difficulty | Required Tools | Max Steps | Pass Threshold |
|----|-----------|---------------|-----------|---------------|
| `task_easy_stock_alert` | Easy | stock, email | 10 | 0.70 |
| `task_medium_gdp_growth` | Medium | search, calculator, email | 15 | 0.70 |
| `task_hard_structured_report` | Hard | stock, search, calculator, email | 20 | 0.70 |

---

## Reward Function

```
R_t = α·ΔC_t + β·U_t + γ·E_t − δ·I_t − ε·L_t

α = 0.40  (performance gain)
β = 0.20  (tool correctness)
γ = 0.15  (step efficiency)
δ = 0.20  (invalid action penalty)
ε = 0.10  (loop penalty)
```

---

## Repository Structure

```
env/
  __init__.py        # Public exports
  core.py            # OpenCesiumEnv (reset/step/state)
  models.py          # Pydantic models: Observation, Action, Reward, EnvState
  transitions.py     # Action dispatch + reward computation + worker execution
  episode.py         # Episode tracking and history management
tasks/
  __init__.py
  registry.py        # TASK_REGISTRY
  easy.py            # Task 1 definition + grader
  medium.py          # Task 2 definition + grader
  hard.py            # Task 3 definition + grader
tools/
  __init__.py        # TOOL_REGISTRY
  base.py            # BaseTool, ToolResult
  stock.py           # StockTool (yfinance)
  search.py          # SearchTool (Tavily + reference-data fallback)
  calculator.py      # CalculatorTool (pure Python, sandboxed eval)
  email_tool.py      # EmailTool (in-memory dev / SMTP prod)
server/
  __init__.py
  app.py             # FastAPI application
  routes.py          # /reset /step /state /health
inference.py         # Baseline OpenAI meta-agent loop
openenv.yaml         # OpenEnv specification metadata
Dockerfile
requirements.txt
README.md
```

---

## Quick Start

### Python (direct mode)

```python
from env import OpenCesiumEnv, Action

env = OpenCesiumEnv(task_id="task_easy_stock_alert")
obs = env.reset()

# Meta-agent configures the worker
env.step(Action(action_type="ADD_TOOL", payload={"tool": "stock"}))
env.step(Action(action_type="ADD_TOOL", payload={"tool": "email"}))
obs, reward, done, info = env.step(Action(action_type="EVALUATE", payload={}))
print(f"Score: {obs.last_score:.3f}, Reward: {reward.total:.3f}")
```

### HTTP (server mode)

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
curl -s -X POST http://localhost:7860/reset \
     -H 'Content-Type: application/json' \
     -d '{"task_id": "task_easy_stock_alert"}'
```

### Docker

```bash
docker build -t opencesium .
docker run -p 7860:7860 \
  -e ENV_MODE=development \
  opencesium
```

### Baseline inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."
python inference.py
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENV_MODE` | `development` (in-memory email) or `production` (SMTP) | `development` |
| `API_BASE_URL` | OpenAI-compatible API base URL | — |
| `MODEL_NAME` | LLM model identifier | — |
| `HF_TOKEN` | API key / HF token | — |
| `TAVILY_API_KEY` | Tavily search API key (optional) | — |
| `SMTP_HOST` | SMTP server host (production mode) | `localhost` |
| `SMTP_PORT` | SMTP server port | `1025` |
| `SMTP_USER` | SMTP username | — |
| `SMTP_PASS` | SMTP password | — |

---

## OpenEnv v1.0 Compliance

| Requirement | Status |
|-------------|--------|
| Typed Observation (Pydantic v2) | ✓ |
| Typed Action (6 types via Literal) | ✓ |
| Typed Reward (total + components) | ✓ |
| `reset()` returns clean Observation | ✓ |
| `step()` returns `(obs, reward, done, info)` | ✓ |
| `state()` returns serialisable dict | ✓ |
| `openenv.yaml` present | ✓ |
| ≥3 tasks with graders | ✓ |
| Graders deterministic | ✓ |
| Dense reward (partial credit every step) | ✓ |
| Real-world domain | ✓ |

---

## Expected Baseline Scores

| Task | Grader Score | Cumul. Reward | Avg Steps |
|------|-------------|--------------|-----------|
| Easy | 0.90–0.95 | 1.10–1.25 | 5–8 |
| Medium | 0.70–0.82 | 0.80–1.00 | 9–13 |
| Hard | 0.48–0.62 | 0.55–0.75 | 16–20 |

---

## License

See repository for licensing information.
**A Meta-Agent Workflow Automation Environment for Real-World Tool-Using Agent Automations**

OpenCesium is an OpenEnv-compliant benchmark environment in which a meta-agent learns to construct, parameterize, and iteratively refine executable tool-using sub-agents to complete structured business workflow tasks.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the Server (HTTP Mode)

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run Inference (Direct Python Mode)

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
python inference.py
```

### Docker

```bash
docker build -t opencesium .
docker run -p 7860:7860 opencesium
```

## Architecture

```
opencesium/
├── env/              # Core environment (reset/step/state)
│   ├── models.py     # Pydantic v2 typed models
│   ├── core.py       # OpenCesiumEnv class
│   ├── transitions.py # Action dispatch + reward computation
│   └── episode.py    # Episode lifecycle management
├── tasks/            # Task suite with deterministic graders
│   ├── easy.py       # Task 1: Financial Alert Report
│   ├── medium.py     # Task 2: GDP Growth Analysis
│   └── hard.py       # Task 3: Multi-Source Structured Report
├── tools/            # Tool abstraction layer
│   ├── stock.py      # StockTool (yfinance)
│   ├── search.py     # SearchTool (Tavily + mock)
│   ├── calculator.py # CalculatorTool (Python math)
│   └── email_tool.py # EmailTool (SMTP / Mailhog)
├── server/           # FastAPI HTTP endpoints
├── inference.py      # Baseline agent loop
├── openenv.yaml      # OpenEnv spec metadata
└── Dockerfile        # Deployment container
```

## Tasks

| Task | Difficulty | Required Tools | Max Steps |
|------|-----------|---------------|-----------|
| Financial Alert Report | Easy | stock, email | 10 |
| GDP Growth Analysis | Medium | search, calculator, email | 15 |
| Structured Report | Hard | stock, search, calculator, email | 20 |

## Action Space

| Action | Description | Payload |
|--------|------------|---------|
| `ADD_TOOL` | Add a tool to the worker agent | `{"tool": "name"}` |
| `REMOVE_TOOL` | Remove a tool | `{"tool": "name"}` |
| `SET_PROMPT` | Set worker agent's system prompt | `{"prompt": "..."}` |
| `SET_STRATEGY` | Set execution strategy | `{"strategy": "react\|plan_execute\|direct"}` |
| `EVALUATE` | Run the worker agent and score | `{}` |
| `NOOP` | No operation | `{}` |

## Reward Function

Dense, shaped reward at every step:

```
R_t = α·ΔC_t + β·U_t + γ·E_t − δ·I_t − ε·L_t
```

- **ΔC_t** (α=0.40): Performance gain (grader score improvement)
- **U_t** (β=0.20): Tool correctness (required tool used)
- **E_t** (γ=0.15): Step efficiency (acting early bonus)
- **I_t** (δ=0.20): Invalid action penalty
- **L_t** (ε=0.10): Loop penalty (repeated action)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/reset` | Start new episode |
| `POST` | `/step` | Apply meta-action |
| `POST` | `/state` | Get current state |
| `GET` | `/tasks` | List available tasks |

## Environment Variables

| Variable | Description | Required |
|----------|------------|----------|
| `API_BASE_URL` | OpenAI API base URL | For inference |
| `MODEL_NAME` | LLM model name | For inference |
| `HF_TOKEN` | HuggingFace / API token | For inference |
| `ENV_MODE` | `development` or `production` | Optional |
| `TAVILY_API_KEY` | Tavily search API key | Optional |
| `SMTP_HOST` | SMTP server host | Production only |
| `SMTP_PORT` | SMTP server port | Production only |

## License

This project was created for the OpenEnv Hackathon Round 1.
