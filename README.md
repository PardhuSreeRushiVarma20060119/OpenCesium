# OpenCesium

**A Meta-Agent Workflow Automation Environment for Real-World Tool-Using Agent Workflow Automations**

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
