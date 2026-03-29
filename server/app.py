"""FastAPI server — HTTP interface for OpenCesium (OpenEnv v1.0).

Endpoints
---------
POST /reset        Reset the environment and return initial Observation.
POST /step         Apply an Action; return (Observation, Reward, done, info).
POST /state        Return a serialisable snapshot of the current EnvState.
GET  /health       Health check (returns HTTP 200 + {"status": "ok"}).
"""

from fastapi import FastAPI
from server.routes import router

app = FastAPI(
    title="OpenCesium",
    description=(
        "Meta-agent workflow automation environment (OpenEnv v1.0). "
        "An outer LLM agent constructs and configures tool-using sub-agents "
        "to solve structured business workflows."
    ),
    version="1.0.0",
)

app.include_router(router)
