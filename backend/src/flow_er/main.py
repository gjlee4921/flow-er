"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from flow_er.api import (
    agent,
    datasets,
    executions,
    experiments,
    gates,
    jobs,
    memory,
    models,
    report,
    results,
)
from flow_er.config import settings

app = FastAPI(title="flow-er")


@app.on_event("startup")
def _resume() -> None:
    """Projects whose workflow runs on new files keep doing so after a restart."""
    from flow_er.workspace import executor
    executor.resume_watchers()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in (datasets.router, memory.router, executions.router, report.router, experiments.router, gates.router, results.router, agent.router, models.router,
          jobs.router):
    app.include_router(r)


@app.get("/health")
def health() -> dict[str, str]:
    """Up, and whether the agent has a model to talk to (never the key itself)."""
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "model": settings.openai_model if settings.llm_provider == "openai" else "",
        "llm_key": "loaded" if settings.openai_api_key else "missing",
    }
