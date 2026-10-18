"""Running workflows: start an execution (all of it, up to a step, one step, or from a step on), follow
it live, read past runs, look at what a step made, and turn the new-files trigger on or off."""
import json
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from flow_er.workspace import executor, lab, nodes, store

router = APIRouter(tags=["workflows"])


class RunIn(BaseModel):
    mode: str = "all"
    node: str | None = None


class ActiveIn(BaseModel):
    on: bool


def _exp(exp_id: str) -> dict:
    try:
        return store.meta(exp_id)
    except KeyError:
        raise HTTPException(404, f"no experiment {exp_id}")


@router.get("/node-types")
def node_types() -> dict:
    """The kinds of step a workflow is built from, and the sources it starts from."""
    return {"types": nodes.NODE_TYPES, "sources": nodes.SOURCES}


@router.post("/experiments/{exp_id}/executions")
def run(exp_id: str, body: RunIn) -> dict:
    _exp(exp_id)
    try:
        return executor.start(exp_id, body.mode, body.node, trigger="manual")
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/experiments/{exp_id}/executions")
def runs(exp_id: str) -> list[dict]:
    _exp(exp_id)
    return executor.history(exp_id)


@router.get("/experiments/{exp_id}/executions/{eid}")
def one(exp_id: str, eid: str) -> dict:
    _exp(exp_id)
    try:
        return executor.get(exp_id, eid)
    except KeyError:
        raise HTTPException(404, f"no execution {eid}")


@router.get("/experiments/{exp_id}/executions/{eid}/events")
def events(exp_id: str, eid: str) -> StreamingResponse:
    """Server-sent events: each step as it starts and finishes, then the end."""
    _exp(exp_id)

    def stream() -> Iterator[str]:
        for e in executor.follow(eid):
            yield f"data: {json.dumps(e, default=str)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/experiments/{exp_id}/steps/{step_id}/output")
def output(exp_id: str, step_id: str) -> dict:
    """What a step made last time it ran: its line, and the first rows of any table it wrote."""
    _exp(exp_id)
    last = next((r for r in store.state(exp_id, "runs") if r.get("step") == step_id), None)
    if not last:
        return {"step": step_id, "ran": False}
    rows = lab.read_table(exp_id, last["table"]) if last.get("table") else []
    return {"step": step_id, "ran": True, **last, "rows": rows[:50], "total_rows": len(rows)}


@router.post("/experiments/{exp_id}/active")
def active(exp_id: str, body: ActiveIn) -> dict:
    """The new-files trigger: run the workflow when FCS files arrive in the data folder."""
    _exp(exp_id)
    return executor.watch(exp_id, body.on)
