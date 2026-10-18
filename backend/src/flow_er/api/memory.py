"""What flow-er remembers about the scientist, for the memory panel: read it, add to it, change it,
forget it, or pause remembering altogether."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from flow_er.workspace import memory, store

router = APIRouter(tags=["memory"])


class MemoryIn(BaseModel):
    text: str
    kind: str = "preference"


class PauseIn(BaseModel):
    paused: bool


@router.get("/memory")
def list_memory() -> dict:
    return {"memories": memory.entries(), "paused": memory.paused()}


@router.post("/memory")
def add_memory(body: MemoryIn) -> dict:
    """A memory the scientist typed into the panel themselves."""
    if len(body.text.strip()) < 3:
        raise HTTPException(400, "say what flow-er should remember")
    return memory.add(body.text, body.kind, body.text, {"experiment": None, "title": "typed in the memory panel", "at": store.now()})


@router.patch("/memory/{mem_id}")
def edit_memory(mem_id: str, body: MemoryIn) -> dict:
    try:
        return memory.update(mem_id, body.text)
    except KeyError:
        raise HTTPException(404, f"no memory {mem_id}")


@router.delete("/memory/{mem_id}")
def forget_memory(mem_id: str) -> dict:
    try:
        return {"forgotten": memory.remove(mem_id)}
    except KeyError:
        raise HTTPException(404, f"no memory {mem_id}")


@router.post("/memory/pause")
def pause_memory(body: PauseIn) -> dict:
    memory.set_paused(body.paused)
    return {"paused": memory.paused()}
