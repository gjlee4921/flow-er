"""The gating models in the app: who each file came from, model gates per sample, review, and the
lab's model versions. Heavy work (autogate on many files, retraining) runs as jobs, never in a request."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from flow_er.workspace import models, samplemeta, store

router = APIRouter(tags=["models"])


def _exp(exp_id: str) -> None:
    if not store.path(exp_id).exists():
        raise HTTPException(404, f"no experiment {exp_id}")


@router.get("/models/status")
def status() -> dict:
    return models.status()


class MetaIn(BaseModel):
    rows: list[dict] | None = None


@router.get("/experiments/{exp_id}/samples/meta")
def sample_meta(exp_id: str) -> dict:
    """Each file's individual, condition and unstimulated well: kept, or read from file names."""
    _exp(exp_id)
    return {"rows": samplemeta.get(exp_id)}


@router.post("/experiments/{exp_id}/samples/meta")
def draft_sample_meta(exp_id: str, body: MetaIn) -> dict:
    """Stage who each file came from as a draft (the rows given, or those read from file names)."""
    _exp(exp_id)
    try:
        return {"draft": samplemeta.draft(exp_id, body.rows)}
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
