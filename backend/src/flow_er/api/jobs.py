"""Long jobs (retraining a lab's models): status, progress lines and the summary when done."""
from fastapi import APIRouter, HTTPException

from flow_er.workspace import models

router = APIRouter(tags=["jobs"])


@router.get("/jobs")
def jobs() -> list[dict]:
    return models.jobs()


@router.get("/jobs/{jid}")
def job(jid: str) -> dict:
    j = models.job(jid)
    if not j:
        raise HTTPException(404, f"no job {jid}")
    return j
