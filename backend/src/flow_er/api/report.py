"""The results page as one document: the agent's words with their numbers resolved, the methods and
the data list written by code. Figures come from /views, as before."""
from fastapi import APIRouter, HTTPException

from flow_er.workspace import report, store

router = APIRouter(tags=["report"])


@router.get("/experiments/{exp_id}/report")
def get_report(exp_id: str) -> dict:
    try:
        store.meta(exp_id)
    except KeyError:
        raise HTTPException(404, f"no experiment {exp_id}")
    return report.build(exp_id)
