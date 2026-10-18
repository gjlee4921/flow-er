"""Gating model versions: metrics, promote, roll back. Reads flow_er.ml; never trains in a request."""
from fastapi import APIRouter

from flow_er.workspace import models

router = APIRouter(tags=["models"])


@router.get("/models/status")
def status() -> dict:
    return models.status()
