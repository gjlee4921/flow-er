"""Training and inference job status. The worker that runs them comes with flow_er.ml."""
from fastapi import APIRouter

router = APIRouter(tags=["jobs"])


@router.get("/jobs")
def jobs() -> list[dict]:
    return []
