"""The gating models in the app: who each file came from, model gates per sample, review, and the
lab's model versions. Heavy work (autogate on many files, retraining) runs as jobs, never in a request."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from flow_er.workspace import lab, models, samplemeta, store

router = APIRouter(tags=["models"])


def _exp(exp_id: str) -> None:
    if not store.path(exp_id).exists():
        raise HTTPException(404, f"no experiment {exp_id}")


def _sid(exp_id: str, sample: str) -> str:
    if sample not in lab.dataset(exp_id).sample_ids:
        raise HTTPException(404, f"no file {sample}")
    return sample


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


@router.get("/experiments/{exp_id}/autogate")
def autogate_files(exp_id: str) -> dict:
    """Which files have model gates yet, the lab and the version answering."""
    _exp(exp_id)
    ds = lab.dataset(exp_id)
    return {"lab": models.lab_for(exp_id), "version": models.version(exp_id),
            "files": [{"sample": s, "gated": models.result(exp_id, s) is not None} for s in ds.sample_ids]}


@router.get("/experiments/{exp_id}/autogate/live")
def autogate_live(exp_id: str) -> dict:
    """The models at work: the run's files, the one being gated now, and those done (and done just now)."""
    _exp(exp_id)
    return models.live(exp_id)


@router.get("/experiments/{exp_id}/autogate/{sample:path}/result")
def autogate_result(exp_id: str, sample: str, run: bool = True) -> dict:
    """One file's model gates: per gate the polygon (display and raw units), count, % of parent, the
    model that answered, its confidence, the parent's density, and FlowJo's % and F1 where known."""
    _exp(exp_id)
    sid = _sid(exp_id, sample)
    body = models.result(exp_id, sid)
    if body is None and run:
        try:
            body = models.autogate(exp_id, sid)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
    if body is None:
        raise HTTPException(404, "this file has not been gated by the models yet")
    return body


@router.post("/experiments/{exp_id}/autogate/{sample:path}/review")
def autogate_review(exp_id: str, sample: str) -> dict:
    """Open a file's model gates in the data lab as one draft, beside FlowJo's."""
    _exp(exp_id)
    try:
        return models.review(exp_id, _sid(exp_id, sample))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


class EditIn(BaseModel):
    gate: str
    polygon: list[list[float]]


@router.post("/experiments/{exp_id}/autogate/{sample:path}/edit")
def autogate_edit(exp_id: str, sample: str, body: EditIn) -> dict:
    """A model gate the scientist moved (display units): it and every gate below it are recomputed."""
    _exp(exp_id)
    return models.edit(exp_id, _sid(exp_id, sample), body.gate, body.polygon)


class AcceptIn(BaseModel):
    gates: dict[str, list[list[float]]] | None = None


@router.post("/experiments/{exp_id}/autogate/{sample:path}/accept")
def autogate_accept(exp_id: str, sample: str, body: AcceptIn) -> dict:
    """The scientist keeps a file's gates: they become training examples for the lab."""
    _exp(exp_id)
    try:
        return models.accept(exp_id, _sid(exp_id, sample), body.gates)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.get("/labs/{lab_name}/models")
def lab_models(lab_name: str) -> dict:
    """Every version of a lab's models, with metrics and which one is current."""
    try:
        return {"lab": lab_name, "versions": models.versions(lab_name), "taught": models.taught(lab_name)}
    except Exception as e:
        raise HTTPException(404, f"no models for lab {lab_name}: {e}") from e


@router.post("/labs/{lab_name}/models/{version}/current")
def lab_set_current(lab_name: str, version: str) -> dict:
    """Roll back (or forward) to an existing version."""
    try:
        return models.set_current(lab_name, version)
    except Exception as e:
        raise HTTPException(400, str(e)) from e


class RetrainIn(BaseModel):
    max_minutes: float | None = 10
    gates: list[str] | None = None


@router.post("/labs/{lab_name}/retrain")
def lab_retrain(lab_name: str, body: RetrainIn) -> dict:
    """Retrain a lab's models on its kept gates, as a background job (poll /jobs/{id})."""
    try:
        return models.retrain(lab_name, body.max_minutes, body.gates)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
