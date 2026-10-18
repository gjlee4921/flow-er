"""Study sheet: experiments, samples, factor roles (compare / pair by / split / ignore).

An experiment is one folder of data plus everything kept about it. Changes are drafts until kept.
"""
import os
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from flow_er.config import settings
from flow_er.llm import tools
from flow_er.schemas import NewExperiment, ProjectPatch
from flow_er.workspace import data, lab, store

router = APIRouter(tags=["experiments"])
MAX_UPLOAD = 1024 * 1024 * 1024  # 1 GB per file


def _get(exp_id: str) -> dict:
    try:
        return store.meta(exp_id)
    except KeyError:
        raise HTTPException(404, f"no experiment {exp_id}")


@router.get("/experiments")
def experiments(archived: bool = False) -> list[dict]:
    return store.list_all(archived)


@router.post("/experiments")
def create(body: NewExperiment) -> dict:
    # Containment is checked on the path as written, so a folder linked into DATA_DIR counts as inside.
    base = Path(os.path.abspath(settings.data_dir.expanduser()))
    known = {d["name"]: d["path"] for d in data.find_datasets()}
    folder = Path(os.path.abspath(Path(known.get(body.dataset, body.dataset)).expanduser()))
    if base not in folder.parents and folder != base:
        raise HTTPException(400, "the dataset must be inside DATA_DIR")
    if not any(folder.rglob("*.fcs")):
        raise HTTPException(400, f"no FCS files in {folder.name}")
    meta = store.create(folder, body.title or folder.name.replace("_", " "))
    return {**meta, "snapshot": tools.snapshot(meta["id"])}


@router.get("/experiments/{exp_id}")
def get(exp_id: str) -> dict:
    meta = _get(exp_id)
    return {
        **meta,
        "snapshot": tools.snapshot(exp_id),
        "plan": store.state(exp_id, "plan"),
        "runs": store.state(exp_id, "runs"),
        "gates": store.state(exp_id, "gates"),
        "views": store.state(exp_id, "views"),
        "macros": store.state(exp_id, "macros"),
        "draft": store.draft(exp_id),
        "tables": lab.tables(exp_id),
    }


@router.patch("/experiments/{exp_id}")
def update(exp_id: str, body: ProjectPatch) -> dict:
    """Rename a project, or archive it (it leaves the list; nothing is deleted) and bring it back."""
    _get(exp_id)
    title = body.title.strip() if body.title is not None else None
    if title == "":
        raise HTTPException(400, "a project needs a name")
    m = store.update_meta(exp_id, title=title, archived=body.archived)
    if body.title is not None:
        store.journal(exp_id, "rename", {"title": title})
    if body.archived is not None:
        store.journal(exp_id, "archive" if body.archived else "unarchive", {})
    return m


@router.post("/experiments/{exp_id}/opened")
def opened(exp_id: str) -> dict:
    """Note when a project was opened, so the project list leads with the one you were just in."""
    _get(exp_id)
    return store.update_meta(exp_id, last_opened=store.now())


@router.delete("/experiments/{exp_id}")
def delete(exp_id: str) -> dict:
    _get(exp_id)
    store.delete(exp_id)
    return {"deleted": exp_id}


@router.get("/experiments/{exp_id}/samples")
def samples(exp_id: str) -> dict:
    _get(exp_id)
    ds = lab.dataset(exp_id)
    return {"samples": ds.samples(), "channels": ds.channels(),
            "workspace": ds.wsp_path.name if ds.wsp_path else None}


@router.post("/experiments/{exp_id}/uploads")
def upload(exp_id: str, files: list[UploadFile]) -> dict:
    """Files added in the chat. Sync on purpose: FastAPI runs it in the threadpool."""
    _get(exp_id)
    saved = []
    for f in files:
        name = re.sub(r"[^A-Za-z0-9._ -]+", "_", Path(f.filename or "file").name)[:120]
        dest = store.path(exp_id, "uploads", name)
        size = 0
        with open(dest, "wb") as out:
            while chunk := f.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(413, f"{name} is larger than 1 GB")
                out.write(chunk)
        saved.append({"name": name, "bytes": size})
        store.journal(exp_id, "upload", {"name": name, "bytes": size})
    return {"saved": saved}


@router.post("/experiments/{exp_id}/keep")
def keep(exp_id: str) -> dict:
    _get(exp_id)
    d = store.keep(exp_id)
    if not d:
        raise HTTPException(409, "nothing is waiting to be kept")
    lab.after_change(exp_id, d["kind"])
    return {"kept": d}


@router.post("/experiments/{exp_id}/undo")
def undo(exp_id: str) -> dict:
    _get(exp_id)
    d = store.undo(exp_id)
    if not d:
        raise HTTPException(409, "nothing is waiting to be undone")
    lab.after_change(exp_id, d["kind"])
    return {"dropped": d}


@router.get("/experiments/{exp_id}/journal")
def journal(exp_id: str) -> list[dict]:
    _get(exp_id)
    return store.read_journal(exp_id)
