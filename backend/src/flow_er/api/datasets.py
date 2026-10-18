"""Bringing data in: example folders, files uploaded on the start page, and folders on this machine
read in place. Data stays outside the repo, under DATA_DIR: uploads in uploads/<date>-<name>/, links
in linked/<name> (a link to the folder, nothing copied)."""
import os
import re
import shutil
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, UploadFile
from pydantic import BaseModel

from flow_er.config import settings
from flow_er.workspace import data, store

router = APIRouter(tags=["datasets"])
NOTES = {".csv", ".tsv", ".txt", ".md", ".pdf", ".docx", ".xlsx"}  # protocols and plate maps, kept beside the data
CHUNK = 1024 * 1024


class LinkIn(BaseModel):
    path: str
    name: str | None = None


def _base() -> Path:
    return Path(os.path.abspath(settings.data_dir.expanduser()))


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")[:40] or "data"


def _free(folder: Path) -> Path:
    """The folder itself, or the same name with -2, -3... if it is taken."""
    out, n = folder, 2
    while out.exists() or out.is_symlink():
        out, n = folder.with_name(f"{folder.name}-{n}"), n + 1
    return out


def _dataset(folder: Path, name: str, kind: str) -> dict:
    d = data.describe_dataset(folder, name, kind)
    if not d:
        raise HTTPException(400, f"no FCS files in {folder}")
    return d


@router.get("/datasets")
def datasets() -> list[dict]:
    return data.find_datasets()


@router.post("/datasets")
def upload(files: list[UploadFile], name: str = Form("")) -> dict:
    """Files from the start page become a dataset of their own. FCS files are checked by their header,
    a FlowJo workspace by its XML; protocols and plate maps are kept beside them; anything else is
    turned away with the reason. Files are copied to disk in chunks, never held in memory."""
    # named by the FlowJo workspace if there is one (it names the experiment), else by the first file
    names = [Path(f.filename or "upload").stem for f in files]
    wsp = [Path(f.filename or "").stem for f in files if (f.filename or "").lower().endswith(".wsp")]
    stem = name or (wsp[0] if wsp else names[0])
    folder = _free(_base() / "uploads" / f"{date.today().isoformat()}-{_slug(stem)}")
    folder.mkdir(parents=True)
    limit = settings.upload_max_mb * CHUNK
    kept, rejected = [], []
    for f in files:
        fname = Path(f.filename or "file").name
        ext = Path(fname).suffix.lower()
        if ext not in {".fcs", ".wsp"} | NOTES:
            rejected.append({"name": fname, "why": "not an FCS file, a FlowJo workspace or a protocol"})
            continue
        dest, size = folder / fname, 0
        with open(dest, "wb") as out:
            while chunk := f.file.read(CHUNK):
                size += len(chunk)
                if size > limit:
                    break
                out.write(chunk)
        why = None
        if size > limit:
            why = f"larger than {settings.upload_max_mb} MB"
        elif ext == ".fcs" and not data.fcs_header(dest):
            why = "named .fcs but it is not an FCS file"
        elif ext == ".wsp" and b"<Workspace" not in dest.read_bytes()[:4096]:
            why = "named .wsp but it is not a FlowJo workspace"
        if why:
            dest.unlink(missing_ok=True)
            rejected.append({"name": fname, "why": why})
        else:
            kept.append({"name": fname, "bytes": size})
    if not any(k["name"].lower().endswith(".fcs") for k in kept):
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(400, {"message": "none of these files is an FCS file", "rejected": rejected})
    return {**_dataset(folder, f"uploads/{folder.name}", "upload"), "kept": kept, "rejected": rejected}


@router.post("/datasets/link")
def link(body: LinkIn) -> dict:
    """A folder on this machine, read where it is: big files are not copied. Only folders under
    IMPORT_ROOTS can be linked."""
    folder = Path(os.path.abspath(Path(body.path).expanduser()))
    roots = [Path(os.path.abspath(Path(r.strip()).expanduser())) for r in settings.import_roots.split(",") if r.strip()]
    if not any(folder == r or r in folder.parents for r in roots):
        raise HTTPException(400, f"only folders under {', '.join(str(r) for r in roots)} can be used")
    if not folder.is_dir():
        raise HTTPException(400, f"there is no folder at {folder}")
    if not any(folder.glob("*.fcs")) and not any(folder.glob("*/*.fcs")):
        raise HTTPException(400, f"there are no FCS files in {folder.name}")
    links = _base() / "linked"
    links.mkdir(parents=True, exist_ok=True)
    for existing in links.iterdir():
        if existing.is_symlink() and Path(os.path.realpath(existing)) == Path(os.path.realpath(folder)):
            return _dataset(existing, f"linked/{existing.name}", "folder")
    target = _free(links / _slug(body.name or folder.name))
    target.symlink_to(folder, target_is_directory=True)
    return _dataset(target, f"linked/{target.name}", "folder")


@router.delete("/datasets/{name:path}")
def remove(name: str) -> dict:
    """Remove an uploaded copy or a link (the linked folder itself is never touched). Examples
    cannot be removed here, and nothing a project still uses."""
    container = name.split("/")[0]
    if container not in data.CONTAINERS or "/" not in name or ".." in name:
        raise HTTPException(400, "only uploaded or linked datasets can be removed")
    path = _base() / name
    if not (path.exists() or path.is_symlink()):
        raise HTTPException(404, f"no dataset {name}")
    users = [m["title"] for m in store.list_all() if os.path.abspath(m.get("data_path", "")) == os.path.abspath(path)]
    if users:
        raise HTTPException(409, f"still used by {len(users)} project(s): {', '.join(users[:3])}")
    path.unlink() if path.is_symlink() else shutil.rmtree(path)
    return {"removed": name}
