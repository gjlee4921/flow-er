"""One folder per experiment, outside the repo.

    <workspace_dir>/<id>/
        experiment.json   title, data folder, question
        state/            plan.json, gates.json, views.json, macros.json, runs.json
        draft.json        the one change waiting for keep or undo
        tables/           saved tables; every number the agent shows comes from one
        uploads/          files added in the chat
        journal.jsonl     every tool call, keep and undo
        chat.jsonl        the conversation
        lab_chat.jsonl    the data lab assistant's own conversation, apart from the main one

Every change goes through a draft: nothing in state/ moves until the scientist keeps it.
"""
from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from flow_er.config import settings

STATE_FILES = {"plan": [], "gates": [], "views": [], "macros": [], "runs": []}


def root() -> Path:
    p = settings.workspace_dir.expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _dir(exp_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9-]{4,64}", exp_id):
        raise KeyError(exp_id)
    d = root() / exp_id
    if not (d / "experiment.json").exists():
        raise KeyError(exp_id)
    return d


def _read(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write(path: Path, value: Any) -> None:
    """Write a whole file at once: to a temporary file of its own, then swapped into place, so two
    writes at the same moment never share a temporary file and a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ---- experiments -------------------------------------------------------------------------

def create(data_path: Path, title: str = "New analysis") -> dict:
    exp_id = f"{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
    d = root() / exp_id
    (d / "state").mkdir(parents=True)
    (d / "tables").mkdir()
    (d / "uploads").mkdir()
    meta = {"id": exp_id, "title": title, "data_path": str(data_path), "created": now(), "question": ""}
    _write(d / "experiment.json", meta)
    for name, empty in STATE_FILES.items():
        _write(d / "state" / f"{name}.json", empty)
    journal(exp_id, "create", {"data_path": str(data_path)})
    return meta


def list_all(archived: bool = False) -> list[dict]:
    """Projects, the one opened most recently first, each with how far it has got. Archived projects
    are left out unless asked for."""
    out = []
    for d in root().iterdir():
        meta = _read(d / "experiment.json", None)
        if not meta or bool(meta.get("archived")) != archived:
            continue
        out.append({**meta, "plan_kept": bool(_read(d / "state" / "plan.json", [])),
                    "figures": len(_read(d / "state" / "views.json", []))})
    return sorted(out, key=lambda m: m.get("last_opened") or m["created"], reverse=True)


def meta(exp_id: str) -> dict:
    return _read(_dir(exp_id) / "experiment.json", {})


def update_meta(exp_id: str, **fields: Any) -> dict:
    m = meta(exp_id)
    m.update({k: v for k, v in fields.items() if v is not None})
    _write(_dir(exp_id) / "experiment.json", m)
    return m


def delete(exp_id: str) -> None:
    shutil.rmtree(_dir(exp_id))


def path(exp_id: str, *parts: str) -> Path:
    return _dir(exp_id).joinpath(*parts)


# ---- state -------------------------------------------------------------------------------

def state(exp_id: str, name: str) -> Any:
    return _read(_dir(exp_id) / "state" / f"{name}.json", STATE_FILES.get(name, []))


def set_state(exp_id: str, name: str, value: Any) -> None:
    _write(_dir(exp_id) / "state" / f"{name}.json", value)


# ---- the draft: one pending change -------------------------------------------------------

def draft(exp_id: str) -> dict | None:
    return _read(_dir(exp_id) / "draft.json", None)


def set_draft(exp_id: str, kind: str, payload: Any, summary: str) -> dict:
    """Stage a change. Replaces any draft that was waiting: the newest proposal wins."""
    d = {"id": uuid.uuid4().hex[:8], "kind": kind, "payload": payload, "summary": summary, "created": now()}
    _write(_dir(exp_id) / "draft.json", d)
    journal(exp_id, "draft", {"kind": kind, "summary": summary})
    return d


APPLY = {
    # kind -> (state file, how the payload lands in it)
    "plan": ("plan", "replace"),
    "report": ("report", "replace"),
    "gate": ("gates", "append"),
    "view": ("views", "append"),
    "macro": ("macros", "append"),
}


def keep(exp_id: str, who: str = "scientist") -> dict | None:
    d = draft(exp_id)
    if not d:
        return None
    name, how = APPLY[d["kind"]]
    if how == "replace":
        set_state(exp_id, name, d["payload"])
    else:
        # a draft can hold several items at once (a quadrant is four gates)
        new = d["payload"] if isinstance(d["payload"], list) else [d["payload"]]
        ids = {x.get("id") or x.get("key") for x in new}
        items = [x for x in state(exp_id, name) if (x.get("id") or x.get("key")) not in ids]
        set_state(exp_id, name, items + new)
    (_dir(exp_id) / "draft.json").unlink(missing_ok=True)
    journal(exp_id, "keep", {"kind": d["kind"], "summary": d["summary"], "who": who})
    return d


def undo(exp_id: str, who: str = "scientist") -> dict | None:
    d = draft(exp_id)
    if not d:
        return None
    (_dir(exp_id) / "draft.json").unlink(missing_ok=True)
    journal(exp_id, "undo", {"kind": d["kind"], "summary": d["summary"], "who": who})
    return d


# ---- journal and chat --------------------------------------------------------------------

def journal(exp_id: str, event: str, data: dict) -> None:
    with open(root() / exp_id / "journal.jsonl", "a") as f:
        f.write(json.dumps({"t": now(), "event": event, **data}, default=str) + "\n")


def read_journal(exp_id: str, limit: int = 200) -> list[dict]:
    p = _dir(exp_id) / "journal.jsonl"
    if not p.exists():
        return []
    lines = p.read_text().splitlines()[-limit:]
    return [json.loads(x) for x in lines if x.strip()]


THREADS = ("chat", "lab_chat")


def add_message(exp_id: str, msg: dict, thread: str = "chat") -> dict:
    assert thread in THREADS, thread
    msg = {"id": uuid.uuid4().hex[:10], "t": now(), **msg}
    with open(_dir(exp_id) / f"{thread}.jsonl", "a") as f:
        f.write(json.dumps(msg, default=str) + "\n")
    return msg


def messages(exp_id: str, thread: str = "chat") -> list[dict]:
    assert thread in THREADS, thread
    p = _dir(exp_id) / f"{thread}.jsonl"
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
