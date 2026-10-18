"""What flow-er remembers about the scientist: definitions, preferences, lab conventions, in their own
words, kept across projects in <WORKSPACE_DIR>/_user/memory.jsonl.

A memory is only ever written from something the scientist said in the chat (the agent passes the
exact words and code checks they were said this turn), or typed into the memory panel themselves, so a
document or a web page cannot plant one. Raw data, file contents and results are never stored here.
"""
from __future__ import annotations

import json
import re
import uuid

from flow_er.workspace import store

KINDS = ("definition", "preference", "lab", "project")
ALL_IN = 40  # up to this many, every memory goes to the agent; beyond, the most relevant ones


def _dir():
    d = store.root() / "_user"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path():
    return _dir() / "memory.jsonl"


def entries() -> list[dict]:
    p = _path()
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _save(items: list[dict]) -> None:
    tmp = _dir() / f".memory.{uuid.uuid4().hex[:8]}.tmp"
    tmp.write_text("".join(json.dumps(m) + "\n" for m in items))
    tmp.replace(_path())


def paused() -> bool:
    p = _dir() / "settings.json"
    return bool(json.loads(p.read_text()).get("paused")) if p.exists() else False


def set_paused(value: bool) -> None:
    (_dir() / "settings.json").write_text(json.dumps({"paused": bool(value)}))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("’", "'").replace("“", '"').replace("”", '"')).strip().lower()


def said(quote: str, message: str) -> bool:
    """Whether the quoted words are in what the scientist wrote (spacing and curly quotes aside)."""
    q = _norm(quote)
    return len(q) >= 8 and q in _norm(message)


def add(text: str, kind: str, quote: str, source: dict) -> dict:
    if kind not in KINDS:
        kind = "preference"
    m = {"id": uuid.uuid4().hex[:8], "text": text.strip(), "kind": kind, "quote": quote.strip(),
         "source": source, "created": store.now(), "last_used": None}
    _save(entries() + [m])
    return m


def update(mem_id: str, text: str) -> dict:
    items = entries()
    for m in items:
        if m["id"] == mem_id:
            m["text"] = text.strip()
            _save(items)
            return m
    raise KeyError(mem_id)


def remove(mem_id: str) -> dict:
    items = entries()
    gone = next((m for m in items if m["id"] == mem_id), None)
    if not gone:
        raise KeyError(mem_id)
    _save([m for m in items if m["id"] != mem_id])
    return gone


def find(words: str) -> dict | None:
    """The memory that best matches some words, for "forget what I said about Tregs"."""
    ws = set(re.findall(r"[a-z0-9+]+", words.lower()))
    scored = [(len(ws & set(re.findall(r"[a-z0-9+]+", (m["text"] + " " + m["quote"]).lower()))), m) for m in entries()]
    best = max(scored, key=lambda t: t[0], default=(0, None))
    return best[1] if best[0] else None


def for_agent(context: str) -> list[dict]:
    """The memories that go into the agent's instructions: all of them while there are few, else the
    most relevant to the conversation by shared words, then the newest."""
    items = entries()
    if len(items) <= ALL_IN:
        return items
    ws = set(re.findall(r"[a-z0-9+]+", context.lower()))
    scored = sorted(items, key=lambda m: (len(ws & set(re.findall(r"[a-z0-9+]+", m["text"].lower()))), m["created"]), reverse=True)
    return scored[:ALL_IN]


def past_projects(query: str, exclude: str | None = None, limit: int = 5) -> list[dict]:
    """Other projects that match some words: their titles, what they were for, kept plans and figures,
    and what was kept along the way. The lab's memory across experiments."""
    ws = set(re.findall(r"[a-z0-9+]+", query.lower()))
    out = []
    for meta in store.list_all() + store.list_all(archived=True):
        if meta["id"] == exclude:
            continue
        kept = [e.get("summary", "") for e in store.read_journal(meta["id"]) if e.get("event") == "keep"]
        plan = [s.get("title", "") for s in store.state(meta["id"], "plan")]
        figs = [v.get("title", "") for v in store.state(meta["id"], "views")]
        text = " ".join([meta.get("title", ""), meta.get("question", ""), *plan, *figs, *kept])
        hits = len(ws & set(re.findall(r"[a-z0-9+]+", text.lower())))
        if hits:
            out.append({"id": meta["id"], "title": meta["title"], "created": meta["created"][:10], "question": meta.get("question") or None,
                        "plan": plan, "figures": figs, "kept": kept[-5:], "score": hits})
    return sorted(out, key=lambda p: (p["score"], p["created"]), reverse=True)[:limit]
