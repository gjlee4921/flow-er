"""Running a workflow: the kept plan's steps in the order their inputs allow, each one recorded, and
nothing done twice that has not changed.

An execution runs all of the plan, up to a step, one step, or from a step on. Each step's result is
kept as an artifact under a key made from what it is (its kind, code version, settings and inputs)
and the data it read (the files and the kept gates), so running again reuses every step whose key
is unchanged ("cached") and redoes only what changed and what depends on it. Executions are kept in
<experiment>/executions/ and their progress is streamed to the screen as it happens.

The new-files trigger watches the dataset folder; when FCS files arrive it runs the workflow for
those files only.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from pathlib import Path

from flow_er.workspace import data, lab, nodes, store

log = logging.getLogger(__name__)
MODES = ("all", "to", "only", "from")
_live: dict[str, dict] = {}  # execution id -> {"events": [...], "done": bool, "cond": Condition}
_lock = threading.Lock()
_watchers: dict[str, threading.Event] = {}


def _dir(exp_id: str) -> Path:
    d = store.path(exp_id, "executions")
    d.mkdir(exist_ok=True)
    return d


def _save(exp_id: str, rec: dict) -> None:
    p = _dir(exp_id) / f"{rec['id']}.json"
    tmp = p.with_name(f".{p.name}.{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=str))
    tmp.replace(p)


def get(exp_id: str, eid: str) -> dict:
    p = _dir(exp_id) / f"{eid}.json"
    if not p.exists():
        raise KeyError(eid)
    return json.loads(p.read_text())


def history(exp_id: str, limit: int = 20) -> list[dict]:
    recs = [json.loads(p.read_text()) for p in _dir(exp_id).glob("*.json")]
    return sorted(recs, key=lambda r: r["started"], reverse=True)[:limit]


def select(steps: list[dict], mode: str, node: str | None) -> list[dict]:
    """Which steps a mode runs: all; up to a step (it and everything it reads from); only it; or it and
    everything that reads from it."""
    if mode == "all" or not node:
        return steps
    by_id = {s["id"]: s for s in steps}
    if node not in by_id:
        raise ValueError(f"no kept step called '{node}'")
    made_by = {o: s["id"] for s in steps for o in s.get("outputs", [])}
    if mode == "only":
        return [by_id[node]]
    if mode == "to":
        need, todo = set(), [node]
        while todo:
            n = todo.pop()
            if n in need:
                continue
            need.add(n)
            todo += [made_by[i] for i in by_id[n].get("inputs", []) if i in made_by]
        return [s for s in steps if s["id"] in need]
    after, changed = {node}, True
    while changed:
        changed = False
        for s in steps:
            if s["id"] not in after and any(made_by.get(i) in after for i in s.get("inputs", [])):
                after.add(s["id"])
                changed = True
    return [s for s in steps if s["id"] in after]


def _fingerprint(exp_id: str) -> dict:
    """The data steps read: the files (names, sizes, times), and the kept gates, which only the
    steps that count gates depend on (so keeping a gate does not redo compensation)."""
    ds = lab.dataset(exp_id)
    files = [(p.name, p.stat().st_size, int(p.stat().st_mtime)) for p in ds.fcs]
    sha = lambda v: hashlib.sha1(json.dumps(v, default=str).encode()).hexdigest()
    try:  # the model version and who each file came from: what the model steps also depend on
        from flow_er.workspace import models
        model_print = sha([models.version(exp_id), store.state(exp_id, "sample_meta")])
    except Exception:  # noqa: BLE001  without the models nothing reads this
        model_print = "none"
    return {"files": sha(files), "gates": sha(store.state(exp_id, "gates")), "models": model_print}


def _key(step: dict, input_keys: list[str], data_print: dict, samples: list[str] | None) -> str:
    spec = {k: v for k, v in step.items() if k not in ("title", "why", "id", "x", "y")}
    kind = nodes.NODE_TYPES.get(step["op"], {})
    data = {"files": data_print.get("files"), **({"gates": data_print.get("gates")} if kind.get("reads_gates") else {}),
            **({"models": data_print.get("models")} if kind.get("reads_models") else {})}
    body = {"step": spec, "version": kind.get("version"), "inputs": input_keys, "data": data, "samples": samples}
    return hashlib.sha1(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _emit(eid: str, event: dict) -> None:
    with _lock:
        live = _live.get(eid)
    if not live:
        return
    with live["cond"]:
        live["events"].append(event)
        live["cond"].notify_all()


def start(exp_id: str, mode: str = "all", node: str | None = None, trigger: str = "manual",
          samples: list[str] | None = None, wait: bool = False) -> dict:
    """Begin an execution and return its record; with `wait`, return it when it has finished."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    steps = store.state(exp_id, "plan")
    if not steps:
        raise ValueError("there is no kept plan to run yet")
    chosen = select(steps, mode, node)
    rec = {"id": time.strftime("%H%M%S") + "-" + uuid.uuid4().hex[:4], "trigger": trigger, "mode": mode, "node": node,
           "samples": samples, "started": store.now(), "finished": None, "status": "running",
           "nodes": {s["id"]: {"status": "queued", "title": s["title"], "op": s["op"]} for s in chosen}}
    _save(exp_id, rec)
    with _lock:
        _live[rec["id"]] = {"events": [{"type": "start", "execution": rec}], "done": False, "cond": threading.Condition()}
    t = threading.Thread(target=_run, args=(exp_id, rec, steps, chosen), daemon=True)
    _live[rec["id"]]["thread"] = t
    t.start()
    return wait_for(exp_id, rec["id"]) if wait else rec


def wait_for(exp_id: str, eid: str, timeout: float = 300) -> dict:
    """The execution's record once it has finished (or after `timeout` seconds)."""
    with _lock:
        t = _live.get(eid, {}).get("thread")
    if t:
        t.join(timeout=timeout)
    return get(exp_id, eid)


def _run(exp_id: str, rec: dict, steps: list[dict], chosen: list[dict]) -> None:
    keys: dict[str, str] = {}  # output name -> the key of the artifact that made it
    try:
        data_print = _fingerprint(exp_id)
    except Exception:
        log.exception("could not read the dataset for %s", exp_id)
        data_print = {"files": "unreadable"}
    made_by = {o: s for s in steps for o in s.get("outputs", [])}
    chosen_ids = {s["id"] for s in chosen}
    for s in steps:
        input_keys = [keys.get(i) or i for i in s.get("inputs", [])]
        key = _key(s, input_keys, data_print, rec.get("samples"))
        for o in s.get("outputs", []):
            keys[o] = key
        if s["id"] not in chosen_ids:
            continue
        n = rec["nodes"][s["id"]]
        upstream_failed = [i for i in s.get("inputs", []) if i in made_by and rec["nodes"].get(made_by[i]["id"], {}).get("status") in ("failed", "blocked")]
        t0 = time.time()
        n.update(status="running", started=store.now(), key=key)
        _emit(rec["id"], {"type": "node", "id": s["id"], **n})
        art = store.path(exp_id, "artifacts", key)
        manifest = art / "manifest.json"
        if upstream_failed:
            result = {"status": "blocked", "summary": f"Waiting on {', '.join(made_by[i]['title'] for i in upstream_failed)}."}
        elif manifest.exists():
            result = {**json.loads(manifest.read_text()), "cached": True}
        else:
            try:
                result = nodes.run(exp_id, s, rec.get("samples"))
            except Exception as e:
                log.exception("step %s failed", s["id"])
                result = {"status": "failed", "summary": f"{e.__class__.__name__}: {e}"}
            if result["status"] == "done":
                art.mkdir(parents=True, exist_ok=True)
                manifest.write_text(json.dumps({**result, "op": s["op"], "made": store.now()}, default=str))
        n.update(status="cached" if result.get("cached") and result["status"] == "done" else result["status"],
                 summary=result.get("summary", ""), table=result.get("table"), seconds=round(time.time() - t0, 2))
        _emit(rec["id"], {"type": "node", "id": s["id"], **n})
        _save(exp_id, rec)
        # the step's latest result, which the plan's lanes and the agent read
        runs = [r for r in store.state(exp_id, "runs") if r.get("step") != s["id"]]
        store.set_state(exp_id, "runs", runs + [{"step": s["id"], "op": s["op"], "status": "done" if n["status"] == "cached" else n["status"],
                                                 "summary": n["summary"], "table": n.get("table"), "at": store.now(), "execution": rec["id"]}])
    statuses = [n["status"] for n in rec["nodes"].values()]
    rec.update(finished=store.now(), status="failed" if "failed" in statuses else "blocked" if "blocked" in statuses else "done")
    _save(exp_id, rec)
    store.journal(exp_id, "execution", {"id": rec["id"], "trigger": rec["trigger"], "status": rec["status"]})
    _emit(rec["id"], {"type": "done", "execution": rec})
    with _lock:
        live = _live.get(rec["id"])
    if live:
        with live["cond"]:
            live["done"] = True
            live["cond"].notify_all()


def follow(eid: str, timeout: float = 600):
    """The events of a running execution as they happen (for the screen); ends when it finishes."""
    with _lock:
        live = _live.get(eid)
    if not live:
        return
    i, t_end = 0, time.time() + timeout
    while time.time() < t_end:
        with live["cond"]:
            while i >= len(live["events"]) and not live["done"]:
                live["cond"].wait(timeout=15)
                if i >= len(live["events"]) and not live["done"]:
                    yield {"type": "ping"}
            batch, i = live["events"][i:], len(live["events"])
            finished = live["done"]
        yield from batch
        if finished and i >= len(live["events"]):
            return


# ---- the new-files trigger -------------------------------------------------------------------

def watch(exp_id: str, on: bool) -> dict:
    """Run the workflow when new FCS files arrive in the project's data folder (the workflow is then
    active), or stop watching."""
    store.update_meta(exp_id, active=bool(on))
    stop = _watchers.pop(exp_id, None)
    if stop:
        stop.set()
    if on:
        ev = threading.Event()
        _watchers[exp_id] = ev
        threading.Thread(target=_watch, args=(exp_id, ev), daemon=True).start()
    store.journal(exp_id, "trigger", {"new_files": bool(on)})
    return {"active": bool(on)}


def _watch(exp_id: str, stop: threading.Event, every: float = 2.0) -> None:
    folder = Path(store.meta(exp_id)["data_path"])
    seen = {p.name for p in folder.rglob("*.fcs")}
    while not stop.wait(every):
        try:
            now = {p.name for p in folder.rglob("*.fcs")}
            new = sorted(now - seen)
            if not new:
                continue
            time.sleep(1.0)  # let a file that is still being copied finish
            seen = now
            data.open_dataset.cache_clear()
            ids = [sid for sid in lab.dataset(exp_id).sample_ids if sid in new]
            if ids and store.state(exp_id, "plan"):
                store.journal(exp_id, "new_files", {"files": ids})
                start(exp_id, "all", trigger="new files", samples=ids)
        except Exception:
            log.exception("watching %s", exp_id)


def resume_watchers() -> None:
    """After a restart, pick up the projects whose new-files trigger was on."""
    for m in store.list_all():
        if m.get("active"):
            watch(m["id"], True)
