"""The app's side of the gating models in `flow_er.ml` (Gijeong's): pick a lab and its template,
gate files with it, score the result against FlowJo, take the scientist's corrections as training
examples, and retrain in the background. `flow_er.ml` is only called through its public API.

- A lab is a folder in models/labs/ (its template, model versions and `current`). An experiment uses
  the lab whose template names the gates its FlowJo workspace has; without one, the template is the
  workspace's gates on the first file, so every gate is a template copy.
- Model gates are drafts: they reach the lab as gates only when the scientist keeps them.
- Scoring against FlowJo (F1 per gate, % of parent) is for showing how well the model did; it never
  feeds the gating.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np

from flow_er.config import settings
from flow_er.workspace import lab as labmod
from flow_er.workspace import samplemeta, store

log = logging.getLogger(__name__)
_results: OrderedDict = OrderedDict()  # (exp, sample) -> GatingResult, the last few, for edits and keeps
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _ml():
    from flow_er import ml
    return ml


def labs_root() -> Path:
    return Path(settings.models_dir).expanduser() / "labs"


def status() -> dict:
    """Whether the gating models can run, and which labs there are."""
    try:
        _ml()
        import torch  # noqa: F401  the models need it
        ok = True
    except Exception:  # noqa: BLE001  not installed or not importable: say so, don't pretend
        ok = False
    labs = sorted(p.name for p in labs_root().glob("*") if (p / "template.json").exists()) if labs_root().exists() else []
    msg = (f"Gating models are ready: labs {', '.join(labs)}. Model gates are drafts the scientist keeps."
           if ok and labs else "Gating models are installed, but no lab has a template yet." if ok
           else "The gating models need PyTorch (uv sync --group ml).")
    return {"available": ok, "labs": labs, "message": msg}


# ---- which lab and template --------------------------------------------------------------------

def _flowjo_keys(exp_id: str) -> set[str]:
    ds = labmod.dataset(exp_id)
    for sid in ds.sample_ids:
        keys = {g.key for g in ds.flowjo_gates(sid)}
        if keys:
            return keys
    return set()


def lab_for(exp_id: str) -> str | None:
    """The lab whose template names most of this experiment's FlowJo gates (at least 80% of the
    template's gates), or None. Labs that match as well (a copy of a lab, say) are told apart by
    having a current version, then by name, so the choice never depends on folder order."""
    keys = _flowjo_keys(exp_id)
    found = []
    for t in labs_root().glob("*/template.json") if keys and labs_root().exists() else []:
        names = {g["key"] for g in json.loads(t.read_text()).get("gates", [])}
        share = len(names & keys) / max(1, len(names))
        if share >= 0.8:
            found.append((-share, not (t.parent / "current").exists(), t.parent.name))
    return min(found)[2] if found else None


def template(exp_id: str):
    """The gate tree the models fill in: the lab's template, else the workspace's gates on the first
    file (saved in the experiment, so it stays the same between runs)."""
    ml = _ml()
    lab = lab_for(exp_id)
    if lab:
        return ml.load_template(str(labs_root() / lab / "template.json"))
    saved = store.path(exp_id, "models", "template.json")
    if saved.exists():
        return ml.load_template(str(saved))
    ds = labmod.dataset(exp_id)
    sid = next((s for s in ds.sample_ids if ds.workspace_for(s)), None)
    if not sid:
        raise ValueError("there is no FlowJo workspace to take a template from; a lab template or one gated file is needed")
    name = ds._where[sid][1]  # the file's name inside its workspace
    tree = ml.template_from_workspace(str(ds.workspace_for(sid)), name, fcs_path=str(ds.sample_path(sid)))
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(json.dumps(tree.to_json()))
    return tree


def version(exp_id: str) -> str:
    """The model version answering for this experiment: the lab's current, or 'template'."""
    lab = lab_for(exp_id)
    cur = labs_root() / lab / "current" if lab else None
    return cur.read_text().strip() if cur and cur.exists() else "template"


# ---- gating a file --------------------------------------------------------------------------

def _path(exp_id: str, sid: str) -> Path:
    return store.path(exp_id, "models", "results", f"{sid}.json")


def autogate(exp_id: str, sid: str, force: bool = False) -> dict:
    """Gate one file with the models: per gate the polygon, count, % of parent, which model answered
    and how sure it was, plus FlowJo's % and an F1 when the file has FlowJo gates. Cached per file,
    template and model version."""
    ml = _ml()
    out = _path(exp_id, sid)
    ver, lab = version(exp_id), lab_for(exp_id)
    if out.exists() and not force:
        cached = json.loads(out.read_text())
        if cached.get("model_version_used") == ver:
            return cached
    ds = labmod.dataset(exp_id)
    meta = samplemeta.of(exp_id, sid)
    tree = template(exp_id)
    unstim = meta.get("unstim")
    unstim_path = str(ds.sample_path(unstim)) if unstim and unstim != sid else None
    t0 = time.time()
    res = ml.autogate(str(ds.sample_path(sid)), tree, lab=lab, unstim=unstim_path, plot=True)
    seconds = round(time.time() - t0, 2)
    body = res.to_json(plot=True)
    for g in body["gates"]:
        (g.get("plot") or {}).pop("cells", None)  # keep the density; the sampled cells are only for drawing live
    _score(exp_id, sid, res, body)
    body.update(sample=sid, individual=meta.get("individual"), unstim=unstim, model_version_used=ver, seconds=seconds)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body))
    with _lock:
        _results[(exp_id, sid)] = res
        while len(_results) > 12:
            _results.popitem(last=False)
    return body


def _score(exp_id: str, sid: str, res, body: dict) -> None:
    """FlowJo's % of parent and an event-level F1 per gate, where the file has that FlowJo gate.
    Time/QC gates are left unscored: they always copy the template, and FlowKit scales the Time
    channel differently from the models' recipe, so a comparison there would mislead."""
    ds = labmod.dataset(exp_id)
    keys = {g.key for g in ds.flowjo_gates(sid)}
    qc = {g.key for g in template(exp_id).gates if getattr(g, "qc", False)}
    for g in body["gates"]:
        key = g["gate"]
        if key not in keys or key in qc:
            continue
        try:
            truth = ds.flowjo_membership(sid, key)
            parent = ds.flowjo_membership(sid, g["parent"]) if g["parent"] in keys else np.ones_like(truth)
        except Exception:
            log.info("cannot score %s", key, exc_info=True)
            continue
        pred = res.gates[key].membership
        if len(pred) != len(truth):
            continue
        tp = int((pred & truth).sum())
        g["flowjo_pct"] = round(100.0 * truth.sum() / max(1, parent.sum()), 4)
        g["f1"] = round(2 * tp / max(1, int(pred.sum()) + int(truth.sum())), 4)
        g["error_pp"] = round(abs(g["percent_of_parent"] - g["flowjo_pct"]), 4)


def result(exp_id: str, sid: str) -> dict | None:
    p = _path(exp_id, sid)
    return json.loads(p.read_text()) if p.exists() else None


def _live(exp_id: str, sid: str):
    """The GatingResult in memory (gated again if it was dropped)."""
    with _lock:
        res = _results.get((exp_id, sid))
    if res is None:
        autogate(exp_id, sid, force=True)
        with _lock:
            res = _results[(exp_id, sid)]
    return res


def table(exp_id: str, samples: list[str] | None = None) -> list[dict]:
    """One row per file and gate, from the cached results: what the autogate step saves."""
    rows = []
    for sid in samples or labmod.dataset(exp_id).sample_ids:
        body = result(exp_id, sid) or autogate(exp_id, sid)
        for g in body["gates"]:
            rows.append({"sample": sid, "gate": g["gate"], "model": g["model"],
                         "confidence": g["confidence"] if g["confidence"] is not None else "",
                         "low_confidence": bool(g["low_confidence"]), "count": g["count"], "pct_parent": g["percent_of_parent"],
                         "flowjo_pct": g.get("flowjo_pct", ""), "f1": g.get("f1", "")})
    return rows


# ---- review: edits, keeps, retraining -------------------------------------------------------------

def review(exp_id: str, sid: str) -> dict:
    """A file's model gates as one draft of gates in the data lab, beside FlowJo's under the same
    parents ("CD3 (model)"), so the scientist sees both on the plot and keeps or undoes them. Polygons
    go in by their raw corners and the lab's own transforms. Opens on the first gate to review."""
    body = result(exp_id, sid) or autogate(exp_id, sid)
    ds = labmod.dataset(exp_id)
    xf = ds._transforms(sid)
    taken = {g["key"] for g in labmod.tree(exp_id, sid)}
    qc = {g.key for g in template(exp_id).gates if getattr(g, "qc", False)}
    gates, first = [], None
    for g in body["gates"]:
        raw = (g.get("polygon") or {}).get("raw")
        if not raw or g["gate"] in qc or (g["parent"] and g["parent"] not in taken):
            continue
        name = g["gate"].split("/")[-1] + " (model)"
        key = f"{g['parent']}/{name}" if g["parent"] else name
        pts = np.asarray(raw, dtype=float)
        cols = [xf[c].apply(pts[:, j]) if xf.get(c) is not None else pts[:, j] for j, c in enumerate((g["x"], g["y"]))]
        gates.append({"key": key, "name": name, "parent": g["parent"] or None, "source": "user", "kind": "polygon",
                      "dims": [g["x"], g["y"]], "vertices": [[float(a), float(b)] for a, b in zip(*cols)]})
        if first is None or (g["low_confidence"] and not first["low_confidence"]):
            first = {**g, "key": key}
    if not gates:
        raise ValueError("the models gave no gates to review for this file")
    d = store.set_draft(exp_id, "gate", gates, f"{len(gates)} model gates on {sid} to review")
    return {"draft": d, "gates": [g["key"] for g in gates], "sample": sid, "parent": first["parent"] or None,
            "x": first["x"], "y": first["y"], "gate": first["key"]}


def edit(exp_id: str, sid: str, gate: str, polygon_display: list[list[float]]) -> dict:
    """A gate the scientist moved: it and every gate below it are recomputed."""
    ml = _ml()
    ds = labmod.dataset(exp_id)
    res = ml.apply_edit(_live(exp_id, sid), template(exp_id), gate, np.asarray(polygon_display, dtype=float),
                        fcs_path=str(ds.sample_path(sid)))
    body = res.to_json(plot=True)
    for g in body["gates"]:
        (g.get("plot") or {}).pop("cells", None)
    _score(exp_id, sid, res, body)
    body.update(sample=sid, model_version_used=version(exp_id), edited=True)
    _path(exp_id, sid).write_text(json.dumps(body))
    with _lock:
        _results[(exp_id, sid)] = res
    return body


def accept(exp_id: str, sid: str, gates: dict[str, list[list[float]]] | None = None) -> dict:
    """The scientist keeps a file's gates (as the models left them, or as given): they become the
    lab's training examples, under the individual the file came from."""
    ml = _ml()
    lab = lab_for(exp_id)
    if not lab:
        raise ValueError("this experiment has no lab to learn from; its gates match no lab template")
    meta = samplemeta.of(exp_id, sid)
    res = _live(exp_id, sid)
    accepted = gates or {k: g.polygon_display.tolist() for k, g in res.gates.items() if g.polygon_display is not None}
    ds = labmod.dataset(exp_id)
    unstim = meta.get("unstim")
    files = ml.add_labels(lab, str(ds.sample_path(sid)), template(exp_id),
                          {k: np.asarray(v, dtype=float) for k, v in accepted.items()},
                          meta.get("individual") or sid, sample_id=f"{exp_id}/{sid}", condition=meta.get("condition") or "",
                          unstim=str(ds.sample_path(unstim)) if unstim and unstim != sid else None)
    store.journal(exp_id, "labels_kept", {"sample": sid, "lab": lab, "gates": len(accepted)})
    return {"lab": lab, "gates": len(accepted), "files": [str(f) for f in files or []], "individual": meta.get("individual")}


def versions(lab: str) -> list[dict]:
    return _ml().list_versions(lab)


def set_current(lab: str, v: str) -> dict:
    _ml().set_current(lab, v)
    return {"lab": lab, "current": v}


def retrain(lab: str, max_minutes: float | None = 10) -> dict:
    """Start retraining a lab in the background; its progress lines and summary land in the job."""
    jid = time.strftime("%H%M%S") + "-" + uuid.uuid4().hex[:4]
    job = {"id": jid, "kind": "retrain", "lab": lab, "status": "running", "started": store.now(), "lines": [], "summary": None}
    with _lock:
        _jobs[jid] = job

    def work():
        try:
            job["summary"] = _ml().retrain(lab, max_minutes=max_minutes, log=lambda line: job["lines"].append(str(line)))
            job["status"] = "done"
        except Exception as e:
            log.exception("retrain %s failed", lab)
            job["status"], job["error"] = "failed", f"{e.__class__.__name__}: {e}"
        job["finished"] = store.now()

    threading.Thread(target=work, daemon=True).start()
    return job


def job(jid: str) -> dict | None:
    return _jobs.get(jid)


def jobs() -> list[dict]:
    return sorted(_jobs.values(), key=lambda j: j["started"], reverse=True)


def train(exp_id: str, gate: str | None) -> dict:
    """The agent's train_model: retrain the experiment's lab (all gates with new corrections)."""
    lab = lab_for(exp_id)
    if not lab:
        return {"queued": False, "message": "This experiment's gates match no lab template, so there is no model to train."}
    j = retrain(lab)
    store.journal(exp_id, "train_requested", {"gate": gate, "lab": lab, "job": j["id"]})
    return {"queued": True, "job": j["id"], "message": f"Retraining lab {lab} in the background; a new version is used only if it is better on held-out individuals."}
