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
import re
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
    template's gates), or None. Files with no FlowJo gates at all (raw files, nobody has gated them)
    go to a lab whose template is drawn only on channels these files have: the same panel. Labs that
    match as well (a copy of a lab, say) are told apart by having a current version, then by name, so
    the choice never depends on folder order."""
    pinned = store.meta(exp_id).get("lab")
    if pinned and (labs_root() / pinned / "template.json").exists():
        return pinned          # a project pinned to one lab (a copy for a demo, say) teaches only that one
    keys = _flowjo_keys(exp_id)
    ds = labmod.dataset(exp_id)
    channels = set(ds._sample(ds.sample_ids[0]).pnn_labels) if not keys and ds.sample_ids else set()
    found = []
    for t in labs_root().glob("*/template.json") if labs_root().exists() else []:
        gates = json.loads(t.read_text()).get("gates", [])
        if keys:
            names = {g["key"] for g in gates}
            share = len(names & keys) / max(1, len(names))
        else:
            drawn = {c for g in gates for c in (g.get("x"), g.get("y")) if c}
            share = 1.0 if drawn and drawn <= channels else 0.0
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


def fingerprint(exp_id: str) -> str:
    """What a cached result depends on: the version and its calibrated review threshold, so
    calibrating a version re-flags the gates without a new version."""
    ver, lab = version(exp_id), lab_for(exp_id)
    vfile = labs_root() / (lab or "") / ver / "version.json"
    thr = json.loads(vfile.read_text()).get("confidence_threshold") if lab and vfile.exists() else None
    return f"{ver}@{thr}" if thr else ver


# ---- gating a file --------------------------------------------------------------------------

def _path(exp_id: str, sid: str) -> Path:
    return store.path(exp_id, "models", "results", f"{sid}.json")


def autogate(exp_id: str, sid: str, force: bool = False) -> dict:
    """Gate one file with the models: per gate the polygon, count, % of parent, which model answered
    and how sure it was, plus FlowJo's % and an F1 when the file has FlowJo gates. Cached per file,
    template and model version."""
    ml = _ml()
    out = _path(exp_id, sid)
    ver, lab, print_ = version(exp_id), lab_for(exp_id), fingerprint(exp_id)
    if out.exists() and not force:
        cached = json.loads(out.read_text())
        if cached.get("model_fingerprint", cached.get("model_version_used")) == print_:
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
    body.update(sample=sid, individual=meta.get("individual"), unstim=unstim, model_version_used=ver,
                model_fingerprint=print_, confidence_threshold=res.gates and next(iter(res.gates.values())).threshold,
                seconds=seconds)
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


# the models at work, per experiment (named apart from _live(), the gating results in memory)
_progress: dict[str, dict] = {}


def live(exp_id: str) -> dict:
    """The models at work, for watching: the files of the run, which is being gated now, which are done
    and which of those were gated just now (the rest were reused from an earlier run)."""
    st = _progress.get(exp_id)
    if st is None:
        # no run since the app started: the files gated so far, to watch again
        ids = labmod.dataset(exp_id).sample_ids
        st = {"running": False, "files": ids, "done": [s for s in ids if _path(exp_id, s).exists()], "fresh": [], "current": None}
    return {**st, "done": list(st["done"]), "fresh": list(st["fresh"]), "lab": lab_for(exp_id), "version": version(exp_id)}


def table(exp_id: str, samples: list[str] | None = None) -> list[dict]:
    """One row per file and gate, from the cached results: what the autogate step saves. Progress goes
    to live() as each file is gated, so the data lab can show the models working."""
    rows = []
    ids = list(samples or labmod.dataset(exp_id).sample_ids)
    st = _progress[exp_id] = {"running": True, "files": ids, "done": [], "fresh": [], "current": None, "started": store.now()}
    try:
        for sid in ids:
            st["current"] = sid
            body = result(exp_id, sid)
            if body is None:
                body = autogate(exp_id, sid)
                st["fresh"].append(sid)
            st["done"].append(sid)
            rows += _rows(sid, body, samplemeta.of(exp_id, sid))
    finally:
        st["running"], st["current"], st["finished"] = False, None, store.now()
    return rows


def _rows(sid: str, body: dict, meta: dict | None = None) -> list[dict]:
    """The table's rows for one file: who it came from (individual, the group their label starts with,
    the condition) beside each gate's numbers, so a figure can set ALS against HC straight from it."""
    meta = meta or {}
    ind = meta.get("individual") or ""
    grp = m.group(1) if (m := re.match(r"^([A-Za-z]+)-\d+$", ind)) else ""
    return [{"sample": sid, "individual": ind, "group": grp, "condition": meta.get("condition") or "",
             "gate": g["gate"], "gate_name": g["gate"].split("/")[-1], "parent": (g.get("parent") or "").split("/")[-1], "model": g["model"],
             "confidence": g["confidence"] if g["confidence"] is not None else "",
             "low_confidence": bool(g["low_confidence"]), "count": g["count"], "pct_parent": g["percent_of_parent"],
             "flowjo_pct": g.get("flowjo_pct", ""), "f1": g.get("f1", "")} for g in body["gates"]]


# ---- review: edits, keeps, retraining -------------------------------------------------------------

MODEL_NAME = re.compile(r" \(model(?: \d+)?\)$")


def _suffix(n: int) -> str:
    return "(model)" if n == 1 else f"(model {n})"


def review(exp_id: str, sid: str) -> dict:
    """A file's model gates as one draft of gates in the data lab, beside FlowJo's under the same
    parents ("CD3 (model)"), so the scientist sees both on the plot and keeps or undoes them. Polygons
    go in by their raw corners and the lab's own transforms. Opens on the first gate to review."""
    body = result(exp_id, sid) or autogate(exp_id, sid)
    ds = labmod.dataset(exp_id)
    xf = ds._transforms(sid)
    taken = {g["key"] for g in labmod.tree(exp_id, sid)}
    kept = {g["key"] for g in store.state(exp_id, "gates")}
    qc = {g.key for g in template(exp_id).gates if getattr(g, "qc", False)}
    # a second file reviewed in the same project gets "(model 2)", so each kept gate stays tied to its file
    n = 1
    kept_names = {k.split("/")[-1] for k in kept}
    while any(f"{g['gate'].split('/')[-1]} {_suffix(n)}" in kept_names for g in body["gates"]):
        n += 1
    gates, first, placed = [], None, {}
    for g in body["gates"]:
        raw = (g.get("polygon") or {}).get("raw")
        # under FlowJo's gate of that name when the files have one, else under the model's own (raw files)
        parent = g["parent"] if not g["parent"] or g["parent"] in taken else placed.get(g["parent"])
        if not raw or g["gate"] in qc or (g["parent"] and not parent):
            continue
        name = f"{g['gate'].split('/')[-1]} {_suffix(n)}"
        key = f"{parent}/{name}" if parent else name
        placed[g["gate"]] = key
        pts = np.asarray(raw, dtype=float)
        cols = [xf[c].apply(pts[:, j]) if xf.get(c) is not None else pts[:, j] for j, c in enumerate((g["x"], g["y"]))]
        gates.append({"key": key, "name": name, "parent": parent or None, "source": "user", "kind": "polygon",
                      "dims": [g["x"], g["y"]], "vertices": [[float(a), float(b)] for a, b in zip(*cols)],
                      "reviewed_on": sid})
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


def taught(lab: str) -> dict:
    """What a lab has been taught so far: the kept files and the individuals they came from."""
    from flow_er.ml import finetune
    kept = finetune.kept_samples(lab)
    return {"samples": len(kept), "individuals": sorted({p for p in kept.values() if p})}


def lab_status(lab: str) -> dict:
    """live / outdated / offline for the start page's flower; offline too when PyTorch is missing."""
    if not status()["available"]:
        return {"state": "offline", "why": "the gating models need PyTorch (uv sync)", "version": None}
    return _ml().lab_status(lab)


def _scientists_polygons(exp_id: str, sid: str) -> dict[str, list[list[float]]]:
    """The scientist's own version of each model gate on this file: the "(model)" gates reviewed on
    it, as they were kept or left in the draft (moved or not), back in the models' display units, by
    template key. Gates reviewed on another file never count here."""
    from flow_er.cytometry.io import FcsFile
    from flow_er.ml import images
    gates = list(store.state(exp_id, "gates"))
    d = store.draft(exp_id)
    if d and d["kind"] == "gate":
        gates += d["payload"] if isinstance(d["payload"], list) else [d["payload"]]
    mine = [g for g in gates if MODEL_NAME.search(str(g.get("name", ""))) and g.get("kind") == "polygon"
            and g.get("reviewed_on") == sid]
    if not mine:
        return {}
    ds = labmod.dataset(exp_id)
    xf = ds._transforms(sid)
    fcs = FcsFile(ds.sample_path(sid))
    out = {}
    for g in mine:
        key = "/".join(MODEL_NAME.sub("", part) for part in g["key"].split("/"))
        x, y = g["dims"]
        v = np.asarray(g["vertices"], dtype=float)
        raw = np.column_stack([xf[c].inverse(v[:, j]) if xf.get(c) is not None else v[:, j] for j, c in enumerate((x, y))])
        out[key] = images.to_display(raw, [x, y], [fcs.pnr[x], fcs.pnr[y]]).tolist()
    return out


def accept(exp_id: str, sid: str, gates: dict[str, list[list[float]]] | None = None) -> dict:
    """The scientist keeps a file's gates: they become the lab's training examples, under the
    individual the file came from. Each gate is taken as the scientist left it: given in `gates`,
    else their kept or drafted "(model)" gate from review (moved or not), else the model's own."""
    ml = _ml()
    lab = lab_for(exp_id)
    if not lab:
        raise ValueError("this experiment has no lab to learn from; its gates match no lab template")
    meta = samplemeta.of(exp_id, sid)
    res = _live(exp_id, sid)
    theirs = gates or _scientists_polygons(exp_id, sid)
    accepted = {k: g.polygon_display.tolist() for k, g in res.gates.items() if g.polygon_display is not None}
    accepted.update({k: v for k, v in theirs.items() if k in res.gates})
    ds = labmod.dataset(exp_id)
    unstim = meta.get("unstim")
    # add_labels takes the unstimulated well's gating result (its cytokine gates' density), and a sample
    # id that is one safe file name: the training examples are stored as <gate>/<sample id>.npz
    unstim_res = _live(exp_id, unstim) if unstim and unstim != sid else None
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{exp_id}__{sid}")
    files = ml.add_labels(lab, str(ds.sample_path(sid)), template(exp_id),
                          {k: np.asarray(v, dtype=float) for k, v in accepted.items()},
                          meta.get("individual") or sid, sample_id=safe_id, condition=meta.get("condition") or "",
                          unstim=unstim_res)
    store.journal(exp_id, "labels_kept", {"sample": sid, "lab": lab, "gates": len(accepted), "from_scientist": len(theirs)})
    return {"lab": lab, "gates": len(accepted), "from_scientist": len([k for k in theirs if k in res.gates]),
            "files": [str(f) for f in files or []], "individual": meta.get("individual"), "taught": taught(lab)}


def versions(lab: str) -> list[dict]:
    return _ml().list_versions(lab)


def set_current(lab: str, v: str) -> dict:
    _ml().set_current(lab, v)
    return {"lab": lab, "current": v}


def retrain(lab: str, max_minutes: float | None = 10, gates: list[str] | None = None) -> dict:
    """Start retraining a lab in the background (only `gates` and those below them, if given; on the
    Colab GPU when its notebook is up, else here); progress lines and the summary land in the job.
    Refused up front with a plain reason when the lab has kept gates from fewer than 2 people."""
    people = taught(lab)["individuals"]
    if len(people) < 2:
        raise ValueError(f"lab {lab} has kept gates from {len(people)} individual(s) ({', '.join(people) or 'none'}); "
                         "retraining compares on held-out people, so teach a file from at least one more donor first")
    jid = time.strftime("%H%M%S") + "-" + uuid.uuid4().hex[:4]
    job = {"id": jid, "kind": "retrain", "lab": lab, "gates": gates, "status": "running", "started": store.now(), "lines": [], "summary": None}
    with _lock:
        _jobs[jid] = job

    def work():
        try:
            job["summary"] = _ml().retrain(lab, gates=gates, max_minutes=max_minutes, on="auto",
                                           log=lambda line: job["lines"].append(str(line)))
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


def _gate_key(exp_id: str, gate: str) -> str | None:
    """A gate named the way a scientist says it ("CD8", "IFNy of CD8", a full path) as the template's
    key: the key that holds the most of the asked words, the shortest on a tie (so "CD8" is the CD8
    gate, not one below it)."""
    keys = [g.key for g in template(exp_id).gates]
    words = lambda t: set(re.findall(r"[a-z0-9]+", MODEL_NAME.sub("", t).lower())) - {"of", "the", "gate", "in", "on"}
    want = words(gate)
    if not want:
        return None
    exact = [k for k in keys if words(k.split("/")[-1]) == want or k == gate]
    if len(exact) == 1:
        return exact[0]
    score = {k: len(want & words(k)) for k in keys}
    best = max(score.values())
    return min((k for k in keys if score[k] == best), key=len) if best else None


def train(exp_id: str, gate: str | None) -> dict:
    """The agent's train_model: retrain the experiment's lab on the kept gates, just `gate` and the
    gates below it when one is named (a leaf takes 5-8 min on the CPU), else every gate with new
    kept samples."""
    lab = lab_for(exp_id)
    if not lab:
        return {"queued": False, "message": "This experiment's gates match no lab template, so there is no model to train."}
    key = _gate_key(exp_id, gate) if gate else None
    if gate and not key:
        return {"queued": False, "message": f"No gate called {gate} in lab {lab}'s template."}
    try:
        j = retrain(lab, gates=[key] if key else None)
    except ValueError as e:
        return {"queued": False, "message": str(e), "taught": taught(lab)}
    store.journal(exp_id, "train_requested", {"gate": key, "lab": lab, "job": j["id"]})
    what = f"{key.split('/')[-1]} and the gates below it" if key else "every gate with new kept samples"
    return {"queued": True, "job": j["id"], "gates": [key] if key else None,
            "message": f"Retraining {what} for lab {lab} in the background; each gate then uses the best of its current "
                       "model, the new one and template copy on held-out people, and the version is used only if it is better."}
