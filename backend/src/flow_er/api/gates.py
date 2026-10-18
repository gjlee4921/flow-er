"""Gate tree, per-sample gate review, approve/edit/reset, polygon and GatingML export.

For now: the tree with counts, events to draw, and gates the scientist draws (as drafts).
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from flow_er.schemas import GateIn, QuadrantIn
from flow_er.workspace import lab, plan, store, toolkit

router = APIRouter(tags=["gates"])


def _sid(exp_id: str, sample: str | None) -> str:
    try:
        ids = lab.dataset(exp_id).sample_ids
    except KeyError:
        raise HTTPException(404, f"no experiment {exp_id}")
    if not sample:
        return ids[0]
    if sample not in ids:
        raise HTTPException(404, f"no sample {sample}")
    return sample


@router.get("/experiments/{exp_id}/gates")
def gate_tree(exp_id: str, sample: str | None = None) -> dict:
    sid = _sid(exp_id, sample)
    return {"sample": sid, "gates": lab.tree(exp_id, sid)}


@router.get("/experiments/{exp_id}/plot")
def plot(exp_id: str, x: str, y: str, sample: str | None = None, parent: str | None = None,
         stage: str = "processed") -> dict:
    sid = _sid(exp_id, sample)
    try:
        return {"sample": sid, **lab.plot(exp_id, sid, x, y, parent or None, stage)}
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))


@router.get("/toolkit")
def get_toolkit() -> dict:
    """Every plot type, statistic and gate tool the data lab has, and what is planned for later."""
    return toolkit.catalogue()


@router.get("/experiments/{exp_id}/stats")
def gate_stats(exp_id: str, gates: str = "", channels: str = "", cols: str = "") -> dict:
    """The data lab's sheet: count and share of each gate in every file, plus any toolkit statistic
    as a column. `gates` and `channels` are comma-separated ids (no gates means all); `cols` is
    a comma-separated list of stat:parameter[:arg], e.g. median:CD4 PE-Cy7 FLR-A,percentile:CD3 APC-H7 FLR-A:90,
    pct_of::Time/Singlets."""
    _sid(exp_id, None)
    keys = [g for g in gates.split(",") if g]
    chans = [c for c in channels.split(",") if c]
    columns = []
    for c in (x for x in cols.split(",") if x):
        stat, *rest = c.split(":")
        columns.append({"stat": stat, "parameter": rest[0] if rest and rest[0] else None, "arg": rest[1] if len(rest) > 1 else None})
    try:
        rows = lab.stats(exp_id, keys or None, chans, columns=columns)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    names = {c["id"]: c["short"] for c in lab.dataset(exp_id).channels()}
    labels = [{"spec": spec, "label": toolkit.column_label(c["stat"], c["parameter"], c["arg"], names)}
              for spec, c in zip((x for x in cols.split(",") if x), columns, strict=True)]
    return {"rows": rows, "columns": labels}


@router.post("/experiments/{exp_id}/gates")
def draw_gate(exp_id: str, body: GateIn) -> dict:
    """A gate the scientist drew: staged as a draft, counted straight away."""
    sid = _sid(exp_id, None)
    names = {g["key"] for g in lab.tree(exp_id, sid)}
    if body.parent and body.parent not in names:
        raise HTTPException(400, f"no parent gate {body.parent}")
    key = f"{body.parent}/{body.name}" if body.parent else body.name
    if key in names:
        raise HTTPException(409, f"a gate called {body.name} already sits under that parent")
    if body.kind == "range" and (len(body.dims) != 1 or len(body.vertices) != 2):
        raise HTTPException(400, "a range gate is one parameter with a low and a high level")
    if body.kind != "range" and (len(body.dims) != 2 or len(body.vertices) < 3):
        raise HTTPException(400, "a gate is two parameters and at least three corners")
    payload = {"key": key, "name": body.name, "parent": body.parent, "source": "user", "dims": body.dims,
               "vertices": body.vertices, "kind": body.kind}
    d = store.set_draft(exp_id, "gate", payload, f"A gate you drew: {body.name}")
    return {"draft": d, "gate": next(g for g in lab.tree(exp_id, sid) if g["key"] == key)}


@router.post("/experiments/{exp_id}/gates/quadrant")
def draw_quadrant(exp_id: str, body: QuadrantIn) -> dict:
    """Four gates at once, one per quarter around the centre the scientist clicked: staged together as
    one draft, named by which side of each parameter they sit (e.g. CD4+ CD8−)."""
    sid = _sid(exp_id, None)
    names = {g["key"] for g in lab.tree(exp_id, sid)}
    if body.parent and body.parent not in names:
        raise HTTPException(400, f"no parent gate {body.parent}")
    cx, cy = (min(0.999, max(0.001, v)) for v in body.centre)
    short = {c["id"]: c["short"] for c in lab.dataset(exp_id).channels()}
    sx, sy = (short.get(d, d) for d in body.dims)
    labels = body.names or [f"{sx}− {sy}+", f"{sx}+ {sy}+", f"{sx}+ {sy}−", f"{sx}− {sy}−"]
    boxes = [[[0, cy], [cx, cy], [cx, 1], [0, 1]], [[cx, cy], [1, cy], [1, 1], [cx, 1]],
             [[cx, 0], [1, 0], [1, cy], [cx, cy]], [[0, 0], [cx, 0], [cx, cy], [0, cy]]]
    payload = []
    for label, box in zip(labels, boxes, strict=True):
        key = f"{body.parent}/{label}" if body.parent else label
        if key in names:
            raise HTTPException(409, f"a gate called {label} already sits under that parent")
        payload.append({"key": key, "name": label, "parent": body.parent, "source": "user", "dims": body.dims,
                        "vertices": [[round(a, 4), round(b, 4)] for a, b in box], "kind": "polygon"})
    d = store.set_draft(exp_id, "gate", payload, f"A quadrant you drew: {', '.join(labels)}")
    tree = {g["key"]: g for g in lab.tree(exp_id, sid)}
    return {"draft": d, "gates": [tree[p["key"]] for p in payload]}


class ParamsIn(BaseModel):
    params: dict


@router.post("/experiments/{exp_id}/steps/{step_id}/params")
def set_params(exp_id: str, step_id: str, body: ParamsIn) -> dict:
    """The scientist changed a step's settings on the canvas: a draft of the plan, kept or undone."""
    _sid(exp_id, None)
    d = store.draft(exp_id)
    steps = [dict(s) for s in (d["payload"] if d and d["kind"] == "plan" else store.state(exp_id, "plan"))]
    target = next((s for s in steps if s["id"] == step_id), None)
    if not target:
        raise HTTPException(404, f"no step {step_id}")
    target["params"] = {**(target.get("params") or {}), **body.params}
    try:
        clean = plan.validate(steps)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return store.set_draft(exp_id, "plan", clean, f"Changed {target['title']}: " + ", ".join(f"{k} {v}" for k, v in body.params.items()))


@router.post("/experiments/{exp_id}/steps/{step_id}/run")
def run_step(exp_id: str, step_id: str) -> dict:
    try:
        return plan.run(exp_id, step_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
