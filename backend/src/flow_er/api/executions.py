"""Running workflows: start an execution (all of it, up to a step, one step, or from a step on), follow
it live, read past runs, look at what a step made, and turn the new-files trigger on or off."""
import json
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from flow_er.workspace import executor, lab, nodes, render, store

router = APIRouter(tags=["workflows"])


class RunIn(BaseModel):
    mode: str = "all"
    node: str | None = None


class ActiveIn(BaseModel):
    on: bool


def _exp(exp_id: str) -> dict:
    try:
        return store.meta(exp_id)
    except KeyError:
        raise HTTPException(404, f"no experiment {exp_id}")


@router.get("/node-types")
def node_types() -> dict:
    """The kinds of step a workflow is built from, and the sources it starts from."""
    return {"types": nodes.NODE_TYPES, "sources": nodes.SOURCES}


@router.post("/experiments/{exp_id}/executions")
def run(exp_id: str, body: RunIn) -> dict:
    _exp(exp_id)
    try:
        return executor.start(exp_id, body.mode, body.node, trigger="manual")
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/experiments/{exp_id}/executions")
def runs(exp_id: str) -> list[dict]:
    _exp(exp_id)
    return executor.history(exp_id)


@router.get("/experiments/{exp_id}/executions/{eid}")
def one(exp_id: str, eid: str) -> dict:
    _exp(exp_id)
    try:
        return executor.get(exp_id, eid)
    except KeyError:
        raise HTTPException(404, f"no execution {eid}")


@router.get("/experiments/{exp_id}/executions/{eid}/events")
def events(exp_id: str, eid: str) -> StreamingResponse:
    """Server-sent events: each step as it starts and finishes, then the end."""
    _exp(exp_id)

    def stream() -> Iterator[str]:
        for e in executor.follow(eid):
            yield f"data: {json.dumps(e, default=str)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/experiments/{exp_id}/steps/{step_id}/output")
def output(exp_id: str, step_id: str) -> dict:
    """What a step made last time it ran: its line, and the first rows of any table it wrote."""
    _exp(exp_id)
    last = next((r for r in store.state(exp_id, "runs") if r.get("step") == step_id), None)
    if not last:
        return {"step": step_id, "ran": False}
    rows = lab.read_table(exp_id, last["table"]) if last.get("table") else []
    return {"step": step_id, "ran": True, **last, "rows": rows[:50], "total_rows": len(rows)}


def _plan_step(exp_id: str, step_id: str) -> tuple[dict, list[dict]]:
    steps = store.state(exp_id, "plan")
    d = store.draft(exp_id)
    pool = steps + (d["payload"] if d and d["kind"] == "plan" else [])
    step = next((s for s in pool if s["id"] == step_id), None)
    if not step:
        raise HTTPException(404, f"no step {step_id}")
    return step, pool


def _image_ports(exp_id: str, step: dict, pool: list[dict]) -> list[tuple[str, str, object]]:
    """(name, role, folder) for each image a step makes, and for a script, each image it reads."""
    made_by = {o: s for s in pool for o in s.get("outputs", [])}
    out = []
    for o in step.get("outputs", []):
        if nodes.port_type(o, pool) == "image":
            out.append((o, "output", nodes.output_folder(exp_id, step, o)))
    for i in step.get("inputs", []):
        if i in made_by and nodes.port_type(i, pool) == "image":
            out.append((i, "input", nodes.output_folder(exp_id, made_by[i], i)))
    return out


@router.get("/experiments/{exp_id}/steps/{step_id}/images")
def images(exp_id: str, step_id: str, file: str | None = None) -> dict:
    """The density images (and regions) behind a step, for one file: what the canvas draws as a heat map."""
    _exp(exp_id)
    step, pool = _plan_step(exp_id, step_id)
    ports = [(n, r, f) for n, r, f in _image_ports(exp_id, step, pool) if f.exists()]
    files = sorted({p.stem for _, _, f in ports for p in f.glob("*.npz")})
    if not files:
        return {"step": step_id, "files": [], "images": []}
    sid = file if file in files else files[0]
    out = []
    for name, role, folder in ports:
        if (folder / f"{sid}.npz").exists():
            im = render.load(folder, sid)
            out.append({"port": name, "role": role, "kind": im["meta"].get("kind", "density"), "meta": im["meta"],
                        "grid": [[round(float(v), 2) for v in row] for row in im["grid"]]})
    return {"step": step_id, "files": files, "file": sid, "images": out}


class RegionGatesIn(BaseModel):
    file: str | None = None


@router.post("/experiments/{exp_id}/steps/{step_id}/gates")
def region_gates(exp_id: str, step_id: str, body: RegionGatesIn) -> dict:
    """The polygons a step found, as draft gates in the data lab (one per region, for one file's
    regions): the scientist sees them on the plot and keeps or undoes them, as with a drawn gate."""
    _exp(exp_id)
    step, pool = _plan_step(exp_id, step_id)
    last = {r["step"]: r for r in store.state(exp_id, "runs")}.get(step_id)
    if not last:
        raise HTTPException(400, "run this step first")
    # the table of polygons: the step's table with a vertices column
    tables = [p.stem for p in sorted(store.path(exp_id, "tables").glob(f"{step_id}_*.csv"))]
    table = next((t for t in tables if "vertices" in (lab.read_table(exp_id, t) or [{}])[0]), None)
    if not table:
        raise HTTPException(400, "this step made no polygons")
    rows = lab.read_table(exp_id, table)
    files = sorted({r["file"] for r in rows})
    sid = body.file if body.file in files else files[0]
    meta = next((render.load(f, sid)["meta"] for _, role, f in _image_ports(exp_id, step, pool)
                 if role == "input" and (f / f"{sid}.npz").exists()), None)
    if not meta:
        raise HTTPException(400, "cannot find the image these polygons were drawn on")
    taken = {g["key"] for g in lab.tree(exp_id, sid)}
    gates = []
    for r in sorted((r for r in rows if r["file"] == sid), key=lambda r: r["region"]):
        name, n = f"Region {r['region']}", 2
        while (f"{meta['parent']}/{name}" if meta["parent"] else name) in taken:
            name, n = f"Region {r['region']} ({n})", n + 1
        key = f"{meta['parent']}/{name}" if meta["parent"] else name
        gates.append({"key": key, "name": name, "parent": meta["parent"] or None, "source": "user", "kind": "polygon",
                      "dims": [meta["x"], meta["y"]],
                      "vertices": render.to_lab(exp_id, sid, meta["x"], meta["y"], json.loads(r["vertices"]))})
    d = store.set_draft(exp_id, "gate", gates, f"{len(gates)} gates found on the image in {step['title']}")
    return {"draft": d, "gates": [g["key"] for g in gates], "sample": sid, "parent": meta["parent"], "x": meta["x"], "y": meta["y"]}


@router.post("/experiments/{exp_id}/active")
def active(exp_id: str, body: ActiveIn) -> dict:
    """The new-files trigger: run the workflow when FCS files arrive in the data folder."""
    _exp(exp_id)
    return executor.watch(exp_id, body.on)
