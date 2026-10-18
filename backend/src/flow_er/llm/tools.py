"""The tools the agent can call. The same registry backs the MCP server.

Each handler takes a context (the experiment and the list of UI events for this turn) and returns
a small JSON-able dict. Handlers that change anything stage a draft through `store.set_draft`;
nothing lands in the experiment's state until `keep`.
"""
from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flow_er.llm import client
from flow_er.workspace import (
    data,
    executor,
    image_steps,
    lab,
    memory,
    models,
    nodes,
    plan,
    samplemeta,
    scripts,
    store,
    toolkit,
    views,
)


@dataclass
class Ctx:
    exp_id: str
    ui: list[dict] = field(default_factory=list)
    # what the scientist wrote this turn: the only words a memory can be made from
    user_text: str = ""
    # when set, each screen event is also sent straight away, while the tool is still running
    live: Callable[[dict], None] | None = None

    def emit(self, **event: Any) -> None:
        self.ui.append(event)
        if self.live:
            self.live(event)


@dataclass
class Tool:
    name: str
    description: str
    params: dict
    required: list[str]
    handler: Callable[..., dict]

    def schema(self) -> dict:
        return client.tool_schema(self.name, self.description, self.params, self.required)


REGISTRY: dict[str, Tool] = {}


def tool(name: str, description: str, params: dict | None = None, required: list[str] | None = None):
    def wrap(fn):
        REGISTRY[name] = Tool(name, description, params or {}, required or [], fn)
        return fn
    return wrap


def call(ctx: Ctx, name: str, args: dict) -> dict:
    t = REGISTRY.get(name)
    if not t:
        return {"ok": False, "error": f"no tool called {name}"}
    try:
        return {"ok": True, **t.handler(ctx, **args)}
    except (ValueError, KeyError, StopIteration) as e:
        return {"ok": False, "error": str(e) or e.__class__.__name__}


# ---- helpers ------------------------------------------------------------------------------

def _channel(ctx: Ctx, name: str | None) -> str | None:
    if not name:
        return None
    chans = lab.dataset(ctx.exp_id).channels()
    for c in chans:
        if name in (c["id"], c["short"], c["label"]):
            return c["id"]
    low = name.lower()
    for c in chans:
        if c["short"].lower() == low or c["id"].lower().startswith(low):
            return c["id"]
    raise ValueError(f"no channel '{name}'; channels are {[c['short'] for c in chans]}")


def _sample(ctx: Ctx, name: str | None) -> str:
    ids = lab.dataset(ctx.exp_id).sample_ids
    if not name:
        return ids[0]
    for s in ids:
        if name == s or name in s:
            return s
    raise ValueError(f"no sample '{name}'; samples are {ids}")


def _gate(ctx: Ctx, sid: str, name: str | None) -> str | None:
    if not name or name.lower() in ("all", "all events", "root"):
        return None
    t = lab.tree(ctx.exp_id, sid)
    exact = [g for g in t if g["key"] == name]
    if exact:
        return exact[0]["key"]
    named = [g for g in t if g["name"].lower() == name.lower()]
    if len(named) == 1:
        return named[0]["key"]
    if len(named) > 1:
        raise ValueError(f"'{name}' is in more than one place; use one of {[g['key'] for g in named]}")
    raise ValueError(f"no gate '{name}'; gates are {[g['key'] for g in t]}")


def snapshot(exp_id: str) -> dict:
    """What the agent needs to know at the start of a turn, kept compact."""
    m = store.meta(exp_id)
    ds = lab.dataset(exp_id)
    runs = {r["step"]: r["status"] for r in store.state(exp_id, "runs")}
    d = store.draft(exp_id)
    return {
        "title": m.get("title"), "question": m.get("question") or None,
        "data_folder": m.get("data_path", "").split("/")[-1],
        "samples": [{"id": s["id"], "events": s["events"]} for s in ds.samples()],
        # who each file came from (kept, or read from file names until kept): the gate models need it
        "sample_meta": [{k: r[k] for k in ("sample", "individual", "condition", "unstim", "kept")} for r in samplemeta.get(exp_id)],
        "channels": [c["short"] for c in ds.channels()],
        "flowjo_workspace": ds.wsp_path.name if ds.wsp_path else None,
        "plan": [{"id": s["id"], "op": s["op"], "title": s["title"], "run": runs.get(s["id"], "not run")}
                 for s in store.state(exp_id, "plan")],
        "user_gates": [g["name"] for g in store.state(exp_id, "gates")],
        "views": [{"id": v["id"], "title": v["title"]} for v in store.state(exp_id, "views")],
        "macros": [{"id": x["id"], "title": x["title"]} for x in store.state(exp_id, "macros")],
        "tables": lab.tables(exp_id),
        "uploads": [p.name for p in store.path(exp_id, "uploads").iterdir()],
        "draft": {"kind": d["kind"], "summary": d["summary"]} if d else None,
        "gating_models": models.status()["message"],
    }


# ---- reading ------------------------------------------------------------------------------

@tool("status", "The experiment as it stands: samples, channels, plan and what has run, gates, views, tables, the waiting draft.")
def _status(ctx: Ctx) -> dict:
    return snapshot(ctx.exp_id)


@tool("describe_gates", "The gate tree for one sample with counts, % of parent and the two channels each gate is drawn on.",
      {"sample": {"type": "string", "description": "File name or part of it; defaults to the first sample."}})
def _describe(ctx: Ctx, sample: str | None = None) -> dict:
    sid = _sample(ctx, sample)
    chans = {c["id"]: c["short"] for c in lab.dataset(ctx.exp_id).channels()}
    rows = [{"key": g["key"], "name": g["name"], "parent": g["parent"], "source": g["source"], "count": g["count"],
             "pct_parent": g["pct_parent"], "on": [chans.get(d, d) for d in g["dims"]], "draft": g["draft"]}
            for g in lab.tree(ctx.exp_id, sid)]
    return {"sample": sid, "gates": rows}


@tool("population_stats", "Rows from the saved population table (every gate in every sample). Computes it if it is missing.",
      {"gate": {"type": "string", "description": "Gate name or path to filter on."},
       "sample": {"type": "string", "description": "Sample to filter on."},
       "refresh": {"type": "boolean", "description": "Recount from the data first."}})
def _pop(ctx: Ctx, gate: str | None = None, sample: str | None = None, refresh: bool = False) -> dict:
    rows = lab.read_table(ctx.exp_id, "population_stats")
    if refresh or not rows:
        rows = lab.population_table(ctx.exp_id)
        ctx.emit(type="refresh")
    if gate:
        rows = [r for r in rows if r["gate"].lower() == gate.lower() or r["gate_path"] == gate]
    if sample:
        rows = [r for r in rows if sample in r["sample"]]
    return {"table": "population_stats", "rows": rows[:80], "total_rows": len(rows)}


@tool("read_attachment", "Read a file the scientist added in the chat: what kind it is and what is in it.",
      {"name": {"type": "string"}}, ["name"])
def _read_attachment(ctx: Ctx, name: str) -> dict:
    p = store.path(ctx.exp_id, "uploads", name)
    if not p.exists():
        raise ValueError(f"no upload called {name}")
    ext = p.suffix.lower()
    info: dict = {"name": name, "bytes": p.stat().st_size}
    if ext == ".fcs":
        import flowkit as fk
        s = fk.Sample(str(p))
        info.update(kind="fcs", events=int(s.event_count), channels=list(s.pnn_labels))
    elif ext in (".csv", ".tsv", ".txt", ".md"):
        info.update(kind="text", head=p.read_text(errors="replace")[:3000])
    elif ext == ".wsp":
        info.update(kind="flowjo_workspace", note="A FlowJo workspace: its gates can become labels.")
    else:
        info.update(kind="other", note="Kept with the experiment; not read.")
    return info


# ---- changing things (drafts) ---------------------------------------------------------------

@tool("set_question", "Record what the experiment is for, and optionally rename it. Not a draft: it is a note.",
      {"question": {"type": "string"}, "title": {"type": "string", "description": "A short name, 2-5 words."}}, ["question"])
def _question(ctx: Ctx, question: str, title: str | None = None) -> dict:
    m = store.update_meta(ctx.exp_id, question=question, title=title)
    ctx.emit(type="refresh")
    return {"title": m["title"], "question": m["question"]}


_STEP = {"type": "object", "properties": {
    "id": {"type": "string"}, "op": {"type": "string", "enum": sorted(plan.OPS)}, "title": {"type": "string"},
    "inputs": {"type": "array", "items": {"type": "string"}}, "outputs": {"type": "array", "items": {"type": "string"}},
    "why": {"type": "string"}, "gate": {"type": "string"}, "macro_id": {"type": "string"},
    "params": {"type": "object", "additionalProperties": True,
               "description": "The step's settings. A view: chart (bar, dot, dumbbell, line, scatter, heatmap, table), x, y, color "
                              "(columns of the table it reads) and title, so the run draws the figure. render_image: x and y channels."}},
    "required": ["id", "op", "title"]}


@tool("plan_pipeline",
      "Propose the whole analysis as ordered steps; it becomes a draft the scientist keeps or undoes. Ops: "
      + "; ".join(f"{k} ({v[0]}): {v[1]}" for k, v in plan.OPS.items())
      + ". Inputs must be 'fcs', 'wsp', 'uploads' or an output of an earlier step.",
      {"steps": {"type": "array", "items": _STEP}}, ["steps"])
def _plan(ctx: Ctx, steps: list[dict]) -> dict:
    clean = plan.validate(steps)
    d = store.set_draft(ctx.exp_id, "plan", clean, f"A plan with {len(clean)} steps")
    ctx.emit(type="open", page="flow")
    ctx.emit(type="refresh")
    return {"draft": d["id"], "steps": [{"id": s["id"], "kind": s["kind"], "title": s["title"]} for s in clean]}


@tool("edit_pipeline", "Add or drop one step, or change one step's settings (set); the result is a draft of the whole plan.",
      {"action": {"type": "string", "enum": ["add", "drop", "set"]}, "step": _STEP,
       "step_id": {"type": "string", "description": "For drop or set: the step."},
       "params": {"type": "object", "description": "For set: the step's new settings, e.g. {\"value\": \"count\"} for Export for Prism."},
       "after": {"type": "string", "description": "For add: the step it goes after; at the end if omitted."}},
      ["action"])
def _edit(ctx: Ctx, action: str, step: dict | None = None, step_id: str | None = None, after: str | None = None,
          params: dict | None = None) -> dict:
    d = store.draft(ctx.exp_id)
    current = d["payload"] if d and d["kind"] == "plan" else store.state(ctx.exp_id, "plan")
    steps = [dict(s) for s in current]
    if action == "drop":
        users = [s["id"] for s in steps if step_id in s.get("inputs", []) or
                 set(s.get("inputs", [])) & set(next((x.get("outputs", []) for x in steps if x["id"] == step_id), []))]
        if users:
            raise ValueError(f"steps {users} read what '{step_id}' makes; drop or change them first")
        steps = [s for s in steps if s["id"] != step_id]
    elif action == "add" and step:
        i = next((n + 1 for n, s in enumerate(steps) if s["id"] == after), len(steps))
        steps.insert(i, step)
    elif action == "set":
        target = next((s for s in steps if s["id"] == step_id), None)
        if not target:
            raise ValueError(f"no step '{step_id}' in the plan")
        target["params"] = {**(target.get("params") or {}), **(params or {})}
    clean = plan.validate(steps)
    d = store.set_draft(ctx.exp_id, "plan", clean, f"Plan: {action} {step_id or (step or {}).get('id', '')}")
    ctx.emit(type="open", page="flow")
    ctx.emit(type="refresh")
    return {"draft": d["id"], "steps": [s["id"] for s in clean]}


@tool("run_step", "Run one step of the kept plan.", {"step_id": {"type": "string"}}, ["step_id"])
def _run(ctx: Ctx, step_id: str) -> dict:
    r = plan.run(ctx.exp_id, step_id)
    ctx.emit(type="refresh")
    if r.get("table"):
        ctx.emit(type="open", page="measure")
    return r


@tool("write_view",
      "Draft a figure over a saved table. Types: " + "; ".join(f"{k}: {v}" for k, v in views.TYPES.items())
      + ". The browser draws it and makes it interactive. filter is {column: value or [values]}.",
      {"title": {"type": "string"}, "caption": {"type": "string"}, "type": {"type": "string", "enum": sorted(views.TYPES)},
       "table": {"type": "string"}, "x": {"type": "string"}, "y": {"type": "string"}, "color": {"type": "string"},
       "filter": {"type": "object", "additionalProperties": True},
       "agg": {"type": "string", "enum": sorted(views.AGG)}, "sort": {"type": "string", "enum": ["none", "asc", "desc"]},
       "log": {"type": "boolean"}, "y_label": {"type": "string"}, "x_label": {"type": "string"}},
      ["title", "type", "table"])
def _view(ctx: Ctx, **spec: Any) -> dict:
    v = views.validate(ctx.exp_id, spec)
    rows = views.data(ctx.exp_id, v)
    d = store.set_draft(ctx.exp_id, "view", v, f"A {v['type']} view: {v['title']}")
    ctx.emit(type="open", page="results")
    ctx.emit(type="refresh")
    return {"draft": d["id"], "view": v["id"], "rows_drawn": len(rows)}


@tool("write_macro",
      "Add a script step for something no built-in op covers (a QC rule, a ratio, a custom statistic). A subagent finds "
      "reference code on the allowed sites, adapts it (only MIT, BSD, Apache-2.0, ISC or PSF sources), and tests it in the "
      "sandbox; the tested step joins the plan as a draft. inputs: the outputs it reads (fcs for the files, or an earlier "
      "step's output that is a table). If the source's licence is refused, say why and offer to write it from scratch.",
      {"goal": {"type": "string", "description": "What the step must do, in the scientist's terms, with any thresholds they gave."},
       "inputs": {"type": "array", "items": {"type": "string"}},
       "after": {"type": "string", "description": "The step it goes after; by default after the last step it reads from."}},
      ["goal"])
def _macro(ctx: Ctx, goal: str, inputs: list[str] | None = None, after: str | None = None) -> dict:
    from flow_er.llm import script_writer
    d = store.draft(ctx.exp_id)
    steps = [dict(s) for s in (d["payload"] if d and d["kind"] == "plan" else store.state(ctx.exp_id, "plan"))]
    inputs = inputs or ["fcs"]
    ports = {}
    for name in inputs:
        kind = nodes.port_type(name, steps)
        if kind in ("samples", "events"):
            ports[re.sub(r"\W+", "_", name)] = "events"
        elif kind in ("table", "image"):
            ports[re.sub(r"\W+", "_", name)] = kind
        else:
            raise ValueError(f"a script reads the files' events, a table or an image; '{name}' is {kind}")
    chans = [c["id"] for c in lab.dataset(ctx.exp_id).channels()]
    # each table it reads: its columns (and a few real rows when the table exists), so the fixture is true to it
    made_by = {o: s for s in steps for o in s.get("outputs", [])}
    tables = {}
    for (port, kind), name in zip(ports.items(), inputs):
        if kind == "table":
            last = next((r.get("table") for r in store.state(ctx.exp_id, "runs") if r["step"] == made_by.get(name, {}).get("id")), None)
            rows = lab.read_table(ctx.exp_id, last or name)
            cols = nodes.NODE_TYPES.get(made_by.get(name, {}).get("op", ""), {}).get("columns")
            tables[port] = {"columns": cols or (list(rows[0]) if rows else None), "example_rows": rows[:6]}
    # and the real inputs, to try each passing script on before it is offered
    work = Path(tempfile.mkdtemp(prefix="flower-writer-"))
    real_in: dict = {}

    def real(code: str, params: dict) -> str | None:
        if "inputs" not in real_in:
            real_in["inputs"] = scripts.real_inputs(ctx.exp_id, ports, inputs, steps, work)
        return scripts.try_real(code, real_in["inputs"], params) if real_in["inputs"] else None

    # what the writer is doing, shown live under this step (not kept with the message)
    progress = (lambda text: ctx.live({"type": "progress", "text": text})) if ctx.live else None
    try:
        out = script_writer.write(goal, ports, chans, on_progress=progress, tables=tables, real=real)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if out.get("refused"):
        return {"refused": out["refused"], "sources": out.get("fetched", [])}
    sid = re.sub(r"[^a-z0-9]+", "_", out["title"].lower()).strip("_")[:28] or "script"
    while any(s["id"] == sid for s in steps):
        sid += "_2"
    step = {"id": sid, "op": "custom_macro", "title": out["title"], "why": out["summary"], "inputs": inputs,
            "outputs": [f"{sid}_{o}" for o in out["outputs"]],
            "script": {k: out[k] for k in ("code", "provenance", "fixture", "test")}}
    made = {o: n for n, s in enumerate(steps) for o in s.get("outputs", [])}
    at = next((n + 1 for n, s in enumerate(steps) if s["id"] == after), None)
    steps.insert(at if at is not None else max([made[i] + 1 for i in inputs if i in made], default=0), step)
    clean = plan.validate(steps)
    prov = out["provenance"]
    d = store.set_draft(ctx.exp_id, "plan", clean, f"Added a script step: {out['title']} (passed its test in the sandbox)")
    ctx.emit(type="open", page="flow")
    ctx.emit(type="refresh")
    return {"draft": d["id"], "step": sid, "title": out["title"], "summary": out["summary"],
            "source": prov.get("source") or "written for this step", "licence": prov.get("licence"),
            "test": {"passed": True, "same_output_twice": True, "seconds": out["test"].get("seconds")},
            "outputs": step["outputs"]}


@tool("add_image_steps",
      "Gate on a picture: add three steps that draw one population's events on two channels as a density image (the "
      "gate models' recipe), find the populations on it (peaks and watershed with scikit-image; a gate model can take this "
      "step's place later), and turn each region into a gate polygon with its event count. Tested, as one draft. The "
      "regions become gates only when the scientist opens them in the data lab and keeps them.",
      {"parent": {"type": "string", "description": "The population to draw, e.g. CD3+"},
       "x": {"type": "string"}, "y": {"type": "string"}}, ["parent", "x", "y"])
def _image_steps(ctx: Ctx, parent: str, x: str, y: str) -> dict:
    d = store.draft(ctx.exp_id)
    steps = [dict(s) for s in (d["payload"] if d and d["kind"] == "plan" else store.state(ctx.exp_id, "plan"))]
    ds = lab.dataset(ctx.exp_id)
    key = _gate(ctx, ds.sample_ids[0], parent) or ""
    xc, yc = _channel(ctx, x), _channel(ctx, y)
    short = {c["id"]: c["short"] for c in ds.channels()}
    new = image_steps.steps(key, xc, yc, short, {s["id"] for s in steps})
    labels = next((o for s in steps if s["op"] == "gate_from_workspace" for o in s.get("outputs", [])), None)
    if labels:  # drawn after the gates are applied, so the canvas shows what it depends on
        new[0]["inputs"] = [labels]
    at = next((n + 1 for n, s in enumerate(steps) if labels in s.get("outputs", [])), len(steps))
    clean = plan.validate(steps[:at] + new + steps[at:])
    d = store.set_draft(ctx.exp_id, "plan", clean, f"Image steps: {new[0]['title']}")
    ctx.emit(type="open", page="flow")
    ctx.emit(type="refresh")
    return {"draft": d["id"], "steps": [s["id"] for s in new], "population": key or "all events", "x": xc, "y": yc,
            "summary": "Three tested steps: render the density image, find the populations, turn regions into gates and counts."}


@tool("add_autogate_step",
      "Automate the gating: add a step that gates every file with the lab's trained models (a per-gate U-Net where the lab "
      "has one, else its template), giving per gate the count, % of parent, which model answered, how sure it was, and "
      "FlowJo's % and an F1 where FlowJo gates exist. As a draft. The model gates are reviewed and kept by the scientist "
      "in the data lab; you never keep them.")
def _autogate_step(ctx: Ctx) -> dict:
    d = store.draft(ctx.exp_id)
    steps = [dict(s) for s in (d["payload"] if d and d["kind"] == "plan" else store.state(ctx.exp_id, "plan"))]
    if any(s["op"] == "autogate" for s in steps):
        raise ValueError("the plan already has an autogate step")
    lab_name = models.lab_for(ctx.exp_id)
    labels = next((o for s in steps if s["op"] == "gate_from_workspace" for o in s.get("outputs", [])), None)
    inputs = [labels] if labels else ["fcs"]
    step = {"id": "autogate", "op": "autogate", "title": f"Gate every file with {'lab ' + lab_name + chr(39) + 's models' if lab_name else 'the template'}",
            "why": "Automates the manual gating; unsure gates are flagged for review.", "inputs": inputs, "outputs": ["autogate_counts"]}
    at = next((n + 1 for n, s in enumerate(steps) if labels in s.get("outputs", [])), len(steps))
    clean = plan.validate(steps[:at] + [step] + steps[at:])
    d = store.set_draft(ctx.exp_id, "plan", clean, f"Autogate with {lab_name or 'the template'} {models.version(ctx.exp_id)}")
    ctx.emit(type="open", page="flow")
    ctx.emit(type="refresh")
    return {"draft": d["id"], "lab": lab_name, "version": models.version(ctx.exp_id),
            "summary": "An autogate step is waiting to be kept; after a run, its gates are reviewed in the data lab."}


@tool("read_model", "The gating models this experiment uses: the lab, the version answering, every version with its "
      "held-out metrics, and how the last autogate run went per gate (which model, confidence, F1 against FlowJo).")
def _read_model(ctx: Ctx) -> dict:
    lab_name = models.lab_for(ctx.exp_id)
    out = {"lab": lab_name, "version": models.version(ctx.exp_id), "status": models.status()["message"]}
    if lab_name:
        out["versions"] = [{k: v.get(k) for k in ("version", "current", "created", "notes", "metrics")} for v in models.versions(lab_name)]
    if store.path(ctx.exp_id, "tables", "autogate_counts.csv").exists():
        rows = lab.read_table(ctx.exp_id, "autogate_counts")
        out["last_run"] = rows[:60]
        out["summary"] = f"{len(rows)} gate results in autogate_counts"
    return out


@tool("train_model", "Retrain the experiment's lab models on the gates the scientist kept, as a background job. A new version "
      "is used only if it is better on held-out individuals; code decides, you explain.",
      {"gate": {"type": "string"}})
def _train(ctx: Ctx, gate: str | None = None) -> dict:
    return models.train(ctx.exp_id, gate)


@tool("keep", "Keep the waiting draft: it becomes part of the experiment.")
def _keep(ctx: Ctx) -> dict:
    d = store.keep(ctx.exp_id, who="agent, asked by the scientist")
    if d:
        lab.after_change(ctx.exp_id, d["kind"])
    ctx.emit(type="refresh")
    return {"kept": d["summary"] if d else None}


@tool("undo", "Drop the waiting draft. Nothing kept changes.")
def _undo(ctx: Ctx) -> dict:
    d = store.undo(ctx.exp_id, who="agent, asked by the scientist")
    if d:
        lab.after_change(ctx.exp_id, d["kind"])
    ctx.emit(type="refresh")
    return {"dropped": d["summary"] if d else None}


# ---- the screen -----------------------------------------------------------------------------

PAGES = ["flow", "lab", "results", "measure", "models", "clean", "data"]


@tool("open_page", "Show a page in the panel: flow (the pipeline), lab, results, measure, models, clean, data.",
      {"page": {"type": "string", "enum": PAGES}}, ["page"])
def _open(ctx: Ctx, page: str) -> dict:
    ctx.emit(type="open", page=page)
    return {"opened": page}


@tool("focus_lab",
      "Point the data lab at a sample, a parent population and two channels, and choose how to draw it. "
      "stage: raw (no compensation), processed, or labels (coloured by the gates below the parent).",
      {"sample": {"type": "string"}, "parent": {"type": "string", "description": "Gate name or path; omit for all events."},
       "x": {"type": "string", "description": "Channel or marker, e.g. CD4"}, "y": {"type": "string"},
       "stage": {"type": "string", "enum": ["raw", "processed", "labels"]},
       "mode": {"type": "string", "enum": ["density", "dots"]}})
def _focus(ctx: Ctx, sample: str | None = None, parent: str | None = None, x: str | None = None,
           y: str | None = None, stage: str | None = None, mode: str | None = None) -> dict:
    sid = _sample(ctx, sample)
    key = _gate(ctx, sid, parent)
    ev = {"type": "lab", "sample": sid, "parent": key, "x": _channel(ctx, x), "y": _channel(ctx, y),
          "stage": stage, "mode": mode}
    ctx.emit(**{k: v for k, v in ev.items() if v is not None})
    ctx.emit(type="open", page="lab")
    return {"sample": sid, "parent": key, "x": ev["x"], "y": ev["y"]}


# ---- running workflows -------------------------------------------------------------------

@tool("run_workflow",
      "Run the kept plan as a workflow: all of it, up to a step, only one step, or from a step on. Steps whose "
      "inputs and settings have not changed are reused (cached), so running again is cheap. Returns each step's result.",
      {"mode": {"type": "string", "enum": list(executor.MODES)}, "step_id": {"type": "string", "description": "For to, only or from"}})
def _run_workflow(ctx: Ctx, mode: str = "all", step_id: str | None = None) -> dict:
    ctx.emit(type="open", page="flow")
    started = executor.start(ctx.exp_id, mode, step_id, trigger="agent")
    ctx.emit(type="execution", id=started["id"])  # the canvas follows the run live while this waits
    rec = executor.wait_for(ctx.exp_id, started["id"])
    ctx.emit(type="refresh")
    steps = [{"step": k, "status": n["status"], "summary": n.get("summary", "")} for k, n in rec["nodes"].items()]
    counts = {s: sum(1 for x in steps if x["status"] == s) for s in ("done", "cached", "failed", "blocked")}
    said = {"done": "ran", "cached": "reused (nothing they read had changed)", "failed": "failed", "blocked": "waiting on a failed step"}
    return {"summary": f"Run {rec['status']}: " + ", ".join(f"{v} {said[k]}" for k, v in counts.items() if v), "execution": rec["id"], "steps": steps}


@tool("read_execution", "What happened in a workflow run (the latest if no id): each step's status, line and table.",
      {"id": {"type": "string"}})
def _read_execution(ctx: Ctx, id: str | None = None) -> dict:
    recs = executor.history(ctx.exp_id, 1) if not id else [executor.get(ctx.exp_id, id)]
    if not recs:
        raise ValueError("the workflow has not been run yet")
    r = recs[0]
    return {"summary": f"Run {r['id']} ({r['trigger']}): {r['status']}", "execution": r}


# ---- memory: what the scientist tells flow-er about how they work ----------------------------

@tool("remember",
      "Keep something the scientist just told you about how they work, for every project from now on: a definition "
      "(how they define a population), a preference (how they report or look at things), a lab convention, or a fact "
      "about this project. `quote` must be their exact words from this message. Never store data, file contents or results.",
      {"text": {"type": "string", "description": "The memory in one short sentence, e.g. 'Tregs are CD25 high CD127 low'"},
       "kind": {"type": "string", "enum": list(memory.KINDS)},
       "quote": {"type": "string", "description": "Their exact words from this message that say it"}}, ["text", "kind", "quote"])
def _remember(ctx: Ctx, text: str, kind: str, quote: str) -> dict:
    if memory.paused():
        raise ValueError("memory is paused; the scientist turned it off in the memory panel")
    if not memory.said(quote, ctx.user_text):
        raise ValueError("a memory can only be made from the scientist's own words in this message; quote them exactly")
    m = memory.add(text, kind, quote, {"experiment": ctx.exp_id, "title": store.meta(ctx.exp_id).get("title")})
    if m.get("already"):
        ctx.emit(type="memory", action="already", id=m["id"], text=m["text"])
        return {"summary": f"Already remembered: {m['text']}", "id": m["id"]}
    ctx.emit(type="memory", action="added", id=m["id"], text=m["text"])
    return {"summary": f"Remembered: {m['text']}", "id": m["id"]}


@tool("forget", "Forget memories, by id or by words that match them (\"what I said about Tregs\" forgets every memory about Tregs).",
      {"id": {"type": "string"}, "about": {"type": "string"}})
def _forget(ctx: Ctx, id: str | None = None, about: str | None = None) -> dict:
    gone = [x for x in memory.entries() if x["id"] == id] if id else memory.find_all(about or "")
    if not gone:
        raise ValueError("there is no memory like that")
    for m in gone:
        memory.remove(m["id"])
        ctx.emit(type="memory", action="forgotten", id=m["id"], text=m["text"])
    return {"summary": "Forgot: " + "; ".join(m["text"] for m in gone)}


@tool("search_past_experiments", "Find earlier projects by words: their titles, what they were for, their plans, figures and what was kept.",
      {"query": {"type": "string"}}, ["query"])
def _past(ctx: Ctx, query: str) -> dict:
    found = memory.past_projects(query, exclude=ctx.exp_id)
    return {"summary": f"{len(found)} earlier project(s)", "projects": found}


def memories_for_prompt(context: str) -> str:
    """Memories as the agent reads them, or nothing when there are none."""
    items = memory.for_agent(context)
    if not items:
        return ""
    lines = "\n".join(f"- [{m['id']}] ({m['kind']}) {m['text']}" for m in items)
    return f"\n\nWhat the scientist has told you about how they work (their own words, kept across projects):\n{lines}"


# ---- the data lab assistant -----------------------------------------------------------------
# These move what the data lab shows and read numbers off the events. None of them stage a draft.

MAIN_ONLY = ("remember", "forget", "search_past_experiments")
LAB_ONLY = ("show_plot", "compare_files", "select_gate", "gate_stats", "show_sheet", "fcs_keywords", "set_gate_tool", "show_figure")
LAB_TOOLS = (*LAB_ONLY, "describe_gates", "read_figure")


def lab_snapshot(exp_id: str) -> dict:
    """What the lab assistant needs about the data: files, parameters, the gate tree."""
    ds = lab.dataset(exp_id)
    sid = ds.sample_ids[0]
    return {
        "files": [{"id": s["id"], "events": s["events"]} for s in ds.samples()],
        "parameters": [{"id": c["id"], "name": c["name"], "group": c["group"]} for c in ds.channels()],
        "gates": [{"name": t["name"], "path": t["key"], "drawn_on": t["dims"]} for t in lab.tree(exp_id, sid)],
        "saved_figures": [v["title"] for v in store.state(exp_id, "views")],
        "toolkit": {"plot_types": [p["id"] for p in toolkit.PLOT_TYPES], "statistics": [s["id"] for s in toolkit.STATISTICS],
                    "gate_tools": [g["id"] for g in toolkit.GATE_TOOLS], "not_yet": toolkit.LATER},
    }


@tool("show_plot",
      "Show a plot in the data lab: one file, a population (its parent gate) and any two parameters "
      "(scatter, markers or Time). type: density, dots or histogram (histogram uses x only). "
      "stage: raw (no compensation), processed, or labels (coloured by the gates inside the population).",
      {"sample": {"type": "string", "description": "File name or part of it; omit to keep the current one."},
       "population": {"type": "string", "description": "Gate name or path whose events to plot; 'all' for all events."},
       "x": {"type": "string"}, "y": {"type": "string"},
       "type": {"type": "string", "enum": [p["id"] for p in toolkit.PLOT_TYPES], "description": "; ".join(f"{p['id']}: {p['about']}" for p in toolkit.PLOT_TYPES)},
       "stage": {"type": "string", "enum": ["raw", "processed", "labels"]}})
def _show_plot(ctx: Ctx, sample: str | None = None, population: str | None = None, x: str | None = None,
               y: str | None = None, type: str | None = None, stage: str | None = None) -> dict:
    sid = _sample(ctx, sample) if sample else None
    ev: dict[str, Any] = {"type": "lab", "view": "plot", "x": _channel(ctx, x), "y": _channel(ctx, y),
                          "stage": stage, "mode": {"histogram": "hist"}.get(type or "", type)}
    if ev["mode"] and ev["mode"] not in toolkit.PLOT_IDS:
        raise ValueError(f"no plot type '{type}'; the toolkit has {sorted(toolkit.PLOT_IDS)}")
    if sid:
        ev["sample"] = sid
    if population is not None:
        ev["parent"] = _gate(ctx, sid or lab.dataset(ctx.exp_id).sample_ids[0], population)
        ev["sel"] = None
    ctx.emit(**{k: v for k, v in ev.items() if v is not None or k in ("parent", "sel")})
    names = {c["id"]: c["name"] for c in lab.dataset(ctx.exp_id).channels()}
    where = "" if population is None else f" in {ev['parent'].split('/')[-1] if ev.get('parent') else 'all events'}"
    axes = f"{names.get(ev['y'], 'the same y')} against {names.get(ev['x'], 'the same x')}" if ev["mode"] != "hist" else names.get(ev["x"], "the same x")
    return {"summary": f"Showed {axes}{where}",
            **{k: ev.get(k) for k in ("sample", "parent", "x", "y")}}


@tool("compare_files", "Line every file up side by side on the same axes and population (on), or go back to one file (off).",
      {"on": {"type": "boolean"}}, ["on"])
def _compare(ctx: Ctx, on: bool) -> dict:
    ctx.emit(type="lab", view="plot", compare=bool(on))
    return {"summary": "Every file side by side" if on else "One file"}


@tool("select_gate", "Pick out one gate: the lab shows it on its own two parameters, in its parent population, and its numbers.",
      {"gate": {"type": "string"}, "sample": {"type": "string"}}, ["gate"])
def _select_gate(ctx: Ctx, gate: str, sample: str | None = None) -> dict:
    sid = _sample(ctx, sample)
    key = _gate(ctx, sid, gate)
    if key is None:
        raise ValueError("pick a gate, not all events")
    node = next(t for t in lab.tree(ctx.exp_id, sid) if t["key"] == key)
    ev: dict[str, Any] = {"type": "lab", "view": "plot", "sel": key, "parent": node["parent"]}
    if sample:
        ev["sample"] = sid
    if len(node["dims"]) == 2:
        ev["x"], ev["y"] = node["dims"]
    ctx.emit(**ev)
    return {"summary": f"Selected {node['name']} in {sid}", "gate": key,
            "count": node["count"], "pct_parent": node["pct_parent"]}


_STATISTIC = {"type": "object", "properties": {
    "stat": {"type": "string", "enum": [x["id"] for x in toolkit.STATISTICS]},
    "parameter": {"type": "string", "description": "Channel or marker, for statistics of a parameter"},
    "p": {"type": "number", "description": "The percentile, for percentile"},
    "ancestor": {"type": "string", "description": "Gate name or path, for pct_of"}}, "required": ["stat"]}


def _columns(ctx: Ctx, sid: str, statistics: list[dict] | None) -> list[dict]:
    out = []
    for s in statistics or []:
        if s.get("stat") not in toolkit.STAT_IDS:
            raise ValueError(f"no statistic '{s.get('stat')}'; the toolkit has {sorted(toolkit.STAT_IDS)}")
        p = s.get("p")
        arg = (f"{p:g}" if p is not None else None) or (_gate(ctx, sid, s["ancestor"]) if s.get("ancestor") else None)
        out.append({"stat": s["stat"], "parameter": _channel(ctx, s.get("parameter")), "arg": arg})
    return out


@tool("gate_stats",
      "Numbers for gates in every file, computed by code from all events: count, % of parent and of all, the "
      "events of the parent outside the gate (outside_count, pct_outside: e.g. dead cells are those outside a live gate), "
      "and any statistic in the toolkit (mean, median, geometric mean, mode, SD, CV, robust SD and CV, percentile, "
      "median absolute deviation, % of grandparent or of any ancestor) on compensated values, as FlowJo reports them.",
      {"gates": {"type": "array", "items": {"type": "string"}, "description": "Gate names or paths; empty for every gate."},
       "statistics": {"type": "array", "items": _STATISTIC},
       "parameters": {"type": "array", "items": {"type": "string"}, "description": "Shorthand: median and CV of each."}})
def _gate_stats(ctx: Ctx, gates: list[str] | None = None, statistics: list[dict] | None = None,
                parameters: list[str] | None = None) -> dict:
    sid = lab.dataset(ctx.exp_id).sample_ids[0]
    keys = [k for k in (_gate(ctx, sid, g) for g in gates or []) if k]
    chans = [c for c in (_channel(ctx, p) for p in parameters or []) if c]
    rows = lab.stats(ctx.exp_id, keys or None, chans, columns=_columns(ctx, sid, statistics))
    return {"rows": rows, "total_rows": len(rows), "table": "gate stats (not saved)"}


@tool("fcs_keywords", "The FCS file's own keywords: the instrument, when it was acquired, the event count, and each channel's name, stain and voltage.",
      {"sample": {"type": "string"}})
def _fcs_keywords(ctx: Ctx, sample: str | None = None) -> dict:
    sid = _sample(ctx, sample)
    ds = lab.dataset(ctx.exp_id)
    h = data.fcs_header(ds.sample_path(sid))
    if not h:
        raise ValueError(f"could not read the keywords of {sid}")
    k = h["keywords"]
    keep = {key: k[key] for key in ("$CYT", "$CYTSN", "$DATE", "$BTIM", "$ETIM", "$TOT", "$SRC", "$OP", "$INST", "$FIL", "EXPERIMENT NAME", "TUBE NAME") if key in k}
    chans = [{"n": p["n"], "s": p["s"], "voltage": k.get(f"$P{i}V")} for i, p in enumerate(h["params"], 1)]
    return {"sample": sid, "version": h["version"], "events": h["events"], "keywords": keep, "channels": chans,
            "summary": f"{h['version']}, {h['events']:,} events" + (f", {keep['$CYT']}" if "$CYT" in keep else "")}


@tool("set_gate_tool", "Pick the gate tool for the scientist to draw with (you never draw yourself): rect, poly, quadrant, "
      "range (on a histogram), or select to stop drawing.",
      {"tool": {"type": "string", "enum": ["select"] + [g["id"] for g in toolkit.GATE_TOOLS]}}, ["tool"])
def _set_gate_tool(ctx: Ctx, tool: str) -> dict:
    ev: dict[str, Any] = {"type": "lab", "view": "plot", "tool": tool}
    if tool == "range":
        ev["mode"] = "hist"
    ctx.emit(**ev)
    how = next((g["about"] for g in toolkit.GATE_TOOLS if g["id"] == tool), "Click a gate to pick it.")
    return {"summary": f"Ready to draw: {tool}", "how": how}


@tool("show_sheet",
      "Open the sheet (one row per gate per file) on some gates, with toolkit statistics as columns; optionally "
      "filter the rows by text, sort by a column, colour a column by its values, and chart a column as bars with each "
      "file as a dot. Columns are named as the sheet shows them, e.g. 'Median CD4', '% parent', 'Count'.",
      {"gates": {"type": "array", "items": {"type": "string"}}, "statistics": {"type": "array", "items": _STATISTIC},
       "parameters": {"type": "array", "items": {"type": "string"}, "description": "Shorthand: median and CV of each."},
       "filter": {"type": "string"}, "sort": {"type": "string"}, "descending": {"type": "boolean"},
       "colour": {"type": "string", "description": "A column to colour by value"}, "chart": {"type": "string", "description": "A column to chart"}})
def _show_sheet(ctx: Ctx, gates: list[str] | None = None, statistics: list[dict] | None = None, parameters: list[str] | None = None,
                filter: str | None = None, sort: str | None = None, descending: bool = False,
                colour: str | None = None, chart: str | None = None) -> dict:
    sid = lab.dataset(ctx.exp_id).sample_ids[0]
    keys = [k for k in (_gate(ctx, sid, g) for g in gates or []) if k]
    cols = _columns(ctx, sid, statistics) + [{"stat": s, "parameter": c, "arg": None} for c in (_channel(ctx, p) for p in parameters or []) if c for s in ("median", "cv")]
    spec = [f"{c['stat']}:{c['parameter'] or ''}" + (f":{c['arg']}" if c["arg"] else "") for c in cols]
    ctx.emit(type="lab", view="table", sheetGates=keys, sheetSpec=spec, sheetFilter=filter or "",
             sheetSort={"key": sort, "desc": bool(descending)} if sort else None, sheetColour=colour, sheetChart=chart)
    return {"summary": f"Opened the sheet: {len(keys) or 'every'} gate(s), {len(cols)} statistic column(s)"
                       + (f", coloured by {colour}" if colour else "") + (f", charting {chart}" if chart else "")}


# What each tool is doing, in the words shown live in the chat while a turn runs.
LABELS = {
    "status": "Reading the experiment", "describe_gates": "Reading the gate tree", "read_figure": "Reading the figure", "show_figure": "Opening the figure", "population_stats": "Reading the counts",
    "read_attachment": "Reading {name}", "set_question": "Noting what the experiment is for",
    "plan_pipeline": "Drafting the plan", "edit_pipeline": "Changing the plan", "run_step": "Running {step_id}",
    "write_view": "Drawing a figure", "write_macro": "Writing a script", "train_model": "Checking the gating models",
    "add_image_steps": "Setting up image steps", "add_autogate_step": "Setting up the gate models", "read_model": "Reading the gate models",
    "keep": "Keeping the draft", "undo": "Undoing the draft", "run_workflow": "Running the workflow", "read_execution": "Reading the run", "open_page": "Opening {page}", "focus_lab": "Pointing the data lab",
    "show_plot": "Laying out the plot", "compare_files": "Lining up the files", "select_gate": "Picking out {gate}",
    "gate_stats": "Reading the numbers", "show_sheet": "Filling the sheet",
}


def label(name: str, args: dict) -> str:
    text = LABELS.get(name, name.replace("_", " ").capitalize())
    try:
        return text.format(**{k: v for k, v in args.items() if isinstance(v, (str, int, float))})
    except (KeyError, IndexError):
        return text.split(" {")[0]


def schemas(names: list[str] | None = None) -> list[dict]:
    return [t.schema() for n, t in REGISTRY.items() if names is None or n in names]


# the report writer's tool lives in its own module and registers itself here
from flow_er.llm import report_tools  # noqa: F401
