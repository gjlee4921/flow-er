"""The kinds of step a workflow is built from, defined once: what each takes in and makes, the
version of its code, what it is for, and how it runs. The planner's list of ops, the canvas's node
cards and the executor all read this, so a new kind of step is one entry here.

Ports are typed: the files (samples), the FlowJo workspace (gates), files added in the chat (file),
events, labels, a table, a figure, a model. A step may only read what an earlier step or a source
makes, and only of a type it takes.
"""
from __future__ import annotations

import csv

from flow_er.workspace import lab, models, render, scripts, store

SOURCES = {"fcs": "samples", "wsp": "gates", "uploads": "file"}  # inputs that exist before any step runs

NODE_TYPES: dict[str, dict] = {
    # the FlowJo workspace (a "gates" source) also carries the spillover matrix and the transforms
    "compensate": {"name": "Compensate and transform", "kind": "macro", "takes": ["samples", "gates"], "makes": "events", "version": "1",
                   "about": "Apply the spillover matrix and the display transforms (FlowKit), from the FlowJo workspace when there is one."},
    "render_image": {"name": "Render a density image", "kind": "macro", "takes": ["samples", "events", "labels", "gates"], "makes": "image",
                     "version": "1", "reads_gates": True,
                     "about": "Bin one population's events on two channels into a 101 × 101 density image, the way the gate models read them (recipe 1bd7b6e6).",
                     "params": {"x": {"label": "X channel", "choices": "channels"}, "y": {"label": "Y channel", "choices": "channels"},
                                "parent": {"label": "Population", "choices": "gates", "default": ""}}},
    "custom_macro": {"name": "Script", "kind": "macro", "takes": ["samples", "events", "table", "derived events", "image"], "makes": "*", "version": "3",
                     "about": "A script for a step no built-in op covers, written or adapted by the agent, tested and run in the sandbox."},
    "gate_from_workspace": {"name": "Apply the FlowJo gates", "kind": "model", "takes": ["events", "samples", "gates"], "makes": "labels", "version": "1",
                            "about": "Use the gates in the lab's FlowJo workspace as the labels."},
    "train_gate_model": {"name": "Train a gate model", "kind": "model", "takes": ["labels", "events", "samples", "gates"], "makes": "model", "version": "1",
                         "about": "Retrain the lab's gate models on the gates the scientist kept (flow_er.ml); a new version is used only if it is better."},
    "autogate": {"name": "Gate with the lab's models", "kind": "model", "takes": ["samples", "events", "labels", "gates"], "makes": "table",
                 "version": "1", "reads_models": True,
                 "about": "Gate every file with the lab's trained models (a per-gate U-Net, else the lab's template): per gate the count, "
                          "% of parent, which model answered and how sure it was, and FlowJo's % and an F1 where FlowJo gates exist. "
                          "The model gates are drafts the scientist reviews and keeps in the data lab.",
                 "columns": {"sample": "the FCS file name", "gate": "the gate's full path", "model": "which model answered (or template_copy)",
                             "confidence": "how sure the model was near the gate's edge, 0-1 (blank for template copies)",
                             "low_confidence": "true when the gate should be reviewed", "count": "events in the gate",
                             "pct_parent": "% of its parent", "flowjo_pct": "FlowJo's % of parent, where known", "f1": "agreement with FlowJo's gate, 0-1"}},
    "population_stats": {"name": "Count every gate", "kind": "measure", "takes": ["labels", "events", "gates", "samples", "model"], "makes": "table", "version": "2",
                         "about": "Count every gate in every sample: count, % of parent, % of total.", "reads_gates": True,
                         # what a script reading this table is told, so its fixture looks like the real thing
                         "columns": {"sample": "the FCS file name", "gate": "the gate's own name, e.g. CD3+",
                                     "gate_path": "its full path from the top, e.g. Time/Singlets/aAmine-/CD3+",
                                     "parent": "the parent's gate_path ('' for a top gate)", "source": "flowjo or user",
                                     "count": "events in the gate", "pct_parent": "% of its parent", "pct_total": "% of all events"}},
    "export_prism": {"name": "Export for Prism", "kind": "view", "takes": ["table"], "makes": "file", "version": "2",
                     "about": "A Prism-ready CSV: one row per file, one column per gate.",
                     "params": {"value": {"label": "Value in each cell", "enum": ["pct_parent", "count", "pct_total"], "default": "pct_parent",
                                          "names": {"pct_parent": "% of parent", "count": "Count", "pct_total": "% of total"}}}},
    "view": {"name": "Figure", "kind": "view", "takes": ["table"], "makes": "figure", "version": "1",
             "about": "A figure drawn from a saved table."},
}


def params(op: str, given: dict | None) -> dict:
    """A step's settings, checked against its kind's and filled with defaults."""
    spec = NODE_TYPES.get(op, {}).get("params", {})
    out = {}
    for k, p in spec.items():
        v = (given or {}).get(k, p.get("default"))
        if "enum" in p and v not in p["enum"]:
            raise ValueError(f"{k} must be one of {p['enum']}")
        out[k] = v
    unknown = set(given or {}) - set(spec)
    if unknown:
        raise ValueError(f"{NODE_TYPES.get(op, {}).get('name', op)} has no setting {sorted(unknown)}")
    return out


def makes(op: str) -> str:
    return NODE_TYPES.get(op, {}).get("makes", "*")


def port_type(name: str, steps: list[dict]) -> str:
    """What a named input is: a source's type, or the type the step that makes it produces. A script's
    outputs are typed by its OUTPUTS; events a script returns are "derived events", which only other
    scripts read (the built-in steps read the files themselves, so they would not see the change)."""
    if name in SOURCES:
        return SOURCES[name]
    for s in steps:
        if name in s.get("outputs", []):
            if s.get("op") == "custom_macro" and s.get("script"):
                outs = scripts.check(s["script"].get("code", "")).get("outputs") or {}
                kind = dict(zip(s["outputs"], outs.values())).get(name)
                return "derived events" if kind == "events" else kind or "*"
            return makes(s.get("op", ""))
    return "*"


def run(exp_id: str, step: dict, samples: list[str] | None = None) -> dict:
    """Do one step's work. Returns its status (done, failed, blocked or queued), a line in plain words,
    and any table it wrote. `samples` limits the work to some files (a new file arriving)."""
    op = step["op"]
    ds = lab.dataset(exp_id)
    which = samples or ds.sample_ids
    if op == "compensate":
        for sid in which:
            ds.events(sid, "processed")
        return {"status": "done", "summary": f"Compensated and transformed {len(which)} file{'s' if len(which) != 1 else ''}."}
    if op == "gate_from_workspace":
        if not ds.ws:
            return {"status": "failed", "summary": "There is no FlowJo workspace in this folder to take gates from."}
        n = len(ds.flowjo_gates(which[0]))
        return {"status": "done", "summary": f"Took {n} gates from {ds.wsp_path.name} as labels."}
    if op == "train_gate_model":
        r = models.train(exp_id, step.get("gate"))
        return {"status": "queued" if r["queued"] else "blocked", "summary": r["message"]}
    if op == "autogate":
        rows = models.table(exp_id, which)
        if samples and store.path(exp_id, "tables", "autogate_counts.csv").exists():
            rows = [r for r in lab.read_table(exp_id, "autogate_counts") if r["sample"] not in set(samples)] + rows
        lab.save_table(exp_id, "autogate_counts", rows)
        gates = [r for r in rows if r["sample"] in set(which)]
        by_model = sum(1 for r in gates if r["model"] != "template_copy")
        unsure = sum(1 for r in gates if r["low_confidence"])
        f1 = [float(r["f1"]) for r in gates if r["f1"] != ""]
        score = f"; median F1 against FlowJo {sorted(f1)[len(f1) // 2]:.3f}" if f1 else ""
        return {"status": "done", "table": "autogate_counts",
                "summary": f"Gated {len(which)} file{'s' if len(which) != 1 else ''} with {models.lab_for(exp_id) or 'the template'} "
                           f"{models.version(exp_id)}: {by_model} gates by models, {len(gates) - by_model} template copies, {unsure} to review{score}."}
    if op == "population_stats":
        rows = lab.population_table(exp_id, samples)
        added = f" ({len(samples)} new file{'s' if len(samples) != 1 else ''})" if samples else ""
        return {"status": "done", "summary": f"Wrote population_stats: {len(rows)} rows{added}.", "table": "population_stats"}
    if op == "export_prism":
        return _prism(exp_id, (step.get("params") or {}).get("value", "pct_parent"))
    if op == "render_image":
        p = step.get("params") or {}
        if not p.get("x") or not p.get("y"):
            return {"status": "failed", "summary": "Choose the two channels to draw."}
        folder = store.path(exp_id, "derived", step["id"], "image")
        for sid in which:
            render.save(folder, sid, render.render(exp_id, sid, p["x"], p["y"], p.get("parent") or None))
        pop = (p.get("parent") or "all events").split("/")[-1]
        return {"status": "done", "summary": f"Drew {len(which)} density image{'s' if len(which) != 1 else ''} of {pop} (recipe {render.RECIPE})."}
    if op == "custom_macro":
        if not step.get("script"):
            return {"status": "blocked", "summary": "This script step has no script yet."}
        # each input that is a table: the table its step saved when it last ran
        plan_steps = store.state(exp_id, "plan")
        made_by = {o: s["id"] for s in plan_steps for o in s.get("outputs", [])}
        last = {r["step"]: r for r in store.state(exp_id, "runs")}
        table_for = {i: last.get(made_by.get(i, ""), {}).get("table") for i in step.get("inputs", [])}
        by_id = {s["id"]: s for s in plan_steps}
        events_from = {}  # inputs that are another script's events: the folder it left them in
        for i in step.get("inputs", []):
            src = by_id.get(made_by.get(i, ""))
            if src and src.get("op") == "render_image":
                events_from[i] = output_folder(exp_id, src, i)
            elif src and src.get("op") == "custom_macro" and src.get("script"):
                outs = list(scripts.check(src["script"]["code"]).get("outputs") or {})
                port = dict(zip(src["outputs"], outs)).get(i)
                if port:
                    events_from[i] = store.path(exp_id, "derived", src["id"], port)
        return scripts.run_step(exp_id, step, step["script"], table_for, samples, events_from)
    if op == "view":
        return {"status": "done", "summary": "Figures draw themselves from their table."}
    return {"status": "failed", "summary": f"No step of kind {op}."}


def output_folder(exp_id: str, step: dict, output: str):
    """Where a step leaves the files behind one of its outputs: a render step's images, or the events
    or images a script returned."""
    if step.get("op") == "render_image":
        return store.path(exp_id, "derived", step["id"], "image")
    outs = list(scripts.check(step.get("script", {}).get("code", "")).get("outputs") or {})
    port = dict(zip(step.get("outputs", []), outs)).get(output, output)
    return store.path(exp_id, "derived", step["id"], port)


def _prism(exp_id: str, value: str = "pct_parent") -> dict:
    """The population table reshaped the way Prism wants it: files down, gates across, one value each."""
    rows = lab.read_table(exp_id, "population_stats") if store.path(exp_id, "tables", "population_stats.csv").exists() else []
    if not rows:
        return {"status": "failed", "summary": "There is no population table yet: run the counting step first."}
    gates = list(dict.fromkeys(r["gate"] for r in rows))
    files = list(dict.fromkeys(r["sample"] for r in rows))
    by = {(r["sample"], r["gate"]): r[value] for r in rows}
    label = NODE_TYPES["export_prism"]["params"]["value"]["names"][value]
    name = f"prism_{value}"
    with open(store.path(exp_id, "tables", f"{name}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["File", *[f"{g} ({label})" for g in gates]])
        for sid in files:
            w.writerow([sid, *[by.get((sid, g), "") for g in gates]])
    return {"status": "done", "summary": f"Wrote {name}.csv: {len(files)} files × {len(gates)} gates, {label}.", "table": name}
