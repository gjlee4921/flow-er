"""The results page: a short paper about the experiment, on screen.

    state/report.json    title, summary and key numbers the agent wrote and the scientist kept

The words are the agent's; every number in them is a reference to a table cell, written
`{{population_stats | gate=CD4+ | sample=E01_008 | pct_parent}}`, and the page shows the value with the
table, row and step behind it. A number typed straight into the text is refused (`lint`), which carries
"numbers come from saved tables" into the prose. Methods and the data list are written by code from the
files, the plan and the gates, so they cannot drift from what was done.
"""
from __future__ import annotations

import hashlib
import json
import re
from importlib import metadata
from statistics import mean, median, stdev

from flow_er.workspace import data, lab, store, views

TOKEN = re.compile(r"\{\{([^{}]+)\}\}")
AGG = {"mean": mean, "median": median, "min": min, "max": max}
# a percentage, a decimal or a grouped count typed as text; marker names (CD4, CD127) and years pass
UNBOUND = re.compile(r"(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?\s?%|\d+\.\d+|\d{1,3}(?:,\d{3})+)(?![A-Za-z0-9])")
SOFTWARE = [("flow-er", "MIT"), ("flowkit", "BSD-3"), ("flowio", "BSD-3"), ("numpy", "BSD-3"), ("pandas", "BSD-3")]
REFERENCES = [
    {"text": "FlowKit: a Python toolkit for flow cytometry analysis supporting Gating-ML and FlowJo workspaces",
     "url": "https://github.com/whitews/FlowKit"},
    {"text": "Lee JA, Spidlen J, et al. MIFlowCyt: the minimum information about a flow cytometry experiment. "
             "Cytometry A, 2008", "url": "https://pubmed.ncbi.nlm.nih.gov/18752282/"},
]


# ---- bound numbers -------------------------------------------------------------------------

def _matches(cell, want: str) -> int:
    """How well a table cell matches a value in a reference: 2 exactly, 1 as the end of a gate path or
    a piece of a file name (E01_008 in 101_DEN084Y5_15_E01_008_clean.fcs), 0 not at all."""
    c = str(cell)
    if c == want:
        return 2
    if c.endswith("/" + want) or re.search(rf"(^|[^A-Za-z0-9]){re.escape(want)}([^A-Za-z0-9]|$)", c):
        return 1
    return 0


def _fmt(column: str, v: float) -> str:
    if column.startswith("pct"):
        return f"{v:.2f}%" if abs(v) < 1 else f"{v:.1f}%"
    if column in ("count", "events", "n") or float(v).is_integer():
        return f"{round(v):,}"
    return f"{v:.3g}"


def resolve(exp_id: str, ref: str) -> dict:
    """One reference to its value: {display, value, table, column, rows, agg}. ValueError says what is wrong."""
    parts = [p.strip() for p in ref.split("|") if p.strip()]
    if len(parts) < 2:
        raise ValueError(f"'{ref}' needs a table and a column, e.g. population_stats | gate=CD4+ | pct_parent")
    table, *rest = parts
    agg = rest.pop() if rest and rest[-1] in (*AGG, "range") else None
    if not rest or "=" in rest[-1]:
        raise ValueError(f"'{ref}' names no column")
    column, filters = rest[-1], [p.split("=", 1) for p in rest[:-1]]
    rows = lab.read_table(exp_id, table)
    if not rows:
        raise ValueError(f"there is no table '{table}' yet (tables: {', '.join(t['name'] for t in lab.tables(exp_id)) or 'none'})")
    if column not in rows[0]:
        raise ValueError(f"'{table}' has no column '{column}' (columns: {', '.join(rows[0])})")
    for k, v in filters:
        k, v = k.strip(), v.strip()
        cols = [c for c in (k, k + "_path") if c in rows[0]]
        if not cols:
            raise ValueError(f"'{table}' has no column '{k}'")
        scored = [(max(_matches(r[c], v) for c in cols), r) for r in rows]
        best = max((s for s, _ in scored), default=0)
        if best == 0:
            raise ValueError(f"no row of '{table}' has {k}={v}")
        rows = [r for s, r in scored if s == best]
    values = [r[column] for r in rows if isinstance(r[column], (int, float))]
    if not values:
        raise ValueError(f"'{ref}' points at no number")
    if agg is None and len(values) > 1:
        which = "; ".join(" ".join(str(r[c]) for c in ("gate_path", "sample") if c in r) or str(r) for r in rows[:4])
        raise ValueError(f"'{ref}' matches {len(values)} rows ({which}); narrow it, e.g. with the end of the gate "
                         f"path or a sample, or end it with mean, median, min, max or range")
    if agg == "range":
        lo, hi = min(values), max(values)
        display, value = f"{_fmt(column, lo)}–{_fmt(column, hi)}", [lo, hi]
    else:
        value = AGG[agg](values) if agg else values[0]
        display = _fmt(column, value)
    keys = [c for c in rows[0] if c not in (column,) and isinstance(rows[0][c], str)][:4]
    return {"display": display, "value": value, "table": table, "column": column, "agg": agg,
            "rows": [{k: r[k] for k in keys} for r in rows[:12]], "n": len(values)}


def tidy(text: str) -> str:
    """A '%' typed after a reference is dropped: the value carries its own unit."""
    return re.sub(r"(\{\{[^{}]+\}\})\s?%", r"\1", text or "")


def lint(text: str) -> list[str]:
    """Numbers typed into the text instead of referenced from a table."""
    return [m.group(1) for m in UNBOUND.finditer(TOKEN.sub("", text or ""))]


def segments(exp_id: str, text: str) -> list[dict]:
    """Text as the page draws it: plain runs, and bound numbers with what is behind them."""
    text = tidy(text)
    out, at = [], 0
    for m in TOKEN.finditer(text or ""):
        if m.start() > at:
            out.append({"text": text[at:m.start()]})
        try:
            out.append({"number": resolve(exp_id, m.group(1)), "ref": m.group(1).strip()})
        except ValueError as e:
            out.append({"broken": str(e), "ref": m.group(1).strip()})
        at = m.end()
    if at < len(text or ""):
        out.append({"text": text[at:]})
    return out


def validate(exp_id: str, title: str, summary: str, key_numbers: list[dict] | None) -> dict:
    """What the agent wrote, checked: every reference resolves and no number is typed in."""
    title, summary = tidy(title), tidy(summary)
    problems = []
    for where, text in (("title", title), ("summary", summary), *((f"key number '{k.get('label', '')}'", k.get("label", ""))
                                                                  for k in key_numbers or [])):
        for n in lint(text):
            problems.append(f"{where} types the number {n}; reference its table cell instead, as {{{{table | column=value | column}}}}")
        for m in TOKEN.finditer(text or ""):
            try:
                resolve(exp_id, m.group(1))
            except ValueError as e:
                problems.append(f"{where}: {e}")
    keys = []
    for k in key_numbers or []:
        ref = TOKEN.sub(lambda m: m.group(1), str(k.get("value", ""))).strip()
        try:
            resolve(exp_id, ref)
        except ValueError as e:
            problems.append(f"key number '{k.get('label', '')}': {e}")
        keys.append({"label": str(k.get("label", "")).strip()[:60], "value": ref})
    if not (title or "").strip():
        problems.append("the title is empty; say what was found, in one line")
    if problems:
        raise ValueError("; ".join(problems))
    return {"title": title.strip()[:160], "summary": summary.strip()[:2000], "key_numbers": keys[:5]}


# ---- what code writes: methods and the data list ----------------------------------------------

def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return ""


def _checksum(exp_id: str, path) -> str:
    """sha256 of a file, cached by size and time so a large file is read once."""
    cache_p = store.path(exp_id, "cache", "checksums.json")
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    st = path.stat()
    key = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    if key not in cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        cache[key] = h.hexdigest()
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        cache_p.write_text(json.dumps(cache, indent=1))
    return cache[key]


def methods(exp_id: str) -> list[dict]:
    """The MIFlowCyt items, from the files, the gates and the kept plan."""
    ds = lab.dataset(exp_id)
    samples = ds.samples()
    head = data.fcs_header(ds.sample_path(samples[0]["id"])) if samples else None
    k = (head or {}).get("keywords", {})
    out = []
    events = ", ".join(f"{s['id']} ({s['events']:,} events)" for s in samples)
    out.append({"heading": "Samples", "text": f"{len(samples)} FCS files were analysed: {events}."})
    markers = [c["name"] for c in ds.channels() if c["group"] == "marker"]
    scatter = [c["short"] for c in ds.channels() if c["group"] == "scatter"]
    out.append({"heading": "Panel", "text": f"Markers: {', '.join(markers) or 'none named'}. Scatter: {', '.join(scatter)}."})
    inst = " ".join(x for x in (k.get("$CYT"), f"(serial {k['$CYTSN']})" if k.get("$CYTSN") else None) if x)
    when = k.get("$DATE")
    if inst or when:
        out.append({"heading": "Instrument", "text": f"Acquired on {inst or 'an instrument the files do not name'}"
                                                     + (f" on {when}" if when else "") + f", saved as {head['version']}."})
    spill = any(key.upper() in ("$SPILLOVER", "SPILL", "$SPILL") for key in k)
    out.append({"heading": "Compensation", "text": "The files carry a spillover matrix ($SPILLOVER)." if spill
                else "The files carry no spillover matrix."})
    user = store.state(exp_id, "gates")
    gate_text = (f"Gates came from the FlowJo workspace {ds.wsp_path.name}" if ds.wsp_path else "There was no FlowJo workspace")
    if user:
        gate_text += f", with {len(user)} gate{'s' if len(user) != 1 else ''} drawn by the scientist ({', '.join(g['name'] for g in user[:6])})"
    tree = lab.tree(exp_id, samples[0]["id"]) if samples else []
    if tree:
        gate_text += ". Hierarchy: " + "; ".join(g["key"] for g in tree[:14]) + ("; …" if len(tree) > 14 else "")
    out.append({"heading": "Gating strategy", "text": gate_text + "."})
    plan = store.state(exp_id, "plan")
    runs = {r["step"]: r for r in store.state(exp_id, "runs")}
    if plan:
        steps = "; ".join(f"{s['title']} ({'ran' if runs.get(s['id'], {}).get('status') == 'done' else 'not run'})" for s in plan)
        out.append({"heading": "Analysis", "text": f"The kept workflow has {len(plan)} step{'s' if len(plan) != 1 else ''}: {steps}. "
                                                   "Counts and frequencies are computed from every event in each gate; "
                                                   "a frequency is the % of its parent gate."})
    soft = ", ".join(f"{name} {_version(name)}".strip() for name, _ in SOFTWARE if _version(name))
    out.append({"heading": "Software", "text": f"Analysed with {soft}."})
    return out


def data_and_code(exp_id: str) -> dict:
    ds = lab.dataset(exp_id)
    files = [{"name": p.name, "bytes": p.stat().st_size, "sha256": _checksum(exp_id, p)} for p in ds.fcs]
    if ds.wsp_path:
        files.append({"name": ds.wsp_path.name, "bytes": ds.wsp_path.stat().st_size, "sha256": _checksum(exp_id, ds.wsp_path)})
    plan = store.state(exp_id, "plan")
    version = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12] if plan else None
    runs = {r.get("table"): r for r in store.state(exp_id, "runs") if r.get("table")}
    tables = [{**t, "step": runs.get(t["name"], {}).get("step")} for t in lab.tables(exp_id)]
    software = [{"name": n, "version": _version(n), "licence": lic} for n, lic in SOFTWARE if _version(n)]
    return {"files": files, "tables": tables, "workflow_version": version, "software": software}


# ---- figures, read back as numbers --------------------------------------------------------------

BASELINE = re.compile(r"naive|control|ctrl|baseline|untreated|healthy|young|wild|^wt|pre", re.IGNORECASE)


def figures(exp_id: str) -> list[dict]:
    """The figures in report order (kept ones, then the draft), numbered as the page numbers them."""
    out = [{"spec": v, "draft": False} for v in store.state(exp_id, "views")]
    d = store.draft(exp_id)
    if d and d["kind"] == "view":
        out.append({"spec": d["payload"], "draft": True})
    return [{**f, "n": i + 1} for i, f in enumerate(out)]


def read_figure(exp_id: str, f: dict) -> dict:
    """One figure as an assistant can explain it: what it draws, and the numbers behind each mark (mean, SD,
    n and range per x and group), with the fold change from the baseline group when it compares groups."""
    spec, rows = f["spec"], views.data(exp_id, f["spec"])
    x, y, color = spec.get("x"), spec.get("y"), spec.get("color")
    head = {"figure": f"Figure {f['n']}", "title": spec["title"], "caption": spec.get("caption", ""), "type": spec["type"],
            "draft": f["draft"], "table": spec["table"], "x": x, "y": y, "color": color, "agg": spec.get("agg"),
            "filter": spec.get("filter") or {}, "rows_drawn": len(rows)}
    if spec["type"] == "table" or not y:
        return {**head, "rows": rows[:40]}
    cells: dict[tuple, list[float]] = {}
    for r in rows:
        if isinstance(r.get(y), (int, float)):
            cells.setdefault((r.get(x), r.get(color) if color else None), []).append(r[y])
    r4 = lambda v: float(f"{v:.4g}")
    marks = [{"x": k[0], **({"group": k[1]} if color else {}), "mean": r4(mean(v)), "n": len(v),
              **({"sd": r4(stdev(v)), "min": r4(min(v)), "max": r4(max(v))} if len(v) > 1 else {})}
             for k, v in cells.items()]
    groups = list(dict.fromkeys(k[1] for k in cells)) if color else []
    if len(groups) > 1:
        base = next((g for g in groups if BASELINE.search(str(g))), groups[0])
        head["baseline_group"] = base
        means = {k: mean(v) for k, v in cells.items()}
        for m in marks:
            b = means.get((m["x"], base))
            if m.get("group") != base:
                m["fold_vs_baseline"] = round(m["mean"] / b, 2) if b else None
    return {**head, "marks": marks[:60]}


# ---- the whole page ---------------------------------------------------------------------------

def build(exp_id: str) -> dict:
    """Everything the results page draws, numbers resolved. The waiting draft shows in place of the kept text."""
    d = store.draft(exp_id)
    kept = store.state(exp_id, "report") or {}
    text, is_draft = (d["payload"], True) if d and d["kind"] == "report" else (kept, False)
    steps = {r.get("table"): r.get("step") for r in store.state(exp_id, "runs") if r.get("table")}

    def bound(seg: list[dict]) -> list[dict]:
        for s in seg:
            if "number" in s:
                s["number"]["step"] = steps.get(s["number"]["table"])
        return seg
    return {
        "title": text.get("title") or "",
        "summary": bound(segments(exp_id, text.get("summary", ""))),
        "key_numbers": [{"label": k["label"], **bound(segments(exp_id, "{{" + k["value"] + "}}"))[0]}
                        for k in text.get("key_numbers", [])],
        "draft": is_draft,
        "written": bool(text),
        "methods": methods(exp_id),
        "data": data_and_code(exp_id),
        "references": REFERENCES,
    }
