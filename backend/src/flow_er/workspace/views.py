"""Figures the agent writes on request, as declarative specs over a saved table.

A view never carries numbers of its own. It names a table and the columns to draw; the
backend reads the rows, the browser draws them and makes them interactive. That is how a new
way of showing a result can be asked for in a sentence and still be traceable to a table.
"""
from __future__ import annotations

import re
import uuid
from collections import defaultdict
from statistics import mean

from flow_er.workspace import lab

TYPES = {
    "bar": "one bar per x, height y; colour splits bars side by side",
    "dot": "one dot per row, x categories against y, coloured; good for samples or mice",
    "dumbbell": "two values per x joined by a line (needs color with exactly two groups)",
    "line": "y against x, one line per colour",
    "scatter": "y against x, both numeric",
    "heatmap": "x by color as a grid, y as the shade",
    "table": "the rows themselves, with the chosen columns",
    "anatomy": "organs placed where they sit on a 2-D mouse: x is the organ column (spleen, thymus, lung, liver, "
               "blood, draining node, tumour, bone marrow...), y the value, drawn as circle size; color a condition "
               "column with two groups (naive or control first) colours each organ by its fold change; keep agg none "
               "so each animal shows on hover",
}
AGG = {"none", "mean", "sum", "max", "min"}


def validate(exp_id: str, spec: dict) -> dict:
    problems = []
    table = spec.get("table")
    rows = lab.read_table(exp_id, table) if table else []
    if not rows:
        problems.append(f"table '{table}' does not exist or is empty; run a measure step first")
    cols = set(rows[0].keys()) if rows else set()
    vtype = spec.get("type")
    if vtype not in TYPES:
        problems.append(f"type must be one of {sorted(TYPES)}")
    for k in ("x", "y", "color"):
        c = spec.get(k)
        if c and cols and c not in cols:
            problems.append(f"{k} column '{c}' is not in {table} (columns: {sorted(cols)})")
    if vtype not in ("table",) and not spec.get("y"):
        problems.append("y is required")
    if spec.get("agg", "none") not in AGG:
        problems.append(f"agg must be one of {sorted(AGG)}")
    flt = spec.get("filter") or {}
    for k in flt:
        if cols and k not in cols:
            problems.append(f"filter column '{k}' is not in {table}")
    if not problems and flt and not any(_match(r, flt) for r in rows):
        seen = {k: sorted({str(r.get(k)) for r in rows})[:8] for k in flt}
        problems.append(f"the filter matches no rows of {table}; values there: {seen}")
    if problems:
        raise ValueError("; ".join(problems))
    unquote = lambda t: str(t or "").strip().strip('"').strip("'").strip()
    title = unquote(spec.get("title")) or "Untitled view"
    title = title[:120]
    vid = spec.get("id") or re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or uuid.uuid4().hex[:6]
    return {"id": vid, "title": title, "caption": unquote(spec.get("caption"))[:600], "type": vtype,
            "table": table, "x": spec.get("x"), "y": spec.get("y"), "color": spec.get("color"),
            "filter": spec.get("filter") or {}, "agg": spec.get("agg", "none"),
            "sort": spec.get("sort", "none"), "columns": spec.get("columns") or [],
            "x_label": spec.get("x_label"), "y_label": spec.get("y_label"), "log": bool(spec.get("log"))}


def _same(cell, want) -> bool:
    """A filter value against a table cell. Saved tables come back as text, so true and "True",
    1 and "1.0" are the same value; a gate's path ("…/CD3/CD4 Single Positive") is that gate where
    the table names only the gate ("CD4 Single Positive")."""
    if cell == want:
        return True
    a, b = str(cell).strip().lower(), str(want).strip().lower()
    if a == b or (a and "/" not in a and "/" in b and b.rsplit("/", 1)[1] == a):
        return True
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def _match(row: dict, flt: dict) -> bool:
    for k, v in flt.items():
        if not any(_same(row.get(k), x) for x in (v if isinstance(v, list) else [v])):
            return False
    return True


def data(exp_id: str, spec: dict) -> list[dict]:
    """The rows a view draws: filtered, aggregated if asked, sorted."""
    rows = [r for r in lab.read_table(exp_id, spec["table"]) if _match(r, spec.get("filter") or {})]
    x, y, color, agg = spec.get("x"), spec.get("y"), spec.get("color"), spec.get("agg", "none")
    if agg != "none" and x and y:
        groups: dict[tuple, list[float]] = defaultdict(list)
        for r in rows:
            if isinstance(r.get(y), (int, float)):
                groups[(r.get(x), r.get(color) if color else None)].append(r[y])
        fn = {"mean": mean, "sum": sum, "max": max, "min": min}[agg]
        rows = [{x: k[0], **({color: k[1]} if color else {}), y: round(fn(v), 4), "n": len(v)} for k, v in groups.items()]
    if spec.get("sort") in ("asc", "desc") and y:
        rows.sort(key=lambda r: (r.get(y) is None, r.get(y)), reverse=spec["sort"] == "desc")
    return rows[:2000]
