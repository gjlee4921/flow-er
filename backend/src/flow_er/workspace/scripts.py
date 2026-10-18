"""Script steps (spec §3): one Python file per step, written or adapted by the agent (or the scientist),
checked by code and tested in the sandbox before it can be kept, then run there by the workflow.

The contract, at the top of the file, as plain literals:
    PROVENANCE = {"what": ..., "source": url or None, "version": ..., "licence": SPDX id or None, "adapted": ...}
    INPUTS = {"events": "events"}             # port name -> events (one table per file) or table
    OUTPUTS = {"qc": "table"}
    PARAMS = {"window_s": 1.0}                # optional, with defaults
    def run(inputs, params) -> dict           # returns one value per output
Before it can be kept it must parse, import only the pinned environment, carry an allowed licence, give
the expected answer on its fixture and give the same output twice.
"""
from __future__ import annotations

import ast
import csv
import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

from flow_er.workspace import lab, sandbox, store

IMPORTS = {"numpy", "pandas", "scipy", "skimage", "flowkit", "flowio", "flowutils",  # the pinned environment
           "math", "statistics", "collections", "itertools", "functools", "operator", "re", "json", "typing",
           "dataclasses", "enum", "bisect", "heapq", "fractions", "decimal", "warnings", "__future__"}
LICENCES = {"MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC", "PSF-2.0", "0BSD", "Unlicense", "CC0-1.0"}
# modules that reach outside the script; the sandbox stops them anyway, but a check says so plainly
REACH = {"os", "sys", "subprocess", "socket", "shutil", "pathlib", "glob", "io", "tempfile", "importlib", "ctypes", "urllib",
         "http", "ftplib", "smtplib", "requests", "pickle", "marshal", "builtins", "multiprocessing", "threading", "signal", "inspect"}
BANNED_CALLS = {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint", "globals", "vars"}
PORTS = {"events", "table"}


def _literal(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    return None


def licence_ok(spdx: str | None) -> str | None:
    """None if the licence may be used, otherwise why not, in plain words."""
    if not spdx:
        return None
    if spdx in LICENCES:
        return None
    if "GPL" in spdx.upper():
        return (f"{spdx} is a copyleft licence: adapting it would put the whole workflow under it. "
                "Use a permissive source (MIT, BSD, Apache-2.0, ISC, PSF) or write the step from scratch.")
    return f"{spdx} is not on the allowed list ({', '.join(sorted(LICENCES))})."


def check(code: str) -> dict:
    """The checks code can make without running it. Returns ok, the problems in plain words, and the
    script's provenance, ports and settings."""
    problems: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"ok": False, "problems": [f"it does not parse: {e.msg} (line {e.lineno})"]}
    try:
        prov, ins, outs, params = (_literal(tree, n) for n in ("PROVENANCE", "INPUTS", "OUTPUTS", "PARAMS"))
    except (ValueError, SyntaxError):
        return {"ok": False, "problems": ["PROVENANCE, INPUTS, OUTPUTS and PARAMS must be plain literals"]}
    if not isinstance(prov, dict) or not prov.get("what"):
        problems.append("PROVENANCE is missing or does not say what the script does")
        prov = prov if isinstance(prov, dict) else {}
    if prov.get("source") and not prov.get("licence"):
        problems.append("PROVENANCE names a source but not its licence")
    if why := licence_ok(prov.get("licence")):
        problems.append(why)
    for name, ports in (("INPUTS", ins), ("OUTPUTS", outs)):
        if not isinstance(ports, dict) or not ports:
            problems.append(f"{name} must name at least one port and its type (events or table)")
        elif bad := [f"{k}: {v}" for k, v in ports.items() if v not in PORTS]:
            problems.append(f"{name} has ports of a type scripts cannot take yet: {', '.join(bad)} (events or table)")
    if params is not None and not isinstance(params, dict):
        problems.append("PARAMS must be a dict of settings and their defaults")
    runs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"]
    if not runs or len(runs[0].args.args) != 2:
        problems.append("it needs a function run(inputs, params) at the top level")
    for node in ast.walk(tree):
        mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
        for m in mods:
            top = m.split(".")[0]
            if top in REACH:
                problems.append(f"it imports {m}, which scripts may not use: they get no files, network or other programs")
            elif top not in IMPORTS:
                problems.append(f"it imports {m}, which is not in the pinned environment (numpy, pandas, scipy, scikit-image, flowkit, flowio, flowutils)")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            problems.append(f"it calls {node.func.id}(), which scripts may not use (line {node.lineno})")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__") and node.attr not in ("__name__", "__doc__"):
            problems.append(f"it reaches into {node.attr} (line {node.lineno})")
    return {"ok": not problems, "problems": list(dict.fromkeys(problems)), "provenance": prov,
            "inputs": ins or {}, "outputs": outs or {}, "params": params or {}}


# ---- inputs on disk ------------------------------------------------------------------------------

def _write_events(folder: Path, files: dict[str, dict]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for sid, t in files.items():
        np.savez(folder / f"{sid}.npz", data=np.asarray(t["rows"], dtype="float64").reshape(-1, len(t["columns"])),
                 columns=np.array(t["columns"], dtype=str))
        if t.get("keywords"):
            (folder / f"{sid}.json").write_text(json.dumps(t["keywords"]))


def _write_table(path: Path, t: dict) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(t["columns"])
        w.writerows(t["rows"])


def _read_table(path: Path) -> dict:
    with open(path) as f:
        rows = list(csv.reader(f))
    return {"columns": rows[0] if rows else [], "rows": rows[1:]}


def _close(got: dict, want: dict, tol: float = 1e-6) -> str | None:
    """None if a table matches the expected one (numbers within a small tolerance), else how it differs."""
    if [str(c) for c in got["columns"]] != [str(c) for c in want["columns"]]:
        return f"columns {got['columns']} instead of {want['columns']}"
    if len(got["rows"]) != len(want["rows"]):
        return f"{len(got['rows'])} rows instead of {len(want['rows'])}"
    for i, (a, b) in enumerate(zip(got["rows"], want["rows"])):
        for c, x, y in zip(got["columns"], a, b):
            try:
                if abs(float(x) - float(y)) > tol * max(1.0, abs(float(y))):
                    return f"row {i + 1}, {c}: {x} instead of {y}"
            except (TypeError, ValueError):
                if str(x) != str(y):
                    return f"row {i + 1}, {c}: {x} instead of {y}"
    return None


def test(code: str, fixture: dict) -> dict:
    """Run the script twice on its fixture in the sandbox: it must give the expected tables, and the
    same output both times. `fixture` is {"inputs": {port: table or {file: table}}, "params": {...},
    "expected": {output: table}}, each table {"columns": [...], "rows": [[...]]}."""
    c = check(code)
    if not c["ok"]:
        return {"passed": False, "problems": c["problems"]}
    tmp = Path(tempfile.mkdtemp(prefix="flower-fixture-"))
    try:
        inputs = {}
        for port, kind in c["inputs"].items():
            given = (fixture.get("inputs") or {}).get(port)
            if given is None:
                return {"passed": False, "problems": [f"the fixture has no input '{port}'"]}
            if kind == "events":
                _write_events(tmp / "in" / port, given)
                inputs[port] = tmp / "in" / port
            else:
                (tmp / "in").mkdir(exist_ok=True)
                _write_table(tmp / "in" / f"{port}.csv", given)
                inputs[port] = tmp / "in" / f"{port}.csv"
        params = {**c["params"], **(fixture.get("params") or {})}
        runs = [sandbox.run(code, inputs, tmp / f"out{i}", params) for i in (1, 2)]
        if runs[0]["status"] != "done":
            return {"passed": False, "problems": [f"it failed in the sandbox: {runs[0]['error']}"], "log": runs[0]["log"][-1500:]}
        problems = []
        for name, want in (fixture.get("expected") or {}).items():
            path = runs[0]["outputs"].get(name)
            if path is None or not path.is_file():
                problems.append(f"it gave no table '{name}'")
            elif diff := _close(_read_table(path), want):
                problems.append(f"'{name}' is not the expected answer: {diff}")
        if not fixture.get("expected"):
            problems.append("the fixture has no expected answer to check against")
        if runs[1].get("hash") != runs[0]["hash"]:
            problems.append("it gave a different output the second time, so it is not deterministic")
        return {"passed": not problems, "problems": problems, "hash": runs[0]["hash"], "seconds": runs[0]["seconds"]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---- running a kept script in the workflow ---------------------------------------------------------

def _clock(t: str | None) -> float | None:
    try:
        h, m, s = (float(x) for x in str(t).split(":")[:3])
        return h * 3600 + m * 60 + s
    except (TypeError, ValueError):
        return None


def keywords(ds, sid: str, data: np.ndarray, cols: list[str]) -> dict:
    """A file's own FCS keywords (not the per-channel ones), and time_unit: "seconds" when the Time
    channel spans what the acquisition clock ($BTIM to $ETIM) says, otherwise "unknown"."""
    meta = ds._sample(sid).metadata
    kw = {f"${k.upper()}": str(v) for k, v in meta.items() if not re.match(r"p\d+[a-z]$|begin|end|byteord|datatype|mode|nextdata", k)}
    kw["time_unit"] = "unknown"
    if "Time" in cols and len(data):
        t = data[:, cols.index("Time")]
        b, e = _clock(meta.get("btim")), _clock(meta.get("etim"))
        if b is not None and e is not None and e > b and abs(float(t.max() - t.min()) - (e - b)) <= max(2.0, 0.1 * (e - b)):
            kw["time_unit"] = "seconds"
    return kw


def _events_input(exp_id: str, folder: Path, samples: list[str] | None) -> Path:
    """Every file's events, compensated but not display-scaled (the values statistics use), one table
    per file with the channel names as columns, and the file's keywords beside it."""
    ds = lab.dataset(exp_id)
    cols = [c["id"] for c in ds.channels()]
    folder.mkdir(parents=True, exist_ok=True)
    for sid in samples or ds.sample_ids:
        data = ds.compensated(sid)
        np.savez(folder / f"{sid}.npz", data=data, columns=np.array(cols, dtype=str))
        (folder / f"{sid}.json").write_text(json.dumps(keywords(ds, sid, data, cols)))
    return folder


def run_step(exp_id: str, step: dict, macro: dict, table_for: dict[str, str], samples: list[str] | None = None,
             events_from: dict[str, Path] | None = None) -> dict:
    """Run a kept script for its step. `table_for` maps each input that is a table to the saved table
    it names; `events_from` maps an input that is another script's events to the folder they are in.
    Tables it returns are saved as <step>_<output>; events it returns are kept in derived/<step>/."""
    c = check(macro["code"])
    if not c["ok"]:
        return {"status": "failed", "summary": "The script no longer passes its checks: " + "; ".join(c["problems"])}
    tmp = Path(tempfile.mkdtemp(prefix="flower-step-"))
    try:
        inputs = {}
        for port, kind in c["inputs"].items():
            src = (step.get("inputs") or [None])[list(c["inputs"]).index(port)] if step.get("inputs") else None
            if kind == "events" and (events_from or {}).get(src):
                folder = events_from[src]
                if not folder.exists():
                    return {"status": "blocked", "summary": f"Waiting on the events for '{port}'; run the script that makes them first."}
                inputs[port] = folder
            elif kind == "events":
                inputs[port] = _events_input(exp_id, tmp / "in" / port, samples)
            else:
                name = table_for.get(src or port)
                path = store.path(exp_id, "tables", f"{name}.csv") if name else None
                if not path or not path.exists():
                    return {"status": "blocked", "summary": f"Waiting on the table for '{port}'; run the step that makes it first."}
                inputs[port] = path
        params = {**c["params"], **(step.get("params") or {})}
        r = sandbox.run(macro["code"], inputs, tmp / "out", params)
        if r["status"] != "done":
            return {"status": "failed", "summary": f"The script stopped: {r['error']}"}
        saved = []
        for name, path in r["outputs"].items():
            if path.is_file():
                dst = store.path(exp_id, "tables", f"{step['id']}_{name}.csv")
                shutil.copy2(path, dst)
                saved.append(dst.stem)
            else:
                dst = store.path(exp_id, "derived", step["id"], name)
                shutil.rmtree(dst, ignore_errors=True)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(path, dst)
        what = f"Wrote {', '.join(saved)}" if saved else "Ran"
        return {"status": "done", "summary": f"{what} in the sandbox ({r['seconds']} s).", "table": saved[0] if saved else None,
                "hash": r["hash"]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def describe(macro: dict) -> dict:
    """What the panel shows for a script: its code, provenance, ports and last test."""
    c = check(macro.get("code", ""))
    return {"code": macro.get("code", ""), "provenance": c.get("provenance", {}), "inputs": c.get("inputs", {}),
            "outputs": c.get("outputs", {}), "params": c.get("params", {}), "checks": c.get("problems", []),
            "test": macro.get("test"), "title": macro.get("title"), "summary": macro.get("summary")}


def dumps(x) -> str:
    return json.dumps(x, default=str)
