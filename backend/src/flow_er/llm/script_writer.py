"""The script writer (spec §3): a subagent for one step no built-in op covers. It looks for reference
code on the allowed sites, fetches a file with its licence, writes the script, and tests it in the
sandbox on a small fixture it works out by hand. It hands back only a script that passed.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable

from flow_er.llm import client, prompts, sources, web
from flow_er.workspace import scripts

log = logging.getLogger(__name__)
SCHEMAS = [
    web.TOOL,
    client.tool_schema("fetch_source", "Fetch one file of reference code with its licence: a GitHub file link, or "
                       "pypi:<package>/<path in its source>. A source whose licence is not allowed comes back refused, without code. "
                       "A large file comes back as an index of its functions; ask again with lines (e.g. '1200-1350') for one part.",
                       {"ref": {"type": "string"}, "lines": {"type": "string"}}, ["ref"]),
    client.tool_schema("write_script", "Save the script (the whole file) and check it: it parses, imports only the pinned "
                       "environment, declares PROVENANCE, INPUTS, OUTPUTS and PARAMS, and has run(inputs, params).",
                       {"code": {"type": "string"}}, ["code"]),
    client.tool_schema("test_script", "Run the saved script twice in the sandbox on a fixture: it must give the expected "
                       "tables and the same output both times. fixture = {inputs: {port: table, or {file: table} for events}, "
                       "params: {...}, expected: {output: table}}; a table is {columns: [...], rows: [[...], ...]}.",
                       {"fixture": {"type": "object"}}, ["fixture"]),
]


def _final(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def write(goal: str, inputs: dict[str, str], channels: list[str], rounds: int = 12,
          on_progress: Callable[[str], None] | None = None) -> dict:
    """A tested script for `goal`, reading `inputs` (port -> events or table). Returns the code, its
    provenance, the fixture and test result, and a title and summary; or {"refused": why}."""
    st: dict = {"code": None, "check": None, "test": None, "fixture": None, "fetched": []}

    def fetch_source(ref: str, lines: str | None = None) -> dict:
        r = sources.fetch(ref, lines)
        st["fetched"].append({k: r.get(k) for k in ("source", "version", "licence", "refused")})
        return r

    def write_script(code: str) -> dict:
        st.update(code=code, check=scripts.check(code), test=None)
        c = st["check"]
        bad = _unfetched(c.get("provenance") or {}, st["fetched"])
        if bad:
            c = {**c, "ok": False, "problems": [*c["problems"], bad]}
            st["check"] = c
        if c["ok"] and c["inputs"] != inputs:
            c = {**c, "ok": False, "problems": [f"INPUTS must be exactly {inputs}"]}
            st["check"] = c
        return {k: c.get(k) for k in ("ok", "problems", "inputs", "outputs", "params")}

    def test_script(fixture: dict) -> dict:
        if not st["code"]:
            return {"passed": False, "problems": ["save the script with write_script first"]}
        if not st["check"]["ok"]:
            return {"passed": False, "problems": st["check"]["problems"]}
        st["fixture"] = fixture
        st["test"] = scripts.test(st["code"], fixture)
        return st["test"]

    handlers = {"fetch_source": fetch_source, "write_script": write_script, "test_script": test_script}
    say = on_progress or (lambda _text: None)
    say("Looking for reference code")
    task = {"goal": goal, "inputs": inputs, "channels": channels}
    items: list[dict] = [{"role": "user", "content": json.dumps(task)}]
    for _ in range(rounds):
        r = client.respond(prompts.SCRIPT_WRITER, items, SCHEMAS)
        items += r.items
        for s in r.searches:
            q = (s.get("action") or {}).get("query")
            if q:
                say(f"Searched for {q}")
        if not r.calls:
            out = _final(r.text)
            if out.get("refused"):
                return {"refused": out["refused"], "fetched": st["fetched"]}
            if st["test"] and st["test"]["passed"]:
                c = st["check"]
                return {"code": st["code"], "provenance": c["provenance"], "inputs": c["inputs"], "outputs": c["outputs"],
                        "params": c["params"], "fixture": st["fixture"], "test": st["test"], "fetched": st["fetched"],
                        "title": out.get("title") or c["provenance"].get("what", "Script")[:60], "summary": out.get("summary", "")}
            items.append({"role": "user", "content": "The script has not passed test_script yet. Fix it and test it again, "
                                                     "or reply with JSON {\"refused\": \"why\"} if it cannot be done."})
            continue
        for c in r.calls:
            fn = handlers.get(c.name)
            try:
                res = fn(**c.args) if fn else {"error": f"no tool {c.name}"}
            except Exception as e:  # a failed fetch or a bad fixture is something the writer can fix
                log.warning("script writer tool %s failed", c.name, exc_info=True)
                res = {"error": f"{e.__class__.__name__}: {e}"}
            say(_progress(c.name, res))
            items.append(client.tool_result(c, res))
    raise ValueError("the script writer ran out of rounds before its script passed its test")


def _progress(tool: str, res: dict) -> str:
    """One line on what the writer just did, for the scientist watching it work."""
    if res.get("error"):
        return f"{tool.replace('_', ' ').capitalize()} hit a problem; trying again"
    if tool == "fetch_source":
        where = re.sub(r"^https?://(www\.)?", "", res.get("source") or "")[:70]
        return f"Refused {where}: {res['licence'] or 'no licence'}" if res.get("refused") else f"Read {where} ({res.get('licence')})"
    if tool == "write_script":
        return "Wrote the script; it passes the checks" if res.get("ok") else f"Wrote the script; fixing {len(res.get('problems') or [])} problem(s)"
    if tool == "test_script":
        return "Tested in the sandbox: right answer, same output twice" if res.get("passed") else "Tested in the sandbox: not right yet, fixing it"
    return tool


def _unfetched(prov: dict, fetched: list[dict]) -> str | None:
    """A source named in PROVENANCE must be one fetch_source read, with the licence it read, so the
    licence on a script is never taken on trust."""
    src = prov.get("source")
    if not src:
        return None
    hit = next((f for f in fetched if f.get("source") and not f.get("refused") and _same(f["source"], src)), None)
    if not hit:
        return "PROVENANCE names a source that was not fetched with fetch_source, so its licence is unchecked"
    if prov.get("licence") != hit["licence"]:
        return f"PROVENANCE says {prov.get('licence')}, but fetch_source read {hit['licence']} for that source"
    return None


def _same(a: str, b: str) -> bool:
    norm = lambda u: re.sub(r"^https?://|/$|\s.*$", "", u.strip()).lower()
    return norm(a) == norm(b) or norm(a).startswith(norm(b)) or norm(b).startswith(norm(a))
