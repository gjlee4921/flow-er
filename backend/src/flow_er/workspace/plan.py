"""The analysis plan: steps the agent proposes and code validates before anyone sees them.

A step names an `op` the backend knows how to run, so a plan can never ask for something the
tool layer cannot do. The agent decides the order and the reasons; code checks the shape.
"""
from __future__ import annotations

from flow_er.workspace import nodes, scripts, store

KINDS = {"macro", "model", "measure", "view"}
# op -> (kind, what it does), from the node types: the agent sees this list in the plan tool's description
OPS = {op: (t["kind"], t["about"]) for op, t in nodes.NODE_TYPES.items()}
SOURCES = set(nodes.SOURCES)  # inputs that exist before any step runs


def validate(steps: list[dict]) -> list[dict]:
    """Return the cleaned steps or raise ValueError with every problem found."""
    problems, seen_ids, produced = [], set(), set(SOURCES)
    out = []
    for i, s in enumerate(steps):
        sid = str(s.get("id", "")).strip()
        op = s.get("op")
        where = f"step {i + 1} ({sid or 'no id'})"
        if not sid:
            problems.append(f"{where}: needs an id")
        elif sid in seen_ids:
            problems.append(f"{where}: id used twice")
        if op not in OPS:
            problems.append(f"{where}: op must be one of {sorted(OPS)}")
            continue
        kind = OPS[op][0]
        takes = nodes.NODE_TYPES[op]["takes"]
        for inp in s.get("inputs", []):
            if inp not in produced:
                problems.append(f"{where}: input '{inp}' is not made by an earlier step")
                continue
            got = nodes.port_type(inp, out)
            if "*" not in takes and got != "*" and got not in takes:
                problems.append(f"{where}: input '{inp}' is {got}, but {nodes.NODE_TYPES[op]['name']} takes {', '.join(takes)}")
        if op == "custom_macro":
            code = (s.get("script") or {}).get("code")
            if not code:
                problems.append(f"{where}: a script step needs its script; ask write_macro to write and test one")
            else:
                c = scripts.check(code)
                problems += [f"{where}: {p}" for p in c["problems"]]
                if c["ok"] and len(c["inputs"]) != len(s.get("inputs", [])):
                    problems.append(f"{where}: the script reads {len(c['inputs'])} inputs but the step gives it {len(s.get('inputs', []))}")
        seen_ids.add(sid)
        produced.update(s.get("outputs", []))
        out.append({"id": sid, "op": op, "kind": kind, "title": s.get("title") or op.replace("_", " "),
                    "inputs": list(s.get("inputs", [])), "outputs": list(s.get("outputs", [])),
                    "why": s.get("why", ""), **({"macro_id": s["macro_id"]} if s.get("macro_id") else {}),
                    **({"params": nodes.params(op, s.get("params"))} if nodes.NODE_TYPES[op].get("params") else {}),
                    **({"gate": s["gate"]} if s.get("gate") else {}),
                    **({"script": s["script"]} if s.get("script") else {})})
    if problems:
        raise ValueError("; ".join(problems))
    return out


def run(exp_id: str, step_id: str) -> dict:
    """Run one kept step, through the executor so it is recorded and reused when nothing changed."""
    from flow_er.workspace import executor
    if not any(s["id"] == step_id for s in store.state(exp_id, "plan")):
        raise ValueError(f"no kept step called '{step_id}'. Keep the plan first, then run its steps.")
    rec = executor.start(exp_id, "only", step_id, trigger="step", wait=True)
    n = rec["nodes"][step_id]
    return {"step": step_id, "op": n["op"], "status": "done" if n["status"] == "cached" else n["status"],
            "summary": n.get("summary", ""), **({"table": n["table"]} if n.get("table") else {}), "execution": rec["id"]}
