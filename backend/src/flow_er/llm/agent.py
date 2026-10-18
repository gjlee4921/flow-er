"""The agent loop: one turn of conversation, and subagents for delegated work.

A turn: the scientist's message, a fresh snapshot of the experiment, and the last few messages
go to the model; it answers or calls tools; tool results go back; repeat until it answers or
runs out of rounds. What the scientist sees is the answer, the tools that ran (the trace), any
UI events (open a page, point the lab somewhere) and the draft waiting for keep or undo.

The data lab has its own assistant on the same loop: its own thread (lab_chat), its own
instructions, and only the tools that move the lab and read numbers, so it can never change the
analysis.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable

from flow_er.config import settings
from flow_er.llm import client, prompts, tools, web
from flow_er.workspace import plan, store

HISTORY = 12  # messages of conversation sent each turn
log = logging.getLogger(__name__)


def _history(exp_id: str, thread: str = "chat") -> list[dict]:
    out = []
    for m in store.messages(exp_id, thread)[-HISTORY:]:
        if m["role"] == "user":
            text = m["text"] + (f"\n[attached: {', '.join(m['attachments'])}]" if m.get("attachments") else "")
            out.append({"role": "user", "content": text})
        elif m["role"] == "assistant":
            used = [t["tool"] for t in m.get("trace", [])]
            note = f"\n[tools used: {', '.join(used)}]" if used else ""
            out.append({"role": "assistant", "content": (m.get("text") or "") + note})
    return out


def _short(args: dict) -> dict:
    """Tool arguments as shown in the trace: long values trimmed."""
    out = {}
    for k, v in args.items():
        s = json.dumps(v, default=str)
        out[k] = v if len(s) <= 160 else s[:157] + "..."
    return out


def _summary(name: str, result: dict) -> str:
    if not result.get("ok"):
        return f"failed: {result.get('error', '')[:200]}"
    for k in ("summary", "message", "kept", "dropped", "opened"):
        if result.get(k):
            return str(result[k])[:200]
    if "steps" in result:
        return f"{len(result['steps'])} steps"
    if "rows" in result:
        return f"{result.get('total_rows', len(result['rows']))} rows from {result.get('table')}"
    if "gates" in result:
        return f"{len(result['gates'])} gates in {result.get('sample', '')}"
    if "samples" in result:
        return f"{len(result['samples'])} samples"
    return "done"


def complete_steps(args: str) -> list[dict]:
    """The step objects a plan's arguments hold so far, while the model is still writing them: every
    object in the "steps" array that has closed. The JSON as a whole is unfinished, so it is scanned."""
    at = args.find('"steps"')
    at = args.find("[", at) if at >= 0 else -1
    if at < 0:
        return []
    out: list[dict] = []
    depth, start, in_str, esc = 0, -1, False, False
    for i in range(at + 1, len(args)):
        ch = args[i]
        if in_str:
            esc = (not esc) and ch == "\\"
            in_str = esc or ch != '"'
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            start = i if depth == 0 else start
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    out.append(json.loads(args[start:i + 1]))
                except json.JSONDecodeError:
                    pass
        elif ch == "]" and depth == 0:
            break
    return out


def _preview(step: dict) -> dict:
    """A step as the canvas draws it before the plan is checked: its kind from its op."""
    op = str(step.get("op", ""))
    return {"id": str(step.get("id", "")), "op": op, "kind": plan.OPS.get(op, ("macro",))[0],
            "title": str(step.get("title", op.replace("_", " "))), "inputs": step.get("inputs") or [],
            "outputs": step.get("outputs") or [], "why": ""}


def run_turn(exp_id: str, text: str, attachments: list[str] | None = None, *,
             thread: str = "chat", screen: dict | None = None,
             on_event: Callable[[dict], None] | None = None) -> dict:
    """One message in, one answer out, with everything the screen needs.

    thread "lab_chat" is the data lab's assistant: `screen` is what the lab shows right now.
    `on_event` hears the turn as it happens: each model round ("thinking"), each tool as it starts
    ("step") and ends ("step_done"), and the screen changes it asks for ("ui"), so the chat can
    show the work instead of a long wait.
    """
    emit = on_event or (lambda _e: None)
    sent_ui = 0
    shown = 0                      # steps shown live so far: the tools, and the web searches inside a round
    searching: dict[str, int] = {}
    plan_args: dict[str, str] = {}
    plan_seen: dict[str, int] = {}

    def on_delta(d: dict) -> None:
        """The model's words, its web searches and its plan, passed on as it writes them."""
        nonlocal shown
        if d["kind"] == "text":
            emit({"type": "text", "delta": d["delta"]})
        elif d["kind"] == "search":
            searching[d["item"]] = shown
            emit({"type": "step", "i": shown, "tool": "web_search", "label": "Searching the web"})
            shown += 1
        elif d["kind"] == "search_done" and d["item"] in searching:
            emit({"type": "step_done", "i": searching[d["item"]], "ok": d["ok"], "summary": web.search_summary(d["action"])[1]})
        elif d["kind"] == "args" and d["name"] == "plan_pipeline":
            buf = plan_args[d["item"]] = plan_args.get(d["item"], "") + d["delta"]
            steps = complete_steps(buf)
            if len(steps) > plan_seen.get(d["item"], 0):
                plan_seen[d["item"]] = len(steps)
                emit({"type": "plan_partial", "steps": [_preview(s) for s in steps]})
    t0 = time.time()
    in_lab = thread == "lab_chat"
    store.add_message(exp_id, {"role": "user", "text": text, "attachments": attachments or []}, thread)
    ctx = tools.Ctx(exp_id, user_text=text)
    if on_event:
        # screen events go out the moment a tool emits them (a run the canvas should follow live)
        ctx.live = lambda e: emit({"type": "ui", "events": [e]})
    remembered = tools.memories_for_prompt(text)
    if in_lab:
        instructions = (prompts.LAB_AGENT + "\n\nOn screen now:\n" + json.dumps(screen or {}, default=str)
                        + "\n\nThe data:\n" + json.dumps(tools.lab_snapshot(exp_id), default=str) + remembered)
        schemas = tools.schemas(list(tools.LAB_TOOLS))
        rounds = min(6, settings.agent_max_rounds)
    else:
        instructions = (prompts.AGENT + "\n\n" + web.GUIDE + "\n\nThe experiment right now:\n"
                        + json.dumps(tools.snapshot(exp_id), default=str) + remembered)
        schemas = tools.schemas([n for n in tools.REGISTRY if n not in tools.LAB_ONLY]) + [web.TOOL]
        rounds = settings.agent_max_rounds
    items: list[dict] = _history(exp_id, thread)
    trace: list[dict] = []
    sources: list[dict] = []       # the pages the answer cites, numbered as in its text
    reply_text, usage = "", {"input": 0, "output": 0}
    error = None
    try:
        last = rounds - 1
        for n in range(rounds):
            # On the last round tools are withheld, so the turn always ends with an answer.
            emit({"type": "thinking", "round": n})
            r = client.respond(instructions, items, schemas if n < last else None, on_delta=on_delta if on_event else None)
            usage = {k: usage[k] + r.usage.get(k, 0) for k in usage}
            items += r.items
            trace += [{"tool": "web_search", "args": {}, "ok": s["ok"], "summary": web.search_summary(s["action"])[1]}
                      for s in r.searches]
            if not r.calls:
                reply_text, sources = r.text, r.sources
                break
            for c in r.calls:
                step, shown = shown, shown + 1
                emit({"type": "step", "i": step, "tool": c.name, "label": tools.label(c.name, c.args)})
                result = tools.call(ctx, c.name, c.args)
                trace.append({"tool": c.name, "args": _short(c.args), "ok": result.get("ok", False),
                              "summary": _summary(c.name, result)})
                emit({"type": "step_done", "i": step, "ok": trace[-1]["ok"], "summary": trace[-1]["summary"]})
                if len(ctx.ui) > sent_ui and not ctx.live:
                    emit({"type": "ui", "events": ctx.ui[sent_ui:]})
                sent_ui = len(ctx.ui)
                store.journal(exp_id, "tool", {"tool": c.name, "ok": result.get("ok"), "args": _short(c.args)})
                items.append(client.tool_result(c, result))
    except client.NotConfigured as e:
        error = str(e)
        reply_text = f"The agent is not set up yet: {e}."
    except Exception as e:
        log.exception("agent turn failed")
        error = f"{e.__class__.__name__}: {e}"
        reply_text = "Something went wrong reaching the model, so nothing was changed. Try again in a moment."
    d = None if in_lab else store.draft(exp_id)
    msg = store.add_message(exp_id, {"role": "assistant", "text": reply_text, "trace": trace, "ui": ctx.ui, "sources": sources,
                                     "draft": {"id": d["id"], "kind": d["kind"], "summary": d["summary"]} if d else None,
                                     "error": error, "seconds": round(time.time() - t0, 1), "usage": usage}, thread)
    return msg


def run_subagent(instructions: str, task: str, tool_names: list[str] | None = None,
                 exp_id: str | None = None, rounds: int = 4) -> dict:
    """A narrower agent for one delegated job. Returns its final answer parsed as JSON.

    Without tools it is a single structured call (the macro writer). With tools it loops like
    the main agent, but only over the tools it was given.
    """
    items: list[dict] = [{"role": "user", "content": task}]
    schemas = tools.schemas(tool_names) if tool_names else None
    ctx = tools.Ctx(exp_id or "")
    for _ in range(rounds):
        r = client.respond(instructions, items, schemas, json_mode=not schemas)
        items += r.items
        if not r.calls:
            try:
                return json.loads(r.text)
            except json.JSONDecodeError as e:
                raise ValueError(f"the subagent did not return JSON: {e}") from e
        for c in r.calls:
            items.append(client.tool_result(c, tools.call(ctx, c.name, c.args)))
    raise ValueError("the subagent ran out of rounds")
