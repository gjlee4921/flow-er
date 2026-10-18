"""The model behind the agent, behind one function so the provider can change.

The web app runs on OpenAI's Responses API: it supports function tools with reasoning models,
and `store=False` keeps the conversation off OpenAI's side (reasoning comes back encrypted so a
turn can still chain). Anthropic slots in here with the same shape when the app moves to Claude.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from flow_er.config import settings
from flow_er.llm import web


@dataclass
class Call:
    id: str
    name: str
    args: dict


@dataclass
class Reply:
    text: str
    calls: list[Call]
    items: list[dict]          # everything the model returned, to pass back on the next round
    usage: dict = field(default_factory=dict)
    sources: list[dict] = field(default_factory=list)   # pages the text cites, numbered as in the text
    searches: list[dict] = field(default_factory=list)  # the hosted web searches this round ran


class NotConfigured(RuntimeError):
    pass


_client = None


def _openai():
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise NotConfigured("OPENAI_API_KEY is not set in backend/.env")
        from openai import OpenAI
        _client = OpenAI(api_key=settings.openai_api_key, timeout=120, max_retries=1)
    return _client


def tool_schema(name: str, description: str, params: dict, required: list[str]) -> dict:
    return {"type": "function", "name": name, "description": description, "strict": False,
            "parameters": {"type": "object", "properties": params, "required": required}}


def _streamed(kwargs: dict, on_delta: Callable[[dict], None]):
    """The same call, streamed: text and tool arguments are passed on piece by piece as the model
    writes them; the finished response is returned as if it had not been streamed."""
    names: dict[str, str] = {}
    final = None
    for ev in _openai().responses.create(**kwargs, stream=True):
        t = ev.type
        if t == "response.output_item.added" and getattr(ev.item, "type", "") == "function_call":
            names[ev.item.id] = ev.item.name
            on_delta({"kind": "call", "name": ev.item.name, "item": ev.item.id})
        elif t == "response.output_item.added" and getattr(ev.item, "type", "") == "web_search_call":
            on_delta({"kind": "search", "item": ev.item.id})
        elif t == "response.output_item.done" and getattr(ev.item, "type", "") == "web_search_call":
            on_delta({"kind": "search_done", "item": ev.item.id, "action": _action(ev.item),
                      "ok": getattr(ev.item, "status", "") == "completed"})
        elif t == "response.function_call_arguments.delta":
            on_delta({"kind": "args", "name": names.get(ev.item_id, ""), "item": ev.item_id, "delta": ev.delta})
        elif t == "response.output_text.delta":
            on_delta({"kind": "text", "delta": ev.delta})
        elif t in ("response.completed", "response.incomplete"):
            final = ev.response
        elif t in ("response.failed", "error"):
            err = getattr(getattr(ev, "response", None), "error", None) or getattr(ev, "message", None)
            raise RuntimeError(f"the model stream failed: {err}")
    if final is None:
        raise RuntimeError("the model stream ended without a response")
    return final


def respond(instructions: str, items: list[dict], tools: list[dict] | None = None,
            max_output_tokens: int = 4000, json_mode: bool = False,
            on_delta: Callable[[dict], None] | None = None) -> Reply:
    """One round: send the conversation so far, get text and/or tool calls back. With `on_delta`
    the round is streamed and each piece of text or tool arguments is passed on as it comes."""
    if settings.llm_provider != "openai":
        raise NotConfigured(f"provider '{settings.llm_provider}' is not wired yet; set LLM_PROVIDER=openai")
    kwargs: dict = {"model": settings.openai_model, "instructions": instructions, "input": items, "store": False,
                     "include": ["reasoning.encrypted_content"], "max_output_tokens": max_output_tokens}
    if tools:
        kwargs["tools"] = tools
    if json_mode:
        kwargs["text"] = {"format": {"type": "json_object"}}
        # the API refuses JSON mode unless the word "json" is in the input itself (the instructions do not count)
        if "json" not in json.dumps(items, default=str).lower():
            kwargs["input"] = [*items, {"role": "user", "content": "Reply with one JSON object."}]
    r = _streamed(kwargs, on_delta) if on_delta else _openai().responses.create(**kwargs)
    calls, searches, parts = [], [], []
    for o in r.output:
        if o.type == "function_call":
            try:
                args = json.loads(o.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparsed": o.arguments}
            calls.append(Call(id=o.call_id, name=o.name, args=args))
        elif o.type == "web_search_call":
            searches.append({"action": _action(o), "ok": getattr(o, "status", "") == "completed"})
        elif o.type == "message":
            for c in o.content or []:
                if getattr(c, "type", "") == "output_text":
                    parts.append((c.text, [{"start": a.start_index, "end": a.end_index, "url": a.url, "title": a.title}
                                           for a in (c.annotations or []) if getattr(a, "type", "") == "url_citation"]))
    text, sources = web.numbered(parts) if any(cites for _, cites in parts) else (r.output_text or "", [])
    usage = {"input": getattr(r.usage, "input_tokens", 0), "output": getattr(r.usage, "output_tokens", 0)}
    return Reply(text=text, calls=calls, items=[o.model_dump(exclude_none=True) for o in r.output], usage=usage,
                 sources=sources, searches=searches)


def _action(item) -> dict:
    """What one hosted web-search step did: a search, a page opened, or a phrase looked for."""
    a = getattr(item, "action", None)
    return a.model_dump(exclude_none=True) if hasattr(a, "model_dump") else dict(a or {})


def tool_result(call: Call, result: dict) -> dict:
    return {"type": "function_call_output", "call_id": call.id, "output": json.dumps(result, default=str)[:20000]}
