"""The agent's tools for the results page: write its words (a title, a summary and key numbers, as a
draft), read a figure back as numbers so the main chat and the lab assistant can explain it, and put a
saved figure on screen in the data lab.

Registered into the shared tool registry when `tools` imports this module at its end.
"""
from __future__ import annotations

import re

from flow_er.llm.tools import Ctx, tool
from flow_er.workspace import report, store


@tool("read_figure",
      "Read a saved figure (a diagram on the results page or in the data lab's Saved diagrams) the way it is "
      "drawn: its type, table, axes and caption, and the numbers behind each mark (mean, SD, n and range per x "
      "and group, and the fold change from the baseline group when it compares groups). Use it to explain or "
      "check a figure; quote only these numbers. figure: its number ('Figure 1'), or words from its title; "
      "omit for every figure.",
      {"figure": {"type": "string"}})
def _read_figure(ctx: Ctx, figure: str | None = None) -> dict:
    figs = _pick(ctx, figure, one=False)
    out = [report.read_figure(ctx.exp_id, f) for f in figs]
    return {"figures": out, "summary": "read " + ", ".join(f"{o['figure']} ({o['title']})" for o in out)}


def _pick(ctx: Ctx, figure: str | None, one: bool) -> list[dict]:
    """The figures a request names: by number ('Figure 2'), by words of its title, else by the most words
    shared with its title, caption and kind ('the mice diagram' finds the mouse map), else the only one."""
    figs = report.figures(ctx.exp_id)
    if not figs:
        raise ValueError("there are no figures yet; ask the main chat to draw one")
    if not figure:
        return figs[-1:] if one else figs
    num = re.fullmatch(r"\s*(?:fig(?:ure)?\.?\s*)?(\d+)\s*", figure, re.IGNORECASE)
    if num:
        want = [f for f in figs if f["n"] == int(num.group(1))]
    else:
        want = [f for f in figs if figure.lower() in f["spec"]["title"].lower()]
        if not want:
            words = set(re.findall(r"[a-z0-9+]{3,}", figure.lower())) - {"the", "diagram", "figure", "chart", "plot", "show"}
            text = lambda f: " ".join([f["spec"]["title"], f["spec"].get("caption", ""), f["spec"]["type"],
                                       "mouse mice organ anatomy" if f["spec"]["type"] == "anatomy" else ""]).lower()
            scored = sorted(((len(words & set(re.findall(r"[a-z0-9+]{3,}", text(f)))), f) for f in figs), key=lambda t: -t[0])
            want = [f for s, f in scored if s and s == scored[0][0]] or (figs if len(figs) == 1 else [])
    if not want:
        raise ValueError(f"no figure matches '{figure}'; the figures are: "
                         + "; ".join(f"Figure {f['n']} {f['spec']['title']}" for f in figs))
    return want[:1] if one else want


@tool("show_figure",
      "Put a saved figure on screen: switch the data lab to Saved diagrams with that figure open, and get the "
      "numbers behind it (as read_figure gives) to explain what is on screen. Use it whenever the scientist asks "
      "to see, show or open a figure. figure: its number ('Figure 1') or words from its title; omit for the latest.",
      {"figure": {"type": "string"}})
def _show_figure(ctx: Ctx, figure: str | None = None) -> dict:
    f = _pick(ctx, figure, one=True)[0]
    if f["draft"]:
        raise ValueError(f"Figure {f['n']} is still a draft on the results page; keep it first and it joins Saved diagrams")
    ctx.emit(type="lab", view="saved", fig=f["spec"]["id"])
    out = report.read_figure(ctx.exp_id, f)
    return {"shown": out["figure"], "figure": out, "summary": f"opened {out['figure']} ({out['title']}) in Saved diagrams"}


@tool("write_report",
      "Draft the results page's words: a title that says what was found (not what was done), a summary of "
      "3-5 sentences, and up to 5 key numbers. Every number must be a reference to a saved table cell, never "
      "typed: {{population_stats | gate=CD4+ | sample=E01_008 | pct_parent}}; gate may be a name or the end "
      "of a path (CD4+/IFNg+), sample a piece of the file name. For one number across files end with mean, "
      "median, min, max or range: {{population_stats | gate=CD4+ | pct_parent | range}}. Typed percentages "
      "or decimals are refused. A reference shows its own unit (61.7%), so write no % after it. Keep the title "
      "under 90 characters. Figures, methods and the data list are added by the page itself.",
      {"title": {"type": "string"},
       "summary": {"type": "string", "description": "3-5 sentences; numbers only as {{...}} references"},
       "key_numbers": {"type": "array", "items": {"type": "object", "properties": {
           "label": {"type": "string", "description": "what it is, e.g. 'CD4+ of CD3+, range across files'"},
           "value": {"type": "string", "description": "one reference, e.g. population_stats | gate=CD4+ | pct_parent | range"}},
           "required": ["label", "value"]}}},
      ["title", "summary"])
def _write_report(ctx: Ctx, title: str, summary: str, key_numbers: list[dict] | None = None) -> dict:
    r = report.validate(ctx.exp_id, title, summary, key_numbers)
    d = store.set_draft(ctx.exp_id, "report", r, f"The results page: {r['title']}")
    ctx.emit(type="open", page="results")
    ctx.emit(type="refresh")
    shown = [s["number"]["display"] for s in report.segments(ctx.exp_id, r["summary"]) if "number" in s]
    return {"draft": d["id"], "numbers_shown": shown,
            "summary": f"drafted the results page with {len(shown)} bound numbers and {len(r['key_numbers'])} key numbers"}
