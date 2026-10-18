"""The data lab's toolkit, defined once. The toolbar lists its plot types, the sheet offers its
statistics as columns, the lab assistant's tools take their names from it, and report widgets will
draw from it, so a new plot type or statistic is one entry here. Entries marked later are listed so
the assistant can say a tool is not there yet and name the closest one that is.

Statistics follow FlowJo's definitions (docs.flowjo.com, statistic definitions); FlowJo computes them
on binned data, ours use every event, so the last digit can differ.
"""
from __future__ import annotations

import numpy as np

PLOT_TYPES = [
    {"id": "density", "name": "Density", "axes": 2, "render": "density",
     "about": "Events binned on two parameters; colour shows how many fall in each bin."},
    {"id": "dots", "name": "Dots", "axes": 2, "render": "dots", "about": "Every drawn event as a dot."},
    {"id": "contour", "name": "Contour", "axes": 2, "render": "contour",
     "about": "Lines of equal event density, like a map's height lines; populations read as islands."},
    {"id": "hist", "name": "Histogram", "axes": 1, "render": "hist",
     "about": "One parameter: the share of events at each level."},
]
STATISTICS = [
    {"id": "count", "name": "Count", "needs": []},
    {"id": "pct_parent", "name": "% of parent", "needs": []},
    {"id": "pct_grandparent", "name": "% of grandparent", "needs": []},
    {"id": "pct_of", "name": "% of an ancestor", "needs": ["ancestor"]},
    {"id": "pct_total", "name": "% of total", "needs": []},
    {"id": "mean", "name": "Mean", "needs": ["parameter"]},
    {"id": "median", "name": "Median", "needs": ["parameter"]},
    {"id": "geo_mean", "name": "Geometric mean", "needs": ["parameter"]},
    {"id": "mode", "name": "Mode", "needs": ["parameter"]},
    {"id": "sd", "name": "SD", "needs": ["parameter"]},
    {"id": "cv", "name": "CV", "needs": ["parameter"]},
    {"id": "rsd", "name": "Robust SD", "needs": ["parameter"]},
    {"id": "rcv", "name": "Robust CV", "needs": ["parameter"]},
    {"id": "percentile", "name": "Percentile", "needs": ["parameter", "p"]},
    {"id": "mad", "name": "Median absolute deviation", "needs": ["parameter"]},
]
GATE_TOOLS = [
    {"id": "rect", "name": "Rectangle", "key": "R", "about": "Drag a box."},
    {"id": "poly", "name": "Polygon", "key": "P", "about": "Click the corners; click the first one again to close."},
    {"id": "quadrant", "name": "Quadrant", "key": "Q", "about": "Click the centre: four gates, one per quarter."},
    {"id": "range", "name": "Range", "key": "G", "about": "On a histogram, drag across the levels to keep."},
]
LATER = [
    "zebra plot", "CDF plot", "colour by a third parameter or a statistic", "linear, log, logicle or arcsinh per axis; axis range",
    "overlaid histograms by file or condition", "backgating", "statistics on a derived parameter", "pivot the sheet",
    "a sheet column from a formula (e.g. CD4/CD8)", "data bars", "save the sheet as a table", "box and whisker", "scatter of two statistics",
    "line across time points or doses", "heat map of gate × file", "derived parameters (ratio or sum of channels)", "UMAP or tSNE",
    "ellipse, freehand, spider quadrant, bisector and Boolean gates", "SVG export", "a layout of chosen plots", "the same layout for every file",
]

STAT_IDS = {s["id"] for s in STATISTICS}
PLOT_IDS = {p["id"] for p in PLOT_TYPES}


def catalogue() -> dict:
    return {"plot_types": PLOT_TYPES, "statistics": STATISTICS, "gate_tools": GATE_TOOLS, "later": LATER}


def column_label(stat: str, parameter: str | None = None, arg: str | None = None, names: dict | None = None) -> str:
    """How a statistic column reads in the sheet: 'Median CD4', 'P90 CD3', '% of Singlets'."""
    short = (names or {}).get(parameter or "", parameter or "")
    if stat == "percentile":
        return f"P{arg} {short}"
    if stat == "pct_of":
        return f"% of {(arg or '').split('/')[-1] or 'all events'}"
    name = next((s["name"] for s in STATISTICS if s["id"] == stat), stat)
    return f"{name} {short}".strip() if parameter else name


def value(stat: str, v: np.ndarray, arg: str | None = None, mode_of=None) -> float | None:
    """One statistic of one parameter's values (compensated, before display scaling). `mode_of`
    turns the values into the mode, which is found on the display scale where the peak is."""
    if not len(v):
        return None
    if stat == "mean":
        return float(v.mean())
    if stat == "median":
        return float(np.median(v))
    if stat == "geo_mean":
        pos = v[v > 0]
        return float(np.exp(np.log(pos).mean())) if len(pos) else None
    if stat == "mode":
        return mode_of(v) if mode_of else None
    if stat == "sd":
        return float(v.std(ddof=1)) if len(v) > 1 else None
    if stat == "cv":
        m = v.mean()
        return float(v.std(ddof=1) / abs(m) * 100) if len(v) > 1 and m else None
    if stat == "rsd":
        lo, hi = np.percentile(v, [15.87, 84.13])
        return float((hi - lo) / 2)
    if stat == "rcv":
        lo, hi = np.percentile(v, [15.87, 84.13])
        med = np.median(v)
        return float(100 * (hi - lo) / 2 / abs(med)) if med else None
    if stat == "percentile":
        return float(np.percentile(v, float(arg or 50)))
    if stat == "mad":
        return float(np.median(np.abs(v - np.median(v))))
    raise ValueError(f"no statistic '{stat}'; the toolkit has {sorted(STAT_IDS)}")
