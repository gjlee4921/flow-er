"""Gating on an image as three steps (spec §3, Image steps): render (a built-in step, the gate models'
recipe), identify (a script: scikit-image finds the populations) and convert (a script: regions become
gate polygons and counts). Identify is the step a gate model (§2) replaces when Gijeong's code lands.

The two scripts are ordinary script steps with their provenance, fixture and test, so they run in the
sandbox, show their code in the panel, and can be edited or swapped like any other.
"""
from __future__ import annotations

import re

from flow_er.workspace import scripts

IDENTIFY = '''
PROVENANCE = {"what": "Find the populations on a density image: smooth it, mark one peak per population, and grow each peak's region by watershed down to a floor, so neighbouring populations split along the valley between them.",
              "source": None, "version": None, "licence": None,
              "adapted": "written for flow-er; calls scikit-image (BSD-3-Clause) peak_local_max and watershed, and scipy's gaussian_filter"}
INPUTS = {"density": "image"}
OUTPUTS = {"regions": "image", "peaks": "table"}
PARAMS = {"sigma": 1.5, "min_distance": 8, "peak_floor": 0.25, "region_floor": 0.2, "max_regions": 6}

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def centre(edges, i):
    # the display value at the middle of pixel i
    return round(float(edges[0] + (i + 0.5) * (edges[1] - edges[0])), 4)


def run(inputs, params):
    regions, rows = {}, []
    for name, img in sorted(inputs["density"].items()):
        counts = np.asarray(img["grid"], dtype=float)
        g = np.log1p(counts)  # the gate models' normalisation: log1p(count) / max
        g = g / g.max() if g.max() > 0 else g
        s = ndi.gaussian_filter(g, float(params["sigma"]))
        s = s / s.max() if s.max() > 0 else s
        peaks = peak_local_max(s, min_distance=int(params["min_distance"]), threshold_abs=float(params["peak_floor"]),
                               num_peaks=int(params["max_regions"]), exclude_border=False)
        markers = np.zeros(s.shape, dtype=int)
        for k, (r, c) in enumerate(peaks, 1):
            markers[r, c] = k
        labels = watershed(-s, markers, mask=s >= float(params["region_floor"]))
        regions[name] = {"grid": labels.astype("float32"), "x_edges": img["x_edges"], "y_edges": img["y_edges"],
                         "meta": {**img.get("meta", {}), "kind": "regions"}}
        total = counts.sum()
        for k, (r, c) in enumerate(peaks, 1):
            n = counts[labels == k].sum()
            rows.append({"file": name, "region": k, "peak_x": centre(img["x_edges"], c), "peak_y": centre(img["y_edges"], r),
                         "events": int(n), "pct_of_parent": round(100 * n / total, 3) if total else 0.0})
    cols = ["file", "region", "peak_x", "peak_y", "events", "pct_of_parent"]
    return {"regions": regions, "peaks": pd.DataFrame(rows, columns=cols)}
'''

CONVERT = '''
PROVENANCE = {"what": "Turn each region found on a density image into a gate polygon (its outline, simplified, in the image's display units) and count the events inside it as % of the population drawn.",
              "source": None, "version": None, "licence": None,
              "adapted": "written for flow-er; calls scikit-image (BSD-3-Clause) find_contours and approximate_polygon"}
INPUTS = {"density": "image", "regions": "image"}
OUTPUTS = {"polygons": "table", "counts": "table"}
PARAMS = {"tolerance": 0.75}

import json

import numpy as np
import pandas as pd
from skimage.measure import approximate_polygon, find_contours


def run(inputs, params):
    polys, counts = [], []
    for name, reg in sorted(inputs["regions"].items()):
        labels = np.rint(np.asarray(reg["grid"])).astype(int)
        dens = np.asarray(inputs["density"][name]["grid"], dtype=float)
        xe, ye = np.asarray(reg["x_edges"], dtype=float), np.asarray(reg["y_edges"], dtype=float)
        dx, dy = xe[1] - xe[0], ye[1] - ye[0]
        total = dens.sum()
        for k in sorted(int(v) for v in np.unique(labels) if v > 0):
            outline = max(find_contours(np.pad(labels == k, 1).astype(float), 0.5), key=len)
            outline = approximate_polygon(outline, tolerance=float(params["tolerance"])) - 1  # undo the padding
            # pixel (row, col) to display units, the middle of pixel i being edges[0] + (i + 0.5) * step
            verts = [[round(float(xe[0] + (c + 0.5) * dx), 4), round(float(ye[0] + (r + 0.5) * dy), 4)] for r, c in outline[:-1]]
            n = dens[labels == k].sum()
            polys.append({"file": name, "region": k, "corners": len(verts), "vertices": json.dumps(verts)})
            counts.append({"file": name, "region": k, "events": int(n), "pct_of_parent": round(100 * n / total, 3) if total else 0.0})
    return {"polygons": pd.DataFrame(polys, columns=["file", "region", "corners", "vertices"]),
            "counts": pd.DataFrame(counts, columns=["file", "region", "events", "pct_of_parent"])}
'''


def _blocks() -> tuple[list[list[float]], list[list[float]]]:
    """A 21 × 21 fixture: a block of 9 pixels with 10 events each, and one with 5 each, far apart."""
    g = [[0.0] * 21 for _ in range(21)]
    lab = [[0.0] * 21 for _ in range(21)]
    for r in range(3, 6):
        for c in range(3, 6):
            g[r][c], lab[r][c] = 10.0, 1.0
    for r in range(14, 17):
        for c in range(13, 16):
            g[r][c], lab[r][c] = 5.0, 2.0
    return g, lab


def fixtures() -> dict:
    """Hand-worked answers: 90 events in the first block (peak at pixel 4, 4 = 0.2, 0.2 on 0.05 steps),
    45 in the second (peak at column 14, row 15 = 0.7, 0.75); 90 / 135 = 66.667%, 45 / 135 = 33.333%."""
    g, lab = _blocks()
    edges = [round((i - 0.5) / 20, 4) for i in range(22)]
    img = {"f1": {"grid": g, "x_edges": edges, "y_edges": edges, "meta": {}}}
    regions = {"f1": {"grid": lab, "x_edges": edges, "y_edges": edges, "meta": {}}}
    return {
        "identify": {"inputs": {"density": img}, "params": {"sigma": 0.8, "min_distance": 3, "peak_floor": 0.3, "region_floor": 0.2},
                     "expected": {"peaks": {"columns": ["file", "region", "peak_x", "peak_y", "events", "pct_of_parent"],
                                            "rows": [["f1", 1, 0.2, 0.2, 90, 66.667], ["f1", 2, 0.7, 0.75, 45, 33.333]]}}},
        "convert": {"inputs": {"density": img, "regions": regions},
                    "expected": {"counts": {"columns": ["file", "region", "events", "pct_of_parent"],
                                            "rows": [["f1", 1, 90, 66.667], ["f1", 2, 45, 33.333]]}}},
    }


def steps(parent: str, x: str, y: str, short: dict[str, str] | None = None, taken: set[str] | None = None) -> list[dict]:
    """The three steps, wired and tested, ready to go into a plan after the step that makes the gates."""
    short = short or {}
    name = re.sub(r"[^a-z0-9]+", "_", f"{short.get(x, x)}_{short.get(y, y)}".lower()).strip("_")[:20] or "image"
    base, taken, n = f"img_{name}", taken or set(), 2
    while any(i in taken for i in (f"{base}_render", f"{base}_find", f"{base}_gates")):
        base, n = f"img_{name}_{n}", n + 1
    fx = fixtures()
    tested = {}
    for key, code in (("identify", IDENTIFY), ("convert", CONVERT)):
        t = scripts.test(code, fx[key])
        if not t["passed"]:
            raise ValueError(f"the built-in {key} script failed its own test: {'; '.join(t['problems'])}")
        tested[key] = {"code": code, "provenance": scripts.check(code)["provenance"], "fixture": fx[key], "test": t}
    pop = parent.split("/")[-1] if parent else "all events"
    return [
        {"id": f"{base}_render", "op": "render_image", "title": f"Density image: {pop} on {short.get(x, x)} × {short.get(y, y)}",
         "why": "Bins the population's events the way the gate models read them (recipe 1bd7b6e6).",
         "inputs": ["fcs"], "outputs": [f"{base}_density"], "params": {"x": x, "y": y, "parent": parent}},
        {"id": f"{base}_find", "op": "custom_macro", "kind": "model", "title": f"Find the populations in {pop}",
         "why": "Peaks and watershed on the density image (scikit-image); a gate model can take this step's place later.",
         "inputs": [f"{base}_density"], "outputs": [f"{base}_find_regions", f"{base}_find_peaks"], "script": tested["identify"]},
        {"id": f"{base}_gates", "op": "custom_macro", "kind": "measure", "title": f"Gates and counts from the regions in {pop}",
         "why": "Each region's outline becomes a gate polygon; events counted per region.",
         "inputs": [f"{base}_density", f"{base}_find_regions"], "outputs": [f"{base}_gates_polygons", f"{base}_gates_counts"],
         "script": tested["convert"]},
    ]
