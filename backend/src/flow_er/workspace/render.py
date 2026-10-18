"""Density images of a population on two channels, made the way Gijeong's gate models expect them
(spec §2.1, recipe 1bd7b6e6), so a script and a model read the same picture.

101 × 101 bins of the parent population's events, rows = y channel. Scatter channels are linear and
fluorescence channels logicle (w 0.5, m 4.5), both with T from the file's $PnR; a pixel is
round(100 × v) clipped to 0–100 on the display value v. The image holds event counts; models
normalise them as log1p(count) / max themselves.

Regions found on an image come back as polygons in this recipe's display space; `to_lab` turns them
into the lab's display space, where gates are drawn (the same space when the lab uses the models'
display; see `data.py`).
"""
from __future__ import annotations

import json
from pathlib import Path

import flowkit as fk
import numpy as np

from flow_er.workspace import lab
from flow_er.workspace.data import SCATTER

RECIPE = "1bd7b6e6"
BINS = 101


def _range(ds, sid: str, pnn: str) -> float:
    s = ds._sample(sid)
    n = list(s.pnn_labels).index(pnn) + 1
    return float(s.metadata.get(f"p{n}r") or 262144)


def transform(ds, sid: str, pnn: str):
    """The recipe's transform for one channel of one file."""
    t = _range(ds, sid, pnn)
    if SCATTER.match(pnn) or pnn.lower() == "time":
        return fk.transforms.LinearTransform(param_t=t, param_a=0)
    return fk.transforms.LogicleTransform(param_t=t, param_w=0.5, param_m=4.5, param_a=0)


def render(exp_id: str, sid: str, x: str, y: str, parent: str | None) -> dict:
    """One file's density image: counts per pixel, the pixel edges in display units, and what maps it
    back to events (file, population, channels, transforms, recipe)."""
    ds = lab.dataset(exp_id)
    cols = list(ds._sample(sid).pnn_labels)
    for c in (x, y):
        if c not in cols:
            raise ValueError(f"no channel {c} in {sid}")
    ev = ds.compensated(sid)[lab.membership(exp_id, sid, parent or None)]
    px = [np.clip(np.rint(100 * transform(ds, sid, c).apply(ev[:, cols.index(c)])), 0, 100).astype(int) for c in (x, y)]
    grid = np.bincount(px[1] * BINS + px[0], minlength=BINS * BINS).reshape(BINS, BINS).astype("float32")
    edges = (np.arange(BINS + 1) - 0.5) / 100
    return {"grid": grid, "x_edges": edges, "y_edges": edges.copy(),
            "meta": {"recipe": RECIPE, "sample": sid, "parent": parent or "", "x": x, "y": y, "events": len(ev),
                     "transforms": {c: {"t": _range(ds, sid, c), "kind": "linear" if SCATTER.match(c) or c.lower() == "time" else "logicle"}
                                    for c in (x, y)}}}


def save(folder: Path, sid: str, img: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / f"{sid}.npz", grid=np.asarray(img["grid"], dtype="float32"),
             x_edges=np.asarray(img["x_edges"]), y_edges=np.asarray(img["y_edges"]))
    (folder / f"{sid}.json").write_text(json.dumps(img.get("meta") or {}))


def load(folder: Path, sid: str) -> dict:
    z = np.load(folder / f"{sid}.npz", allow_pickle=False)
    meta = folder / f"{sid}.json"
    return {"grid": z["grid"], "x_edges": z["x_edges"], "y_edges": z["y_edges"],
            "meta": json.loads(meta.read_text()) if meta.exists() else {}}


def to_lab(exp_id: str, sid: str, x: str, y: str, vertices: list[list[float]]) -> list[list[float]]:
    """Polygon corners in the recipe's display space, moved into the lab's display space: back to the
    channel's values with the recipe's transform, then forward with the lab's."""
    ds = lab.dataset(exp_id)
    lab_xf = ds._transforms(sid)
    v = np.asarray(vertices, dtype=float)
    out = []
    for j, c in enumerate((x, y)):
        raw = transform(ds, sid, c).inverse(v[:, j])
        out.append(lab_xf[c].apply(raw) if c in lab_xf and lab_xf[c] is not None else raw)
    return [[float(a), float(b)] for a, b in zip(*out)]
