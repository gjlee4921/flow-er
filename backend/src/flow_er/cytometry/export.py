"""Export gates as GatingML 2.0 (FlowKit): polygon gates in raw (compensated) units.

A polygon is straight-edged in the shared display space, so each edge is first densified to
DENSIFY display units (about half a pixel) and then taken to compensated values: the raw-unit
polygon follows the curved image of the displayed edge. Compensated dimensions reference the
spillover matrix used for gating ("spill"), which is written into the same file.
"""

import re
from pathlib import Path

import flowkit as fk
import numpy as np

from flow_er.ml import images

DENSIFY = 0.005
MATRIX_ID = "spill"


def densify(polygon: np.ndarray, step: float = DENSIFY) -> np.ndarray:
    """The closed polygon with extra vertices so no edge is longer than `step`."""
    out = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.extend(a + (b - a) * t for t in np.arange(n) / n)
    return np.asarray(out)


def xml_id(name: str) -> str:
    """A gate name as a GatingML gate id (xs:ID: letters, digits, '.', '-', '_')."""
    s = re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9._-]", "_", name)).strip("_") or "gate"
    return s if re.match(r"[A-Za-z_]", s) else f"_{s}"


def gatingml_ids(tree) -> dict[str, str]:
    """Gate key -> unique GatingML id; a name used twice gets its parent's id appended."""
    ids = [xml_id(s.name) for s in tree.gates]
    return {s.key: i if ids.count(i) == 1 else
            f"{i}__{xml_id(s.parent.split('/')[-1]) if s.parent else 'root'}"
            for s, i in zip(tree.gates, ids)}


def to_gatingml(result, tree, path: str | Path) -> Path:
    """Write a GatingResult (ml/runtime.py) as GatingML; gates without a polygon are skipped
    together with everything below them."""
    gs = fk.GatingStrategy()
    if result.comp is not None:
        spill, detectors = result.comp
        gs.add_comp_matrix(MATRIX_ID, fk.Matrix(np.asarray(spill, float), list(detectors),
                                                fluorochromes=list(detectors)))
    export_name = gatingml_ids(tree)
    skipped = set()
    for spec in tree.gates:
        g = result.gates.get(spec.key)
        if g is None or g.polygon_display is None or spec.parent in skipped:
            skipped.add(spec.key)
            continue
        chans = [spec.x, spec.y]
        raw = images.from_display(densify(np.asarray(g.polygon_display)), chans,
                                  [result.channels[c] for c in chans])
        dims = [fk.Dimension(c, compensation_ref=MATRIX_ID if comp and result.comp is not None
                             else "uncompensated") for c, comp in zip(chans, spec.compensated)]
        parts = spec.parent.split("/") if spec.parent else []
        parent_path = ("root", *(export_name["/".join(parts[:i + 1])] for i in range(len(parts))))
        gs.add_gate(fk.gates.PolygonGate(export_name[spec.key], dims, raw.tolist()),
                    gate_path=parent_path)
    path = Path(path)
    with open(path, "wb") as fh:
        fk.export_gatingml(gs, fh)
    return path
