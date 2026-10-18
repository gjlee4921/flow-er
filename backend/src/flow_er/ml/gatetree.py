"""A lab's gate tree (its template): gates in tree order, channels, template polygons.

Templates are polygons in the shared display space of ml/images.py, whatever their source:
- a FlowJo workspace (from_workspace): each FlowKit gate is rasterised exactly (images.gate_mask)
  and traced back to a polygon, so rectangles, biex-curved edges and open ranges all work;
- a saved template or a previous experiment's accepted gates (from_json / with_polygons).
"""

import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import flowkit as fk
import numpy as np
from matplotlib.path import Path as MplPath
from skimage.draw import polygon as fill_polygon

from flow_er.cytometry.gating import name_and_path
from flow_er.cytometry.io import FcsFile
from flow_er.ml import images
from flow_er.ml.unet.polygon import prob_to_polygon

# Gates whose name looks like a cytokine readout get the donor's unstimulated well as an input.
CYTOKINE = re.compile(r"^(IFN|IL-|TNF)")


@dataclass
class GateSpec:
    key: str                          # path below root joined by "/", e.g. "Lymphocytes/CD3"
    x: str                            # channel names (FlowJo-style $PnN)
    y: str
    compensated: tuple[bool, bool]    # does the gate use compensated values on x / y?
    template: list[list[float]]       # polygon vertices (x, y) in display units
    is_cytokine: bool = False
    qc: bool = False                  # acquisition QC gate (Time): template only, never modelled

    @property
    def name(self) -> str:
        return self.key.split("/")[-1]

    @property
    def parent(self) -> str:
        return "/".join(self.key.split("/")[:-1])

    def template_array(self) -> np.ndarray:
        return np.asarray(self.template, dtype=float).reshape(-1, 2)


@dataclass
class GateTree:
    gates: list[GateSpec]             # parents before children
    source: dict = field(default_factory=dict)

    def __post_init__(self):
        keys = [g.key for g in self.gates]
        for g in self.gates:
            if g.parent and g.parent not in keys[:keys.index(g.key)]:
                raise ValueError(f"{g.key}: parent {g.parent!r} must come before it")

    def __getitem__(self, key: str) -> GateSpec:
        return next(g for g in self.gates if g.key == key)

    @property
    def keys(self) -> list[str]:
        return [g.key for g in self.gates]

    def channels(self) -> list[tuple[str, bool]]:
        """(channel, compensated) pairs the tree needs, in first-use order."""
        out = []
        for g in self.gates:
            for c in zip((g.x, g.y), g.compensated):
                if c not in out:
                    out.append(c)
        return out

    def with_polygons(self, polygons: dict[str, np.ndarray], source: dict | None = None
                      ) -> "GateTree":
        """The same tree with some templates replaced (e.g. a user's accepted gates)."""
        gates = [replace(g, template=np.asarray(polygons[g.key]).tolist()) if g.key in polygons
                 else g for g in self.gates]
        return GateTree(gates, source or self.source)

    def to_json(self) -> dict:
        return {"format": "gate-tree-v1", "recipe_id": images.recipe_id(), "source": self.source,
                "gates": [asdict(g) for g in self.gates]}

    @classmethod
    def from_json(cls, data: dict | str | Path) -> "GateTree":
        if isinstance(data, (str, Path)):
            data = json.loads(Path(data).read_text())
        gates = [GateSpec(**{**g, "compensated": tuple(g["compensated"])}) for g in data["gates"]]
        return cls(gates, data.get("source", {}))


def template_mask(polygon: np.ndarray) -> np.ndarray:
    """(101, 101) uint8: 1 where the pixel centre is inside the display-space polygon."""
    mask = np.zeros((images.BINS, images.BINS), np.uint8)
    if len(polygon) >= 3:
        rr, cc = fill_polygon(polygon[:, 1] * (images.BINS - 1), polygon[:, 0] * (images.BINS - 1),
                              mask.shape)
        mask[rr, cc] = 1
    return mask


def inside(polygon: np.ndarray, disp: np.ndarray) -> np.ndarray:
    """Which (n, 2) display points lie inside the polygon."""
    if polygon is None or len(polygon) < 3 or len(disp) == 0:
        return np.zeros(len(disp), bool)
    return MplPath(polygon).contains_points(disp)


def from_workspace(wsp_path: Path, sample_id: str, fcs: FcsFile | None = None,
                   gate_keys: list[str] | None = None) -> GateTree:
    """The gate tree of one workspace sample. `fcs` (that sample's file, or any file of the same
    panel) supplies $PnR for the display transform; 262144 is assumed without it."""
    gs = fk.Workspace(str(wsp_path), load_missing_file_data=True).get_gating_strategy(sample_id)
    ids = gs.get_gate_ids()
    keys = gate_keys or ["/".join(path[1:] + (name,)) for name, path in ids]
    keys = sorted(keys, key=lambda k: k.count("/"))
    gates = []
    for key in keys:
        gate = gs.get_gate(*name_and_path(key))
        dims = gate.dimensions[:2]
        chans = [d.id for d in dims]
        pnrs = [fcs.pnr.get(c, 262144.0) if fcs else 262144.0 for c in chans]
        mask = images.gate_mask(gate, gs.transformations, chans, pnrs)
        poly = prob_to_polygon(mask.astype(float), tolerance=0.5)
        gates.append(GateSpec(
            key=key, x=chans[0], y=chans[1],
            compensated=tuple(d.compensation_ref not in (None, "uncompensated") for d in dims),
            template=[] if poly is None else poly.tolist(),
            is_cytokine=bool(CYTOKINE.match(key.split("/")[-1])),
            qc="Time" in chans))
    return GateTree(gates, {"kind": "workspace", "path": str(wsp_path), "sample": sample_id})
