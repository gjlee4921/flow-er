"""Gate an unseen FCS file with a lab's gate tree: autogate(fcs, tree, lab) -> GatingResult.

Per gate, in tree order (a cascade): the parent's cells -> their 101 x 101 density in the
shared display space (from at most MAX_IMAGE_CELLS cells) + the template mask (+ the donor's
unstimulated-well density for cytokine gates) -> the routed model's probability map (or the
template itself) -> polygon (0.5 contour) -> membership of ALL parent cells inside that polygon
(display space) -> the in-gate cells are the next gates' parent. Output: docs/runtime-api.md.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from flow_er.cytometry.gating import compensate
from flow_er.cytometry.io import FcsFile
from flow_er.ml import images, registry
from flow_er.ml.gatetree import GateTree, inside, template_mask
from flow_er.ml.unet.polygon import polygon_to_raw, prob_to_polygon

MAX_IMAGE_CELLS = 200_000      # parent cells used for each density image (seed 0)
PLOT_CELLS = 20_000            # sampled parent cells returned for plotting
BAND = 2                       # pixels around the 0.5 contour that count as "near the boundary"
LOW_CONFIDENCE = 0.6           # confidence below this flags the gate for review


@dataclass
class GateOutput:
    key: str
    parent: str
    x: str
    y: str
    model: str                         # model id, or "template_copy"
    polygon_display: np.ndarray | None # (k, 2) in display units
    polygon_raw: np.ndarray | None     # (k, 2) compensated values
    count: int
    parent_count: int
    confidence: float | None           # mean |2p - 1| of parent cells near the boundary
    membership: np.ndarray             # bool over ALL events of the file
    density: np.ndarray                # (101, 101) parent counts used as the model input
    prob: np.ndarray | None = None     # (101, 101) probability map, None for template copy
    plot_cells: np.ndarray | None = None   # (<= PLOT_CELLS, 2) display coords of parent cells

    @property
    def percent_of_parent(self) -> float:
        return 100.0 * self.count / self.parent_count if self.parent_count else 0.0

    @property
    def low_confidence(self) -> bool:
        return self.confidence is not None and self.confidence < LOW_CONFIDENCE

    def to_json(self, plot: bool = False) -> dict:
        out = {"gate": self.key, "parent": self.parent, "x": self.x, "y": self.y,
               "model": self.model,
               "polygon": {"display": _list(self.polygon_display), "raw": _list(self.polygon_raw)},
               "count": self.count, "parent_count": self.parent_count,
               "percent_of_parent": round(self.percent_of_parent, 4),
               "confidence": None if self.confidence is None else round(self.confidence, 4),
               "low_confidence": self.low_confidence}
        if plot:
            out["plot"] = {"density": self.density.astype(int).tolist(),
                           "cells": _list(self.plot_cells), "bins": images.BINS,
                           "axes": "display units 0..1; density[y][x]"}
        return out


@dataclass
class GatingResult:
    fcs: str
    n_events: int
    lab: str | None
    version: str | None
    gates: dict[str, GateOutput] = field(default_factory=dict)
    channels: dict[str, float] = field(default_factory=dict)   # gated channel -> $PnR
    comp: tuple[np.ndarray, list[str]] | None = None           # (spillover, detectors) used

    def to_json(self, plot: bool = False) -> dict:
        return {"format": "gating-result-v1", "fcs": self.fcs, "n_events": self.n_events,
                "lab": self.lab, "model_version": self.version,
                "recipe_id": images.recipe_id(),
                "gates": [g.to_json(plot) for g in self.gates.values()]}


def _list(a: np.ndarray | None):
    return None if a is None else np.round(np.asarray(a, float), 6).tolist()


def read_values(fcs: FcsFile, channels: list[tuple[str, bool]],
                comp: tuple[np.ndarray, list[str]] | None = None) -> np.ndarray:
    """(n events, len(channels)) values, compensated where asked, read CHUNK rows at a time.
    `comp` = (spillover, detectors); default: the file's own spillover keyword."""
    comp = comp or fcs.spillover()
    out = np.zeros((fcs.n_events, len(channels)), np.float32)
    for rows, events in fcs.chunks():
        comped = compensate(events, fcs.channels, *comp) if comp else events
        for c, (ch, compensated) in enumerate(channels):
            out[rows, c] = (comped if compensated else events)[:, fcs.channels.index(ch)]
    return out


def boundary_confidence(prob: np.ndarray, px: np.ndarray) -> float | None:
    """Mean |2p - 1| over cells within BAND pixels of the 0.5 contour: 1 = sure, 0 = coin flip."""
    from scipy import ndimage

    above = prob > 0.5
    edge = above ^ ndimage.binary_erosion(above) | (~above ^ ndimage.binary_erosion(~above))
    band = ndimage.binary_dilation(edge, iterations=BAND)
    near = band[px[:, 1], px[:, 0]]
    if not near.any():
        return None
    return float(np.abs(2 * prob[px[near, 1], px[near, 0]] - 1).mean())


def autogate(fcs_path: str | Path, tree: GateTree, lab: str | None = None,
             comp: tuple[np.ndarray, list[str]] | None = None,
             unstim: "GatingResult | str | Path | None" = None, plot: bool = False,
             seed: int = 0, use_models: bool = True) -> GatingResult:
    """Gate every event of `fcs_path` with `tree`, routing each gate to the lab's best model.
    `unstim`: the donor's unstimulated well (a result or an FCS path), used by cytokine gates.
    use_models=False gates with the tree's own polygons only (e.g. a user's accepted gates)."""
    fcs = FcsFile(Path(fcs_path))
    channels = tree.channels()
    comp = comp or fcs.spillover()
    values = read_values(fcs, channels, comp)
    if unstim is not None and not isinstance(unstim, GatingResult):
        unstim = autogate(unstim, tree, lab, comp, None, False, seed, use_models)
    rng = np.random.default_rng(seed)
    info = registry.version_info(lab) if lab else None
    result = GatingResult(str(fcs.path), fcs.n_events, lab, info["version"] if info else None,
                          channels={c: fcs.pnr[c] for c, _ in channels}, comp=comp)
    everything = np.ones(fcs.n_events, bool)
    for spec in tree.gates:
        parent = result.gates[spec.parent].membership if spec.parent else everything
        cells = np.flatnonzero(parent)
        cols = [channels.index((spec.x, spec.compensated[0])),
                channels.index((spec.y, spec.compensated[1]))]
        chans, pnrs = [spec.x, spec.y], [fcs.pnr[spec.x], fcs.pnr[spec.y]]
        disp = images.to_display(values[cells][:, cols], chans, pnrs) if len(cells) else \
            np.zeros((0, 2))
        sub = (np.arange(len(cells)) if len(cells) <= MAX_IMAGE_CELLS
               else rng.choice(len(cells), MAX_IMAGE_CELLS, replace=False))
        density = images.density(disp[sub])
        template = spec.template_array()
        predictor = registry.route(lab, spec.key, spec.qc) if use_models else None
        if predictor is None or len(cells) == 0:
            prob, polygon, model_id = None, template if len(template) >= 3 else None, \
                registry.TEMPLATE_COPY
            member = inside(polygon, disp)
            confidence = None
        else:
            u = None
            if spec.is_cytokine and unstim is not None and spec.key in unstim.gates:
                u = unstim.gates[spec.key].density
            prob = predictor.predict(density, template_mask(template), u)
            polygon, model_id = prob_to_polygon(prob), predictor.id
            # Membership = the returned polygon, so counts, edits and the GatingML export agree
            # exactly; it differs from the probability map's pixels by < 1% of cells.
            member = inside(polygon, disp)
            confidence = boundary_confidence(prob, images.pixel_index(disp).astype(np.intp))
        membership = np.zeros(fcs.n_events, bool)
        membership[cells[member]] = True
        plot_cells = disp[rng.choice(len(disp), min(PLOT_CELLS, len(disp)), replace=False)] \
            if plot and len(disp) else None
        result.gates[spec.key] = GateOutput(
            key=spec.key, parent=spec.parent, x=spec.x, y=spec.y, model=model_id,
            polygon_display=polygon,
            polygon_raw=None if polygon is None else polygon_to_raw(polygon, chans, pnrs),
            count=int(member.sum()), parent_count=len(cells), confidence=confidence,
            membership=membership, density=density, prob=prob, plot_cells=plot_cells)
    return result


def apply_polygon(result: GatingResult, tree: GateTree, gate_key: str,
                  polygon_display: np.ndarray, fcs_path: str | Path | None = None,
                  comp: tuple[np.ndarray, list[str]] | None = None) -> GatingResult:
    """A user's edit: replace one gate's polygon and recompute it and every gate below it with
    their current polygons (point-in-polygon in display space)."""
    fcs = FcsFile(Path(fcs_path or result.fcs))
    channels = tree.channels()
    values = read_values(fcs, channels, comp)
    polygons = {k: g.polygon_display for k, g in result.gates.items()}
    polygons[gate_key] = np.asarray(polygon_display, float)
    below = [g for g in tree.gates if g.key == gate_key or g.key.startswith(gate_key + "/")]
    for spec in below:
        old = result.gates[spec.key]
        parent = result.gates[spec.parent].membership if spec.parent else np.ones(
            fcs.n_events, bool)
        cells = np.flatnonzero(parent)
        cols = [channels.index((spec.x, spec.compensated[0])),
                channels.index((spec.y, spec.compensated[1]))]
        chans, pnrs = [spec.x, spec.y], [fcs.pnr[spec.x], fcs.pnr[spec.y]]
        disp = images.to_display(values[cells][:, cols], chans, pnrs)
        member = inside(polygons[spec.key], disp)
        membership = np.zeros(fcs.n_events, bool)
        membership[cells[member]] = True
        edited = spec.key == gate_key
        result.gates[spec.key] = GateOutput(
            key=spec.key, parent=spec.parent, x=spec.x, y=spec.y,
            model="user" if edited else old.model, polygon_display=polygons[spec.key],
            polygon_raw=polygon_to_raw(polygons[spec.key], chans, pnrs)
            if polygons[spec.key] is not None else None,
            count=int(member.sum()), parent_count=len(cells),
            confidence=None if edited else old.confidence, membership=membership,
            density=images.density(disp), prob=None if edited else old.prob,
            plot_cells=old.plot_cells)
    return result
