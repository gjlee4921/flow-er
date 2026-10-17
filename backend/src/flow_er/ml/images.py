"""One recipe for 2D plot -> 101x101 density image and gate -> mask, used by BOTH training and inference.

Every channel goes through one shared display transform, whatever the workspace used: linear
x / $PnR for scatter, logicle (T = $PnR, W = 0.5, M = 4.5, A = 0) for everything else. Both map
the data to about [0, 1]; a value v falls in pixel round(100 * v), clipped to the edge pixels
(UNITO's 0-100 integer grid). Masks test each pixel centre against the expert gate in the
workspace's own display space, with FlowKit's gate code, so masks and labels agree.
"""

import hashlib
import json

import numpy as np
import pandas as pd
from flowkit import transforms

RECIPE = {
    "version": 1,
    "bins": 101,
    "scatter": {"transform": "linear", "t": "$PnR", "a": 0.0},
    "fluorescence": {"transform": "logicle", "t": "$PnR", "w": 0.5, "m": 4.5, "a": 0.0},
    "pixel": "round(100 * v), clipped to [0, 100]; v is the display value",
    "image": "event counts of the gate's parent population, rows = y channel, cols = x channel",
}
BINS = RECIPE["bins"]


def recipe_id() -> str:
    return hashlib.sha256(json.dumps(RECIPE, sort_keys=True).encode()).hexdigest()[:8]


def is_scatter(channel: str) -> bool:
    return channel.upper().startswith(("FSC", "SSC"))


def display_transform(channel: str, pnr: float):
    if is_scatter(channel):
        return transforms.LinearTransform(param_t=float(pnr), param_a=RECIPE["scatter"]["a"])
    f = RECIPE["fluorescence"]
    return transforms.LogicleTransform(param_t=float(pnr), param_w=f["w"], param_m=f["m"],
                                       param_a=f["a"])


def to_display(raw: np.ndarray, channels: list[str], pnrs: list[float]) -> np.ndarray:
    """(n, k) compensated values -> (n, k) display values in about [0, 1]."""
    return np.column_stack([display_transform(c, r).apply(raw[:, [i]]).ravel()
                            for i, (c, r) in enumerate(zip(channels, pnrs))])


def from_display(disp: np.ndarray, channels: list[str], pnrs: list[float]) -> np.ndarray:
    return np.column_stack([display_transform(c, r).inverse(disp[:, [i]]).ravel()
                            for i, (c, r) in enumerate(zip(channels, pnrs))])


def pixel_index(disp: np.ndarray) -> np.ndarray:
    """(n, 2) display values -> (n, 2) integer pixel (x, y) in [0, BINS - 1]."""
    return np.clip(np.rint(disp * (BINS - 1)), 0, BINS - 1).astype(np.int16)


def density(disp: np.ndarray) -> np.ndarray:
    """(n, 2) display values (x, y) -> (BINS, BINS) counts, image[y, x]."""
    px = pixel_index(disp)
    img = np.zeros((BINS, BINS), np.float32)
    np.add.at(img, (px[:, 1], px[:, 0]), 1)
    return img


def pixel_centres() -> np.ndarray:
    """(BINS * BINS, 2) display coordinates (x, y) of every pixel centre, row-major as image[y, x]."""
    g = np.arange(BINS) / (BINS - 1)
    yy, xx = np.meshgrid(g, g, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()])


def gate_contains(gate, xforms: dict, raw: np.ndarray) -> np.ndarray:
    """FlowKit's own membership test for (n, 2) compensated values on the gate's two dimensions,
    in the workspace display space the gate was drawn in."""
    cols = {}
    for i, dim in enumerate(gate.dimensions[:2]):
        x = raw[:, [i]]
        if dim.transformation_ref is not None:
            x = xforms[dim.transformation_ref].apply(x)
        cols[dim.id] = np.asarray(x).ravel()
    return np.asarray(gate.apply(pd.DataFrame(cols)), dtype=bool)


def gate_mask(gate, xforms: dict, channels: list[str], pnrs: list[float]) -> np.ndarray:
    """(BINS, BINS) uint8 mask: 1 where the pixel centre is inside the gate."""
    raw = from_display(pixel_centres(), channels, pnrs)
    return gate_contains(gate, xforms, raw).reshape(BINS, BINS).astype(np.uint8)


def cells_from_mask(mask: np.ndarray, disp: np.ndarray) -> np.ndarray:
    """Per-cell membership: each cell takes the value of the pixel it falls in."""
    px = pixel_index(disp)
    return mask[px[:, 1], px[:, 0]].astype(bool)
