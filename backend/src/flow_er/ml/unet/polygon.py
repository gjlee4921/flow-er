"""Probability map -> editable polygon gate: the largest closed 0.5 contour.

The map is padded with zeros so contours touching the plot edge still close. Vertices come out in
display coordinates (pixel / 100, x then y) and can be taken back to compensated values with
images.from_display. A gate that reaches the plot edge is extended past it (extend_to_edges): the
edge pixels hold every cell beyond the axis, so the polygon has to contain those cells too.
"""

import numpy as np
from skimage.measure import approximate_polygon, find_contours

from flow_er.ml import images

# Where a vertex on or past the plot border goes (display units): far enough to hold every event the
# edge pixels stand for (logicle -0.5 is about -15,000 at T = 262144; linear -0.5 is -PnR / 2).
EDGE_OUT = (-0.5, 1.5)


def extend_to_edges(xy: np.ndarray | None) -> np.ndarray | None:
    """Vertices on or past the plot border (<= 0 or >= 1) moved out to EDGE_OUT."""
    if xy is None:
        return None
    xy = np.asarray(xy, float).copy()
    xy[xy <= 0] = EDGE_OUT[0]
    xy[xy >= 1] = EDGE_OUT[1]
    return xy


def polygon_area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def prob_to_polygon(prob: np.ndarray, level: float = 0.5,
                    tolerance: float = 0.5) -> np.ndarray | None:
    """(k, 2) display (x, y) vertices of the largest closed `level` contour, or None if the map
    never reaches it. `tolerance` (pixels) simplifies the outline; 0 keeps every vertex."""
    contours = find_contours(np.pad(prob.astype(float), 1), level)
    closed = [c for c in contours if len(c) >= 4 and np.allclose(c[0], c[-1])]
    if not closed:
        return None
    rc = max(closed, key=polygon_area) - 1
    if tolerance > 0:
        rc = approximate_polygon(rc, tolerance)
    return extend_to_edges(rc[:-1, ::-1] / (images.BINS - 1))


def polygon_to_raw(xy: np.ndarray, channels: list[str], pnrs: list[float]) -> np.ndarray:
    """Display-space vertices -> compensated values of the gate's two channels."""
    return images.from_display(xy, channels, pnrs)
