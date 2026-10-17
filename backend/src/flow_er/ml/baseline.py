"""Classical gating: template copy and population-following template transfer (spec §3 rows 1-2).

Both start from a reference sample's expert gate (the template) and test each target cell with
FlowKit against that gate in the reference workspace's display space ("moved through raw space").
- template_copy: the template unchanged.
- classical: the template moved in the shared display space (ml/images.py).
  - scatter, live and lineage gates: translated by the shift of the 2D density peak inside a
    window around the template (the prototype's peak2d); fewer than FEW parent cells keeps it.
  - cytokine gates: shifted along the cytokine (x) axis until the donor's unstimulated well shows
    the background rate fitted on the training donors; the same shift is used for all the
    donor's wells (the expert sets the threshold from the unstimulated well).
"""

import numpy as np
from scipy import ndimage

from flow_er.ml import images

FEW = 120                 # parent cells below which a gate keeps the template position
WINDOW_GROW = 0.5         # peak window = template bounding box grown by this share per side
SMOOTH_PASSES = 2         # 3 x 3 box smoothing passes before taking the peak
SHIFT_RANGE = 0.4         # largest cytokine shift searched, in display units (1 = full axis)


def apply_template(gate, xforms: dict, raw: np.ndarray, channels: list[str], pnrs: list[float],
                   shift: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Membership of (n, 2) compensated values in the template gate after moving the gate by
    `shift` display units (equivalently, moving the cells by -shift)."""
    if shift != (0.0, 0.0):
        disp = images.to_display(raw, channels, pnrs) - np.asarray(shift)
        raw = images.from_display(disp, channels, pnrs)
    return images.gate_contains(gate, xforms, raw)


def _smooth(img: np.ndarray) -> np.ndarray:
    for _ in range(SMOOTH_PASSES):
        img = ndimage.uniform_filter(img, size=3, mode="constant")
    return img


def peak_window(template_mask: np.ndarray) -> tuple[slice, slice]:
    ys, xs = np.nonzero(template_mask)
    if len(xs) == 0:
        return slice(0, images.BINS), slice(0, images.BINS)

    def grow(lo, hi):
        pad = round(WINDOW_GROW * (hi - lo + 1))
        return slice(max(0, lo - pad), min(images.BINS, hi + pad + 1))

    return grow(ys.min(), ys.max()), grow(xs.min(), xs.max())


def density_peak(density: np.ndarray, window: tuple[slice, slice]) -> np.ndarray:
    """(x, y) pixel of the smoothed density maximum inside the window."""
    sub = _smooth(density.astype(np.float64))[window]
    y, x = np.unravel_index(np.argmax(sub), sub.shape)
    return np.array([x + window[1].start, y + window[0].start], float)


def peak_shift(target_density: np.ndarray, reference_density: np.ndarray,
               template_mask: np.ndarray, n_parent: int) -> tuple[float, float]:
    """Shift (display units) that moves the reference population's peak onto the target's."""
    if n_parent < FEW:
        return 0.0, 0.0
    window = peak_window(template_mask)
    dx, dy = (density_peak(target_density, window) - density_peak(reference_density, window))
    return dx / (images.BINS - 1), dy / (images.BINS - 1)


def threshold_shift(gate, xforms: dict, unstim_raw: np.ndarray, channels: list[str],
                    pnrs: list[float], background: float, iterations: int = 40) -> float:
    """x shift of the template at which the unstimulated well's positive fraction equals
    `background`. Moving a positive gate right only lowers the fraction, so bisect."""
    if len(unstim_raw) < FEW:
        return 0.0
    disp = images.to_display(unstim_raw, channels, pnrs)

    def fraction(s: float) -> float:
        raw = images.from_display(disp - np.array([s, 0.0]), channels, pnrs)
        return float(images.gate_contains(gate, xforms, raw).mean())

    lo, hi = -SHIFT_RANGE, SHIFT_RANGE
    if fraction(lo) <= background:
        return lo
    if fraction(hi) >= background:
        return hi
    for _ in range(iterations):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fraction(mid) > background else (lo, mid)
    return (lo + hi) / 2
