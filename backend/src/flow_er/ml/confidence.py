"""Calibrating the low-confidence flag (PRD success metric: the flags catch the gates that are wrong).

Run the full pipeline on held-out validation donors (never the test donors), compare every model
gate with the expert, and pick the confidence threshold that best separates gates with a large
error from the rest. The threshold is stored in the lab's version.json, and autogate uses it from
then on instead of the default (runtime.LOW_CONFIDENCE).
"""
import json

import numpy as np

from flow_er.ml import registry
from flow_er.ml.runtime import LOW_CONFIDENCE

LARGE_ERROR_PP = 1.0     # a gate whose frequency is off by more than this is "wrong"


def choose_threshold(rows: list[dict], large_pp: float = LARGE_ERROR_PP) -> dict:
    """`rows`: one per model gate and sample, {"confidence": float, "error_pp": float}. Returns the
    threshold that maximises flagged-wrong share minus flagged-right share (Youden's J), the
    lower one on ties so fewer gates are flagged, and how many wrong gates it catches."""
    rows = [r for r in rows if r.get("confidence") is not None]
    if not rows:
        raise ValueError("no model gates to calibrate on")
    conf = np.array([r["confidence"] for r in rows], float)
    wrong = np.array([r["error_pp"] > large_pp for r in rows])
    candidates = np.unique(np.concatenate([conf, [0.0, 1.0001]]))
    best = None
    for t in candidates:
        flag = conf < t
        caught = flag[wrong].mean() if wrong.any() else 0.0
        false = flag[~wrong].mean() if (~wrong).any() else 0.0
        j = caught - false
        if best is None or j > best[0] + 1e-12:
            best = (j, float(t))
    t = best[1]
    flag = conf < t
    # the trade-off at a few thresholds: wrong gates caught against right gates sent for review
    curve = [{"threshold": round(float(c), 4), "wrong_flagged": int(((conf < c) & wrong).sum()),
              "right_flagged": int(((conf < c) & ~wrong).sum())}
             for c in sorted({0.6, 0.7, 0.8, 0.85, 0.9, round(t, 4)})]
    return {"threshold": round(t, 4), "large_error_pp": large_pp, "gates": len(rows),
            "wrong": int(wrong.sum()), "wrong_flagged": int((flag & wrong).sum()),
            "right_flagged": int((flag & ~wrong).sum()), "flagged": int(flag.sum()),
            "default_threshold": LOW_CONFIDENCE,
            "default_wrong_flagged": int(((conf < LOW_CONFIDENCE) & wrong).sum()), "curve": curve}


def store(lab: str, stats: dict, version: str | None = None, donors: list[str] | None = None) -> str:
    """Write the chosen threshold into the version (current by default); autogate reads it."""
    version = version or registry.current_version(lab)
    if version is None:
        raise ValueError(f"lab {lab!r} has no version to store a threshold in")
    path = registry.lab_dir(lab) / version / "version.json"
    info = json.loads(path.read_text())
    info["confidence_threshold"] = stats["threshold"]
    info.setdefault("metrics", {})["confidence"] = {**stats, "calibrated_on": donors or []}
    path.write_text(json.dumps(info, indent=1))
    return version


def threshold(lab: str | None) -> float:
    """The threshold autogate uses for this lab: the current version's calibrated one, else the default."""
    info = registry.version_info(lab) if lab else None
    return float((info or {}).get("confidence_threshold") or LOW_CONFIDENCE)
