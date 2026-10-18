"""Who each file came from: the individual (person or animal), the condition, and that individual's
unstimulated well. The gate models need it: cytokine gates read the unstimulated well, and when the
lab's models are retrained, held-out samples must come from individuals not trained on.

Read from file names until the scientist keeps it. Plates are usually laid out as runs of wells,
one individual after another, each run starting with the unstimulated well ("HC Specimen_001_C4_C04
Negative", "... C5_C05 Peptide", "... C6_C06 Positive"); a group prefix (ALS, HC) keeps groups apart.
Kept rows live in state/sample_meta.json and are changed like anything else: as a draft, then Keep.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from flow_er.workspace import lab, store

UNSTIM = {"negative", "neg", "unstim", "unstimulated", "us", "nostim", "media", "medium", "dmso", "control"}
STIM = {"peptide", "positive", "pos", "stim", "stimulated", "pma", "sea", "seb", "cmv", "cefx", "megapool", "mp"}
WELL = re.compile(r"(?:^|[_\s-])([A-H])0*(\d{1,2})(?=[_\s.-]|$)", re.IGNORECASE)


def _parts(sid: str) -> dict:
    stem = Path(sid).stem
    words = re.split(r"[\s_]+", stem)
    last = words[-1].lower() if words else ""
    cond = words[-1] if last in UNSTIM | STIM else ""
    well = WELL.findall(stem)
    row, col = (well[-1][0].upper(), int(well[-1][1])) if well else ("", 0)
    group = words[0] if len(words) > 2 and not WELL.search(words[0]) and words[0].lower() not in UNSTIM | STIM else ""
    return {"stem": stem, "condition": cond, "unstim": last in UNSTIM, "row": row, "col": col, "group": group}


def infer(sample_ids: list[str]) -> list[dict]:
    """One row per file: individual, condition and the unstimulated well's file, read from names."""
    parts = {sid: _parts(sid) for sid in sample_ids}
    rows = {sid: {"sample": sid, "individual": "", "condition": p["condition"], "unstim": None} for sid, p in parts.items()}
    with_cond = [s for s, p in parts.items() if p["condition"]]
    if with_cond:
        # per group, wells in plate order; each unstimulated well starts the next individual
        for group in sorted({parts[s]["group"] for s in with_cond}):
            members = sorted((s for s in with_cond if parts[s]["group"] == group), key=lambda s: (parts[s]["row"], parts[s]["col"], s))
            n, current, unstim = 0, None, None
            for s in members:
                if parts[s]["unstim"] or current is None:
                    n += 1
                    current = f"{group}-{n}" if group else f"donor-{n}"
                    unstim = s if parts[s]["unstim"] else None
                rows[s]["individual"], rows[s]["unstim"] = current, unstim
    rest = [s for s in sample_ids if not rows[s]["individual"]]
    if rest:
        # no conditions in the names: files sharing a name prefix are taken as one individual
        stems = [parts[s]["stem"] for s in rest]
        common = os.path.commonprefix(stems)
        if any(st[len(common):len(common) + 1] not in ("", "_", "-", " ") for st in stems):
            common = re.sub(r"[^_\s-]*$", "", common)  # don't cut a name part in half
        common = common.rstrip("_- ")
        for s in rest:
            rows[s]["individual"] = common if len(rest) > 1 and len(common) >= 4 else parts[s]["stem"]
    return [rows[s] for s in sample_ids]


def get(exp_id: str) -> list[dict]:
    """The rows in use: the kept ones, with any file they miss read from its name."""
    sids = lab.dataset(exp_id).sample_ids
    kept = {r["sample"]: r for r in store.state(exp_id, "sample_meta")}
    guessed = {r["sample"]: r for r in infer(sids)}
    return [{**(kept.get(s) or guessed[s]), "kept": s in kept} for s in sids]


def of(exp_id: str, sid: str) -> dict:
    return next((r for r in get(exp_id) if r["sample"] == sid), {"sample": sid, "individual": sid, "condition": "", "unstim": None})


def draft(exp_id: str, rows: list[dict] | None = None) -> dict:
    """Stage the rows (or those read from file names) as a draft for the scientist to keep."""
    sids = set(lab.dataset(exp_id).sample_ids)
    rows = rows if rows is not None else infer(sorted(sids))
    clean = []
    for r in rows:
        if r.get("sample") not in sids:
            raise ValueError(f"no file {r.get('sample')} in this experiment")
        if r.get("unstim") and r["unstim"] not in sids:
            raise ValueError(f"no file {r['unstim']} to be the unstimulated well")
        clean.append({"sample": r["sample"], "individual": str(r.get("individual") or r["sample"]),
                      "condition": str(r.get("condition") or ""), "unstim": r.get("unstim") or None})
    n = len({r["individual"] for r in clean})
    return store.set_draft(exp_id, "sample_meta", clean, f"Who each file came from: {len(clean)} files, {n} individuals")
