"""Calibrate the low-confidence flag on held-out validation donors and store it in the lab's version.

Runs the full pipeline (autogate with the lab's current models) on each validation well, compares
every gate with the expert's FlowJo gates, and picks the confidence threshold that best separates
gates with large frequency errors (ml.confidence.choose_threshold). Use the validation donors only,
never the test donors.

    uv run python scripts/calibrate_confidence.py --lab als --wells val_wells.json [--write]

val_wells.json lists one entry per well:
    [{"fcs": ".../HC Specimen_001_C2_C02 Peptide.fcs", "unstim": ".../HC Specimen_001_C1_C01 Negative.fcs",
      "expert_wsp": ".../Screen #3 HC ICS.wsp", "expert_sample": "Specimen_001_C2_C02.fcs",
      "individual": "human:4191"}, ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from flow_er.cytometry.gating import workspace_labels
from flow_er.cytometry.io import FcsFile
from flow_er.ml import confidence, registry
from flow_er.ml.gatetree import GateTree
from flow_er.ml.runtime import autogate


def rows_for(well: dict, tree: GateTree, lab: str) -> list[dict]:
    """One row per gate of one well: model, confidence and the frequency error against the expert."""
    result = autogate(well["fcs"], tree, lab, unstim=well.get("unstim"))
    labels, _, _ = workspace_labels(FcsFile(Path(well["fcs"])), Path(well["expert_wsp"]),
                                    well["expert_sample"], tree.keys, [])
    truth = dict(zip(tree.keys, labels))
    out = []
    for k, g in result.gates.items():
        parent = truth[tree[k].parent] if tree[k].parent else np.ones_like(truth[k])
        expert_pct = 100 * truth[k][parent].mean() if parent.any() else 0.0
        out.append({"well": Path(well["fcs"]).name, "individual": well.get("individual", ""),
                    "gate": k, "model": g.model, "confidence": g.confidence,
                    "auto_pct": g.percent_of_parent, "expert_pct": expert_pct,
                    "error_pp": abs(g.percent_of_parent - expert_pct)})
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--lab", required=True)
    p.add_argument("--wells", type=Path, required=True, help="JSON list of validation wells")
    p.add_argument("--template-json", type=Path, help="the lab template (default: the lab's template.json)")
    p.add_argument("--large-pp", type=float, default=confidence.LARGE_ERROR_PP)
    p.add_argument("--write", action="store_true", help="store the threshold in the current version")
    a = p.parse_args()
    tree = GateTree.from_json(json.loads((a.template_json or registry.lab_dir(a.lab) / "template.json").read_text()))
    wells = json.loads(a.wells.read_text())
    rows = [r for w in wells for r in rows_for(w, tree, a.lab)]
    stats = confidence.choose_threshold(rows, a.large_pp)
    model_rows = [r for r in rows if r["confidence"] is not None]
    print(f"{len(wells)} wells, {len(model_rows)} model gates; wrong = error > {a.large_pp} pp")
    for r in sorted(model_rows, key=lambda r: r["confidence"]):
        flag = "flag" if r["confidence"] < stats["threshold"] else ""
        print(f"  {r['confidence']:.3f}  {r['error_pp']:6.2f} pp  {flag:4s}  {r['well'][:28]:28s} {r['gate'].split('/')[-1]}")
    print(f"threshold {stats['threshold']}: catches {stats['wrong_flagged']} of {stats['wrong']} wrong gates, "
          f"flags {stats['right_flagged']} right ones ({stats['flagged']} of {stats['gates']} flagged); "
          f"the default {stats['default_threshold']} caught {stats['default_wrong_flagged']}")
    print("trade-off:  threshold  wrong caught  right flagged")
    for c in stats["curve"]:
        print(f"            {c['threshold']:.3f}     {c['wrong_flagged']:>2} of {stats['wrong']}       {c['right_flagged']:>2} of {stats['gates'] - stats['wrong']}")
    if a.write:
        v = confidence.store(a.lab, stats, donors=sorted({w.get("individual", "") for w in wells}))
        print(f"stored in {a.lab}/{v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
