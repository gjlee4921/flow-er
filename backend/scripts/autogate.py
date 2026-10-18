"""Autogate one FCS file with a lab's template and write the gates (docs/runtime-api.md).

    # demo: an unseen ALS screen-4 test donor with donor 2008's template, its unstimulated well
    # for the cytokine gates, and the expert's own gates drawn on top
    uv run python scripts/autogate.py "<data>/als_ics_screen4/HC Specimen_001_C5_C05 Peptide.fcs" \\
        --template-wsp "<data>/als_ics_screen3/Screen #3 HC ICS .wsp" \\
        --template-sample Specimen_001_C1_C01.fcs \\
        --template-fcs "<data>/als_ics_screen3/HC Specimen_001_C1_C01 Negative.fcs" \\
        --unstim "<data>/als_ics_screen4/HC Specimen_001_C4_C04 Negative.fcs" --lab als \\
        --expert-wsp "<data>/als_ics_screen4/Screen #4 HC ICS.wsp" \\
        --expert-sample Specimen_001_C5_C05.fcs --out result.json --gatingml gates.xml \\
        --overlay overlay.png
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flow_er.cytometry.export import to_gatingml
from flow_er.cytometry.gating import workspace_labels
from flow_er.cytometry.io import FcsFile
from flow_er.ml.gatetree import GateTree, from_workspace
from flow_er.ml.runtime import autogate


def label(key: str, duplicates: set[str]) -> str:
    parts = key.split("/")
    return parts[-1] + (f" [{parts[-2]}]" if parts[-1] in duplicates else "")


def overlay(result, tree: GateTree, expert: GateTree | None, out: Path) -> None:
    gates = list(result.gates.values())
    cols = 4
    rows = -(-len(gates) // cols)
    names = [g.key.split("/")[-1] for g in gates]
    dup = {n for n in names if names.count(n) > 1}
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    for ax, g in zip(axes.ravel(), gates):
        ax.imshow(np.log1p(g.density), origin="lower", cmap="Greys", extent=(0, 1, 0, 1))
        polys = [(tree[g.key].template_array(), "tab:blue", "--", "template")]
        if expert is not None and g.key in expert.keys:
            polys.append((expert[g.key].template_array(), "tab:green", "-", "expert"))
        if g.polygon_display is not None:
            polys.append((np.asarray(g.polygon_display), "tab:red", "-", g.model.split("/")[-2]
                          if "/unito/" in g.model else g.model.split("/")[-1]))
        for xy, color, style, name in polys:
            if len(xy) >= 3:
                closed = np.vstack([xy, xy[:1]])
                ax.plot(closed[:, 0], closed[:, 1], style, color=color, lw=1.2, label=name)
        flag = " (check)" if g.low_confidence else ""
        ax.set_title(f"{label(g.key, dup)[:30]}\n{g.percent_of_parent:.2f}% of parent{flag}",
                     fontsize=7)
        ax.set_xlabel(g.x, fontsize=6)
        ax.set_ylabel(g.y, fontsize=6)
        ax.tick_params(labelsize=5)
        ax.legend(fontsize=5, loc="upper right")
    for ax in axes.ravel()[len(gates):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fcs", type=Path)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--template-json", type=Path, help="a saved gate tree (GateTree JSON)")
    src.add_argument("--template-wsp", type=Path, help="FlowJo workspace holding the template")
    p.add_argument("--template-sample", help="sample in --template-wsp whose gates are used")
    p.add_argument("--template-fcs", type=Path, help="that sample's FCS (for $PnR); optional")
    p.add_argument("--gates", nargs="*", help="only these gate keys (and their ancestors)")
    p.add_argument("--lab", help="lab whose models answer (default: base model, then template)")
    p.add_argument("--unstim", type=Path, help="the donor's unstimulated well, for cytokine gates")
    p.add_argument("--out", type=Path, help="result JSON")
    p.add_argument("--plot", action="store_true", help="include plot data in the JSON")
    p.add_argument("--gatingml", type=Path, help="write the gates as GatingML 2.0")
    p.add_argument("--save-template", type=Path, help="write the template as GateTree JSON")
    p.add_argument("--overlay", type=Path, help="PNG of every gate: auto, template, expert")
    p.add_argument("--expert-wsp", type=Path, help="workspace with the expert's gates for --fcs")
    p.add_argument("--expert-sample", help="the sample id of --fcs in --expert-wsp")
    a = p.parse_args()

    if a.template_json:
        tree = GateTree.from_json(a.template_json)
    else:
        tfcs = FcsFile(a.template_fcs) if a.template_fcs else FcsFile(a.fcs)
        tree = from_workspace(a.template_wsp, a.template_sample, tfcs, a.gates)
    if a.save_template:
        a.save_template.write_text(json.dumps(tree.to_json(), indent=1))

    t0 = time.time()
    result = autogate(a.fcs, tree, a.lab, unstim=a.unstim, plot=a.plot)
    print(f"{a.fcs.name}: {result.n_events} events, {len(result.gates)} gates in "
          f"{time.time() - t0:.1f}s (lab {a.lab}, version {result.version})")
    if a.out:
        a.out.write_text(json.dumps(result.to_json(plot=a.plot)))
    if a.gatingml:
        to_gatingml(result, tree, a.gatingml)

    expert, truth = None, None
    if a.expert_wsp:
        fcs = FcsFile(a.fcs)
        expert = from_workspace(a.expert_wsp, a.expert_sample, fcs, tree.keys)
        labels, _, _ = workspace_labels(fcs, a.expert_wsp, a.expert_sample, tree.keys, [])
        truth = dict(zip(tree.keys, labels))
    names = [k.split("/")[-1] for k in tree.keys]
    dup = {n for n in names if names.count(n) > 1}
    print(f"{'gate':36s} {'model':32s} {'auto %':>8s}" + (f" {'expert %':>9s} {'F1':>6s}"
                                                          if truth else "") + "  confidence")
    for k, g in result.gates.items():
        line = f"{label(k, dup)[:36]:36s} {g.model[-32:]:32s} {g.percent_of_parent:8.2f}"
        if truth:
            e = truth[k]
            parent = truth[tree[k].parent] if tree[k].parent else np.ones_like(e)
            f1 = 2 * (g.membership & e).sum() / max(g.membership.sum() + e.sum(), 1)
            line += f" {100 * e[parent].mean():9.2f} {f1:6.3f}"
        conf = "-" if g.confidence is None else f"{g.confidence:.2f}"
        print(line + f"  {conf}{'  check' if g.low_confidence else ''}")
    if a.overlay:
        overlay(result, tree, expert, a.overlay)
        print(f"overlay: {a.overlay}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
