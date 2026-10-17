"""Score every method's predictions on one role of an evaluation: CSV, markdown and two plots.

For each sample of the role (frozen split) and each gate with a cache file, compares each method's
`pred` with FlowKit's `cell_label` (formats: docs/cache-format.md; metrics: flow_er.ml.evaluate).
Writes results_<role>.csv (one row per method x well x gate), report_<role>.md, f1_<role>.png
and frequency_<role>.png into data/_results/<evaluation>/ (or --out).

    uv run python scripts/report.py --evaluation als --role test
    uv run python scripts/report.py --evaluation als --role test --methods template_copy unito
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flow_er.config import settings
from flow_er.ml import splits
from flow_er.ml.cache import cache_dir, gate_id, gate_path, results_path, sample_id
from flow_er.ml.evaluate import MIN_POSITIVES, STIMULI, add_responses, summarize, well_metrics

# Categorical slots in fixed order (one per method), light chart surface.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
FREQ_FLOOR = 0.01  # % ; log axes show smaller frequencies at this value


def results_root(evaluation: str) -> Path:
    return settings.data_dir / "_results" / evaluation


def available_methods(evaluation: str) -> list[str]:
    root = results_root(evaluation)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name[0] not in "_.")


def cached_gates(sample: dict) -> list[str]:
    """Gate keys with a model file for this sample, sorted by key (parents first)."""
    root, sid = cache_dir() / sample["dataset"], sample_id(sample)
    if not root.is_dir():
        return []
    metas = (d / f"{sid}.json" for d in root.iterdir() if d.is_dir())
    return sorted(json.loads(m.read_text())["gate"] for m in metas if m.is_file())


def gate_labels(gate_keys: list[str]) -> dict[str, str]:
    """Last path element, plus the parent's when two gates share it."""
    last = Counter(k.split("/")[-1] for k in set(gate_keys))
    out = {}
    for k in gate_keys:
        parts = k.split("/")
        out[k] = parts[-1] if last[parts[-1]] == 1 or len(parts) < 2 \
            else f"{parts[-1]} [{parts[-2]}]"
    return out


def score(evaluation: str, methods: list[str], pairs: list[tuple[dict, str]]
          ) -> tuple[pd.DataFrame, Counter]:
    """Per-well metrics of every method on every (sample, gate key) pair; also the number of
    missing (or mismatched) predictions per method."""
    rows, missing = [], Counter()
    for sample, gk in pairs:
        path = gate_path(sample, gk)
        meta = json.loads(path.with_suffix(".json").read_text())
        with np.load(path, allow_pickle=False) as z:
            true = z["cell_label"]
        well = {"sample": splits.key(sample), "sample_id": sample_id(sample),
                "dataset": sample["dataset"], "individual": meta["individual"],
                "cohort": meta["cohort"], "condition": meta["condition"], "gate": gk,
                "gate_id": gate_id(gk), "is_cytokine": bool(meta["is_cytokine"]),
                "unstim_source": meta.get("unstim_source")}
        for m in methods:
            p = results_path(evaluation, m, sample, gk)
            if not p.is_file():
                missing[m] += 1
                continue
            with np.load(p, allow_pickle=False) as z:
                pred = z["pred"]
            if pred.shape != true.shape:
                print(f"  {m}: {p.name} has {pred.shape} cells, truth {true.shape}; skipped")
                missing[m] += 1
                continue
            rows.append({"method": m, **well, **well_metrics(pred, true, meta["counts"])})
    df = pd.DataFrame(rows)
    return (add_responses(df) if len(df) else df), missing


# --- report ---------------------------------------------------------------------------------------


def _fmt(x, digits: int = 2) -> str:
    return "-" if x is None or pd.isna(x) else f"{x:.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(r) + " |" for r in rows), ""]


def markdown(df: pd.DataFrame, summary: pd.DataFrame, methods: list[str], labels: dict,
             title: str, missing: Counter) -> str:
    gates = list(labels)
    cyto = [g for g in gates if df.loc[df["gate"] == g, "is_cytokine"].any()]
    name = {g: labels[g] + (" *" if g in cyto else "") for g in gates}
    s = summary.set_index(["gate", "method"])
    wells = df.groupby(["gate", "sample"], as_index=False).first()  # noise floors per well
    floor = wells.groupby("gate")["noise_floor"].mean()

    def val(g, m, col, digits=2):
        return _fmt(s[col].get((g, m)), digits)

    absent = ", ".join(f"{m} {n}" for m, n in sorted(missing.items()) if n) or "none"
    notes = (f"F1: per cell, wells with >= {MIN_POSITIVES} expert-positive cells (median). "
             "Frequency: % of the parent; errors in percentage points (pp), predicted minus "
             "expert. Noise floor: mean |FlowJo % - FlowKit %| of the same wells. "
             "`*` = cytokine gate.")
    scored = f"Scored wells: {df['sample'].nunique()} samples, {len(gates)} gates."
    lines = [f"# {title}", "", f"{scored} Missing predictions: {absent}.", "", notes, ""]
    lines += ["## Median F1", ""] + _table(
        ["Gate", *methods], [[name[g], *(val(g, m, "median_f1", 3) for m in methods)]
                             for g in gates])
    lines += ["## Mean |frequency error| per well (pp)", ""] + _table(
        ["Gate", "noise floor", *methods],
        [[name[g], _fmt(floor.get(g)), *(val(g, m, "mean_abs_freq_error") for m in methods)]
         for g in gates])
    lines += ["## Per-donor mean |frequency error| (pp, mean over donors)", ""] + _table(
        ["Gate", "noise floor", *methods],
        [[name[g], _fmt(floor.get(g)), *(val(g, m, "donor_mean_abs_freq_error") for m in methods)]
         for g in gates])
    if cyto:
        rows = []
        for g in cyto:
            for cond, st in STIMULI.items():
                rf = wells[(wells["gate"] == g) & (wells["stimulus"] == st)]
                rows.append([name[g], cond, _fmt(rf["response_noise_floor"].mean()),
                             *(val(g, m, f"mean_abs_response_error_{st}") for m in methods)])
        lines += ["## Mean |background-subtracted response error| (pp, cytokine gates)", "",
                  "Response = stimulated-well % minus the donor's unstimulated-well %.", ""]
        lines += _table(["Gate", "stimulus", "noise floor", *methods], rows)
    lines += ["## Per gate", ""]
    cols = [("wells", "n_wells", 0), ("wells with F1", "n_wells_f1", 0),
            ("median F1", "median_f1", 3), ("mean freq error", "mean_freq_error", 2),
            ("mean abs freq error", "mean_abs_freq_error", 2),
            ("donor-mean abs freq error", "donor_mean_abs_freq_error", 2),
            ("noise floor", "mean_noise_floor", 2)]
    for g in gates:
        c = cols + ([(f"abs response error {cond}", f"mean_abs_response_error_{st}", 2)
                     for cond, st in STIMULI.items()] if g in cyto else [])
        lines += [f"### {name[g]}", "", f"`{g}`", ""]
        lines += _table(["Method", *(h for h, _, _ in c)],
                        [[m, *(val(g, m, col, d) for _, col, d in c)] for m in methods
                         if (g, m) in s.index])
    return "\n".join(lines)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_f1(summary: pd.DataFrame, methods: list[str], labels: dict, out: Path) -> None:
    gates = list(labels)
    n = len(methods)
    bar = 0.8 / n
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.18 * n * len(gates) + 0.25 * len(gates)),
                           facecolor=SURFACE)
    s = summary.set_index(["gate", "method"])["median_f1"]
    y = np.arange(len(gates))
    for i, m in enumerate(methods):
        vals = [s.get((g, m), np.nan) for g in gates]
        ax.barh(y - 0.4 + bar * (i + 0.5), vals, height=bar * 0.85, color=COLORS[i % len(COLORS)],
                label=m)
    ax.set_yticks(y, [labels[g] for g in gates])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"median per-cell F1 (wells with >= {MIN_POSITIVES} expert-positive cells)",
                  color=INK, fontsize=9)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=min(n, 4), frameon=False,
              fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def plot_frequency(df: pd.DataFrame, methods: list[str], out: Path) -> None:
    n = len(methods)
    cols = min(n, 3)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.4 * rows), squeeze=False,
                             facecolor=SURFACE)
    lo, hi = FREQ_FLOOR, 100.0
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.set_visible(False)
            continue
        m = methods[i]
        d = df[df["method"] == m]
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle="--", zorder=1)
        for cyto, marker in ((False, "o"), (True, "^")):
            e = d[d["is_cytokine"] == cyto]
            ax.scatter(e["freq_true"].clip(lo, hi), e["freq_pred"].clip(lo, hi), s=14,
                       marker=marker, color=COLORS[i % len(COLORS)], alpha=0.7,
                       edgecolors=SURFACE, linewidths=0.5, zorder=2, clip_on=False,
                       label="cytokine gates" if cyto else "other gates")
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
        ax.set_title(m, fontsize=9, color=INK)
        ax.set_xlabel("expert % of parent", fontsize=8, color=INK)
        ax.set_ylabel("predicted % of parent", fontsize=8, color=INK)
        _style(ax)
        if i == 0:
            ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(f"Predicted vs expert frequency per well (values below {FREQ_FLOOR}% "
                 f"drawn at {FREQ_FLOOR}%)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def write_report(df: pd.DataFrame, methods: list[str], out: Path, role: str, title: str,
                 missing: Counter) -> list[Path]:
    """Write the CSV, markdown and plots; returns their paths."""
    out.mkdir(parents=True, exist_ok=True)
    methods = [m for m in methods if m in set(df["method"])]
    labels = gate_labels(sorted(df["gate"].unique()))
    summary = summarize(df)
    paths = [out / f"results_{role}.csv", out / f"report_{role}.md",
             out / f"f1_{role}.png", out / f"frequency_{role}.png"]
    df.assign(gate_name=df["gate"].map(labels)).to_csv(paths[0], index=False)
    paths[1].write_text(markdown(df, summary, methods, labels, title, missing), encoding="utf-8")
    plot_f1(summary, methods, labels, paths[2])
    plot_frequency(df, methods, paths[3])
    return paths


def select_gates(pairs: list[tuple[dict, str]], wanted: list[str]) -> list[tuple[dict, str]]:
    """Keep pairs whose gate key, gate id or last path element is in wanted."""
    w = set(wanted)
    return [(s, g) for s, g in pairs if {g, gate_id(g), g.split("/")[-1]} & w]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evaluation", default="als", help="evaluation in the split, e.g. als")
    parser.add_argument("--role", default="test", help="role whose samples are scored")
    parser.add_argument("--methods", nargs="+", help="default: every method folder present")
    parser.add_argument("--gates", nargs="+", help="gate keys, gate ids or last path elements")
    parser.add_argument("--out", type=Path, help="default: data/_results/<evaluation>/")
    args = parser.parse_args()

    stages = dict(splits._evaluations(splits.load_split()))[args.evaluation]
    by_key = {splits.key(s): s for s in splits.load_registry()["samples"]}
    samples = [by_key[k] for k in stages[args.role]]
    pairs = [(s, g) for s in samples for g in cached_gates(s)]
    if args.gates:
        pairs = select_gates(pairs, args.gates)
    methods = args.methods or available_methods(args.evaluation)
    if not methods or not pairs:
        print(f"nothing to score: {len(methods)} methods, {len(pairs)} (sample, gate) pairs")
        return 1
    print(f"scoring {len(methods)} methods on {len(samples)} samples, {len(pairs)} pairs")
    df, missing = score(args.evaluation, methods, pairs)
    for m in methods:
        print(f"  {m}: {int((df['method'] == m).sum()) if len(df) else 0} wells scored, "
              f"{missing[m]} missing")
    if not len(df):
        print("no predictions found")
        return 1
    out = args.out or results_root(args.evaluation)
    title = f"{args.evaluation} / {args.role}: gate scores"
    for p in write_report(df, methods, out, args.role, title, missing):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
