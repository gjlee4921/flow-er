"""FCS + workspace -> density images and gate masks on disk, for training.

For every sample an evaluation uses (from the frozen split), load its FCS file ONCE, gate it with
FlowKit through its own workspace, and write the cache in <data>/_cache/v1/
(format: docs/cache-format.md):
- a per-sample source file: the compensated values of every gated channel, FlowKit's membership
  of every gate (the truth), each gate's parent density image and expert mask
  (ml/images.py recipe), and each gate's geometry and workspace transforms (baselines move
  templates through raw space);
- per-(sample, gate) model files: density, mask, template, unstimulated-well density and the
  parent cells' pixels and labels, plus a meta JSON.
Training never reads FCS files again.

    uv run python scripts/build_dataset.py --evaluation als
    uv run python scripts/build_dataset.py --evaluation pretrain --previews 0
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import flowkit as fk
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flow_er.config import settings
from flow_er.ml import images, splits
from flow_er.ml.cache import (
    FORMAT,
    Cached,
    cache_dir,
    gate_id,
    gate_path,
    load_sample,
    sample_id,
    sample_path,
)

ALS_DATASETS = ("als_ics_screen3", "als_ics_screen4")
ALS_REFERENCE_DONOR = "human:2008"      # first ALS training donor: the template for every gate
MAX_CELLS = 200_000


def workspace_path(sample: dict) -> Path:
    folder = settings.data_dir / sample["dataset"]
    direct = folder / sample["workspace"]
    if direct.exists():
        return direct
    copies = sorted((folder / "by_file_id").glob(f"*__{sample['workspace']}"))
    if not copies:
        raise FileNotFoundError(f"workspace {sample['workspace']} not under {folder}")
    return copies[0]


def name_and_path(gate_key: str) -> tuple[str, tuple[str, ...]]:
    parts = gate_key.split("/")
    return parts[-1], ("root", *parts[:-1])


def describe(obj) -> dict:
    """JSON-able description of a FlowKit gate or transform (enough to rebuild it)."""
    out = {"class": type(obj).__name__}
    for k, v in vars(obj).items():
        if k.startswith("_"):
            continue
        if k == "dimensions":
            v = [{"id": d.id, "compensation_ref": d.compensation_ref,
                  "transformation_ref": d.transformation_ref,
                  "min": getattr(d, "min", None), "max": getattr(d, "max", None)} for d in v]
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, list):
            v = [list(map(float, p)) if isinstance(p, (tuple, list)) else p for p in v]
        out[k] = v
    return out


# --- per-sample source layer ------------------------------------------------------------------


def build_sample(sample: dict) -> dict:
    """Gate one sample with FlowKit and return the arrays of its source file."""
    sid, fcs = sample["sample"], settings.data_dir / sample["file"]
    fk_sample = fk.Sample(str(fcs), sample_id=sid, use_flowjo_labels=True)
    pnr = {ch: float(r) for ch, r in zip(fk_sample.pnn_labels, [
        fk_sample.channels.loc[i, "pnr"] for i in fk_sample.channels.index])}
    wsp = fk.Workspace(str(workspace_path(sample)), fcs_samples=[fk_sample],
                       load_missing_file_data=True)
    wsp.analyze_samples(sample_id=sid, cache_events=False, use_mp=False)
    gs = wsp.get_gating_strategy(sid)
    comp = wsp.get_comp_matrix(sid)
    raw_events = fk_sample.get_events(source="raw")
    comp_events = comp.apply(fk_sample) if comp is not None else raw_events
    labels_pnn = fk_sample.pnn_labels

    gate_keys = list(sample["usable_gates"])
    columns, values = [], []          # gated channels; "|raw" marks an uncompensated dimension

    def column(dim) -> int:
        compensated = dim.compensation_ref not in (None, "uncompensated")
        name = dim.id if compensated else f"{dim.id}|raw"
        if name not in columns:
            idx = labels_pnn.index(dim.id)
            columns.append(name)
            values.append((comp_events if compensated else raw_events)[:, idx])
        return columns.index(name)

    n = raw_events.shape[0]
    labels = np.zeros((len(gate_keys), n), bool)
    parent = np.full(len(gate_keys), -1, np.int16)
    dims = np.zeros((len(gate_keys), 2), np.int16)
    gates_meta, xform_meta = [], {}
    for g, key in enumerate(gate_keys):
        name, path = name_and_path(key)
        gate = gs.get_gate(name, path)
        labels[g] = wsp.get_gate_membership(sid, name, path)
        parent_key = "/".join(path[1:])
        parent[g] = gate_keys.index(parent_key) if parent_key else -1
        dims[g] = [column(d) for d in gate.dimensions[:2]]
        for d in gate.dimensions[:2]:
            if d.transformation_ref and d.transformation_ref not in xform_meta:
                xform_meta[d.transformation_ref] = describe(gs.transformations[d.transformation_ref])
        gates_meta.append({"key": key, **describe(gate)})

    raw = np.column_stack(values).astype(np.float32)
    chan = [c.split("|")[0] for c in columns]
    pnrs = [pnr[c] for c in chan]
    disp = images.to_display(raw, chan, pnrs)
    imgs = np.zeros((len(gate_keys), images.BINS, images.BINS), np.float32)
    masks = np.zeros_like(imgs, dtype=np.uint8)
    for g, key in enumerate(gate_keys):
        gate = gs.get_gate(*name_and_path(key))
        in_parent = labels[parent[g]] if parent[g] >= 0 else np.ones(n, bool)
        x, y = dims[g]
        imgs[g] = images.density(disp[in_parent][:, [x, y]])
        masks[g] = images.gate_mask(gate, gs.transformations, [chan[x], chan[y]],
                                    [pnrs[x], pnrs[y]])
    return {"raw": raw, "columns": np.array(columns), "pnr": np.array(pnrs, np.float64),
            "labels": labels, "parent": parent, "dims": dims, "images": imgs, "masks": masks,
            "gate_keys": np.array(gate_keys),
            "meta": json.dumps({"sample": splits.key(sample), "registry": sample,
                                "recipe": images.RECIPE, "gates": gates_meta,
                                "transforms": xform_meta})}


def mask_agreement(arrays: dict) -> list[float]:
    """Per gate: F1 of 'each cell takes its mask pixel' against FlowKit, on the parent cells.
    Close to 1 means images, masks and labels line up; below 1 is the 101-pixel resolution."""
    chan = [c.split("|")[0] for c in arrays["columns"]]
    disp = images.to_display(arrays["raw"], chan, list(arrays["pnr"]))
    out = []
    for g in range(len(arrays["gate_keys"])):
        p = arrays["parent"][g]
        in_parent = arrays["labels"][p] if p >= 0 else np.ones(arrays["labels"].shape[1], bool)
        x, y = arrays["dims"][g]
        pred = images.cells_from_mask(arrays["masks"][g], disp[in_parent][:, [x, y]])
        true = arrays["labels"][g][in_parent]
        tp = (pred & true).sum()
        denom = pred.sum() + true.sum()
        out.append(float(2 * tp / denom) if denom else 1.0)
    return out


def preview(arrays: dict, out: Path) -> None:
    keys = list(arrays["gate_keys"])
    cols = 4
    rows = -(-len(keys) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    for ax, g in zip(axes.ravel(), range(len(keys))):
        ax.imshow(np.log1p(arrays["images"][g]), origin="lower", cmap="viridis")
        ax.contour(arrays["masks"][g], levels=[0.5], colors="red", linewidths=0.8)
        ax.set_title(keys[g].split("/")[-1][:28], fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


# --- per-(sample, gate) model files ----------------------------------------------------------


def is_cytokine(sample: dict, gate_key: str) -> bool:
    return sample["dataset"] in ALS_DATASETS and bool(
        re.match(r"(IFN|IL-)", gate_key.split("/")[-1]))


def roles_by_sample(split: dict) -> dict[str, dict[str, str]]:
    roles = defaultdict(dict)
    for role, keys in split["pretrain"]["stages"].items():
        for k in keys:
            roles[k]["pretrain"] = role
    for name, stages in splits._evaluations(split):
        for role, keys in stages.items():
            for k in keys:
                roles[k][name] = role
    return roles


def reference_sample(sample: dict, gate_key: str, split: dict, by_key: dict) -> dict | None:
    """The template's source for one gate: ALS = donor 2008's unstimulated well; otherwise the
    first training sample (sorted by key) of the same dataset and workspace that has the gate
    (workspaces differ in their trees). None when no training sample has the gate."""
    if sample["dataset"] in ALS_DATASETS:
        return next(s for s in by_key.values() if s["individual"] == ALS_REFERENCE_DONOR
                    and s["condition"] == "unstimulated" and s["dataset"] == "als_ics_screen3")
    train = sorted({k for _, st in splits._evaluations(split) for k in st.get("train", [])}
                   | set(split["pretrain"]["stages"]["train"]))
    return next((by_key[k] for k in train if by_key[k]["dataset"] == sample["dataset"]
                 and by_key[k]["workspace"] == sample["workspace"]
                 and gate_key in by_key[k]["usable_gates"]), None)


def agreeing_gates(registry: dict) -> set[tuple[str, str, str]]:
    """(dataset, workspace, gate key) whose FlowKit counts agree with FlowJo (registry)."""
    return {(d, g["workspace"], g["gate"]) for d, meta in registry["datasets"].items()
            for g in meta["gates"] if g["agrees_with_flowjo"] and not g["qc_time_gate"]}


def inspect_counts(sample: dict) -> dict[str, tuple[float, float]]:
    """gate key -> (FlowKit count, FlowJo count) from the inspection CSV of the sample's
    workspace (scripts/inspect_wsp.py)."""
    stem = re.sub(r"[^\w.-]+", "_", Path(sample["workspace"]).stem).strip("_")
    df = pd.read_csv(settings.data_dir / "_inspect" / sample["dataset"] / f"{stem}.counts.csv")
    df = df[df["sample"] == sample["sample"]]
    return {g: (k, j) for g, k, j in zip(df["gate"], df["flowkit"], df["flowjo"])}


def write_model_files(sample: dict, src: Cached, templates: dict[str, tuple[np.ndarray, str]],
                      unstim: Cached | None, roles: dict, unstim_key: str | None,
                      keep: set[str] | None = None) -> int:
    """templates: gate key -> (template mask, source sample key); gates without a template get
    an empty one. keep: only these gate keys (None = all)."""
    counts = inspect_counts(sample)
    rng = np.random.default_rng(0)
    n_events = src.labels.shape[1]
    written = 0
    for g, gk in enumerate(src.gate_keys):
        if keep is not None and gk not in keep:
            continue
        template, template_key = templates.get(gk, (np.zeros_like(src.masks[g]), None))
        cells = np.flatnonzero(src.in_parent(g))
        n_parent = len(cells)
        if n_parent > MAX_CELLS:
            cells = np.sort(rng.choice(cells, MAX_CELLS, replace=False))
        arrays = {"density": src.images[g], "mask": src.masks[g],
                  "template": template,
                  "cell_bins": images.pixel_index(src.gate_display(g, cells)).astype(np.uint8),
                  "cell_label": src.labels[g][cells], "cell_index": cells.astype(np.uint32)}
        cyto = is_cytokine(sample, gk)
        if cyto and unstim is not None:
            arrays["unstim"] = unstim.images[unstim.gate(gk)]
        parent_key = "/".join(gk.split("/")[:-1])
        chans, pnrs = src.gate_channels(g)
        fk_count, fj_count = counts.get(gk, (np.nan, np.nan))
        fk_parent, fj_parent = (counts.get(parent_key, (np.nan, np.nan)) if parent_key
                                else (n_events, sample["events"]))
        meta = {
            "format": FORMAT, "recipe_id": images.recipe_id(),
            "dataset": sample["dataset"], "sample": splits.key(sample),
            "sample_id": sample_id(sample), "individual": sample["individual"],
            "condition": sample["condition"], "cohort": sample["cohort"],
            "gate": gk, "gate_id": gate_id(gk), "parent": parent_key, "is_cytokine": cyto,
            **{axis: {"channel": c,
                      "transform": {**(images.RECIPE["scatter"] if images.is_scatter(c)
                                       else images.RECIPE["fluorescence"]), "t": r},
                      "lo": 0.0, "hi": 1.0}
               for axis, c, r in zip(("x", "y"), chans, pnrs)},
            "roles": roles, "template_source": template_key,
            "unstim_source": unstim_key if cyto else None,
            "cells": {"n_parent": int(n_parent), "n_kept": len(cells),
                      "subsampled": bool(n_parent > MAX_CELLS), "seed": 0},
            "counts": {"flowkit": float(fk_count), "flowkit_parent": float(fk_parent),
                       "flowjo": float(fj_count), "flowjo_parent": float(fj_parent)},
        }
        out = gate_path(sample, gk)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **arrays)
        out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evaluation", required=True,
                        help="evaluation name in split_v1.json (e.g. als, mo280_round1), or "
                             "'pretrain' for the pool")
    parser.add_argument("--stages", nargs="*", help="only these stages (default: all)")
    parser.add_argument("--previews", type=int, default=2, help="PNG previews to write")
    parser.add_argument("--force", action="store_true", help="rebuild existing source files")
    args = parser.parse_args()

    split, registry = splits.load_split(), splits.load_registry()
    stages = (split["pretrain"]["stages"] if args.evaluation == "pretrain"
              else dict(splits._evaluations(split))[args.evaluation])
    by_key = {splits.key(s): s for s in registry["samples"]}
    keys = sorted({k for st, ks in stages.items() if not args.stages or st in args.stages
                   for k in ks})
    # Templates and unstimulated wells are model inputs too: cache their sources as well.
    refs = {(k, g): reference_sample(by_key[k], g, split, by_key)
            for k in keys for g in by_key[k]["usable_gates"]}
    unstims = {}
    for k in keys:
        s = by_key[k]
        if s["dataset"] in ALS_DATASETS:
            unstims[(s["dataset"], s["individual"])] = next(
                (u for u in by_key.values() if u["dataset"] == s["dataset"] and u["individual"]
                 == s["individual"] and u["condition"] == "unstimulated" and u["usable"]), None)
    sources = sorted(set(keys) | {splits.key(r) for r in refs.values() if r}
                     | {splits.key(u) for u in unstims.values() if u})
    print(f"recipe {images.recipe_id()}; {len(keys)} samples ({len(sources)} with template and "
          f"unstimulated sources) -> {cache_dir()}")
    (cache_dir() / "previews").mkdir(parents=True, exist_ok=True)
    (cache_dir() / "recipe.json").write_text(json.dumps(images.RECIPE, indent=1))
    agreement, t0 = {}, time.time()
    for i, k in enumerate(sources):
        sample = by_key[k]
        out = sample_path(sample)
        if out.exists() and not args.force:
            arrays = dict(np.load(out, allow_pickle=False))
        else:
            arrays = build_sample(sample)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, **arrays)
        agreement[k] = dict(zip(arrays["gate_keys"], mask_agreement(arrays)))
        if i < args.previews:
            preview(arrays, cache_dir() / "previews" / f"{out.stem}.png")
        print(f"  [{i + 1}/{len(sources)}] {k}: {arrays['raw'].shape[0]} events, "
              f"{len(arrays['gate_keys'])} gates, {time.time() - t0:.0f}s", flush=True)
        del arrays

    roles = roles_by_sample(split)
    # Pretraining uses only gates whose FlowKit counts agree with FlowJo (split_v1 rule).
    agree = agreeing_gates(registry) if args.evaluation == "pretrain" else None
    ref_masks: dict[str, dict[str, np.ndarray]] = {}
    files = 0
    for k in keys:
        sample = by_key[k]
        templates = {}
        for g in sample["usable_gates"]:
            ref = refs[(k, g)]
            if ref is None:
                continue
            rk = splits.key(ref)
            if rk not in ref_masks:
                rc = load_sample(ref)
                ref_masks[rk] = dict(zip(rc.gate_keys, rc.masks))
            templates[g] = (ref_masks[rk][g], rk)
        unstim = unstims.get((sample["dataset"], sample["individual"]))
        keep = None if agree is None else {g for g in sample["usable_gates"]
                                           if (sample["dataset"], sample["workspace"], g) in agree}
        files += write_model_files(sample, load_sample(sample), templates,
                                   load_sample(unstim) if unstim else None, roles.get(k, {}),
                                   splits.key(unstim) if unstim else None, keep)
    print(f"wrote {files} per-(sample, gate) model files in {time.time() - t0:.0f}s")

    gates = sorted({g for a in agreement.values() for g in a})
    print("\nmask F1 vs FlowKit labels (median / min over samples), per gate:")
    for g in gates:
        vals = [a[g] for a in agreement.values() if g in a]
        print(f"  {np.median(vals):.3f} / {np.min(vals):.3f}  {g}")
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
