"""Iterative fine-tuning from a lab's accepted gates (called by the worker).

add_labels: a user accepted (or corrected) the gates of one sample -> training examples in the
  cache format (docs/cache-format.md), dataset "lab_<lab>". The sample is gated with the accepted
  polygons (cascade); each accepted gate becomes one example: parent density, accepted polygon
  as the mask, the lab template, and (cytokine gates) the donor's unstimulated well.
  An individual is held out (role "val") when its hash falls in VAL_SHARE, so held-out
  individuals stay held out as labels accumulate.
retrain: a new version that starts from the lab's current shared model (or the base model),
  trains on ALL the lab's examples (replay) at LR with early stopping on the held-out
  individuals, and graduates gates with >= GRADUATE accepted samples to their own UNITO.
  The version is promoted (made current) only if the held-out mean frequency error improves and
  no gate gets worse by more than MAX_WORSE_PP; otherwise the old version stays current.
"""

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from flow_er.cytometry.io import FcsFile
from flow_er.ml import cache, images, registry
from flow_er.ml.gatetree import GateTree, inside, template_mask
from flow_er.ml.runtime import MAX_IMAGE_CELLS, GatingResult, autogate, read_values

LR = 1e-4
PATIENCE = 40
EPOCHS = 300
VAL_SHARE = 0.2
GRADUATE = 15                 # accepted samples of a gate before it gets its own UNITO
MAX_WORSE_PP = 1.0


def lab_dataset(lab: str) -> str:
    return f"lab_{lab}"


def role_of(individual: str) -> str:
    """'val' for a fixed VAL_SHARE of individuals (by hash), else 'train'."""
    h = int(hashlib.sha256(individual.encode()).hexdigest()[:8], 16)
    return "val" if (h % 1000) < VAL_SHARE * 1000 else "train"


# --- labels -------------------------------------------------------------------------------------


def add_labels(lab: str, fcs_path: str | Path, tree: GateTree,
               accepted: dict[str, np.ndarray], individual: str, sample_id: str | None = None,
               condition: str = "", unstim: GatingResult | None = None,
               comp: tuple[np.ndarray, list[str]] | None = None, seed: int = 0) -> list[Path]:
    """Write one training example per accepted gate; returns the written model files.
    `accepted`: gate key -> polygon (display units). Gates not in it keep the template."""
    fcs = FcsFile(Path(fcs_path))
    sample_id = sample_id or fcs.path.stem
    labelled = tree.with_polygons(accepted, {"kind": "accepted", "sample": sample_id})
    result = autogate(fcs.path, labelled, comp=comp, use_models=False, seed=seed)
    channels = tree.channels()
    values = read_values(fcs, channels, result.comp)
    rng = np.random.default_rng(seed)
    dataset = lab_dataset(lab)
    written = []
    for spec in labelled.gates:
        if spec.key not in accepted:
            continue
        parent = (result.gates[spec.parent].membership if spec.parent
                  else np.ones(fcs.n_events, bool))
        cells = np.flatnonzero(parent)
        if len(cells) > MAX_IMAGE_CELLS:
            cells = np.sort(rng.choice(cells, MAX_IMAGE_CELLS, replace=False))
        cols = [channels.index((spec.x, spec.compensated[0])),
                channels.index((spec.y, spec.compensated[1]))]
        disp = images.to_display(values[cells][:, cols], [spec.x, spec.y],
                                 [fcs.pnr[spec.x], fcs.pnr[spec.y]])
        polygon = np.asarray(accepted[spec.key], float)
        arrays = {"density": images.density(disp), "mask": template_mask(polygon),
                  "template": template_mask(tree[spec.key].template_array()),
                  "cell_bins": images.pixel_index(disp).astype(np.uint8),
                  "cell_label": inside(polygon, disp), "cell_index": cells.astype(np.uint32)}
        if spec.is_cytokine and unstim is not None and spec.key in unstim.gates:
            arrays["unstim"] = unstim.gates[spec.key].density
        g = result.gates[spec.key]
        meta = {
            "format": cache.FORMAT, "recipe_id": images.recipe_id(), "dataset": dataset,
            "sample": f"{dataset}/{sample_id}", "sample_id": sample_id, "individual": individual,
            "condition": condition, "cohort": "", "gate": spec.key,
            "gate_id": cache.gate_id(spec.key), "parent": spec.parent,
            "is_cytokine": spec.is_cytokine,
            **{axis: {"channel": c, "lo": 0.0, "hi": 1.0, "transform": {
                **(images.RECIPE["scatter"] if images.is_scatter(c)
                   else images.RECIPE["fluorescence"]), "t": fcs.pnr[c]}}
               for axis, c in zip(("x", "y"), (spec.x, spec.y))},
            "roles": {dataset: role_of(individual)}, "template_source": "lab template",
            "unstim_source": None, "source_fcs": str(fcs.path),
            "accepted_at": datetime.now().isoformat(timespec="seconds"),
            "cells": {"n_parent": int(parent.sum()), "n_kept": len(cells),
                      "subsampled": bool(parent.sum() > len(cells)), "seed": seed},
            "counts": {"flowkit": g.count, "flowkit_parent": g.parent_count,
                       "flowjo": float("nan"), "flowjo_parent": float("nan")},
        }
        out = cache.cache_dir() / dataset / meta["gate_id"] / f"{sample_id}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **arrays)
        out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        written.append(out)
    return written


# --- held-out comparison --------------------------------------------------------------------------


def frequency_errors(predict, items) -> dict[str, float]:
    """Per gate: mean |predicted % - accepted %| (pp) over `items`; `predict(item, arrays)` returns
    a (101, 101) probability map, each cell takes its pixel."""
    errs: dict[str, list[float]] = {}
    for it in items:
        with np.load(it.path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        prob = predict(it, arrays)
        b = arrays["cell_bins"].astype(np.intp)
        pred = prob[b[:, 1], b[:, 0]] > 0.5
        true = arrays["cell_label"]
        err = 100 * abs(pred.mean() - true.mean()) if len(true) else 0.0
        errs.setdefault(it.meta["gate"], []).append(err)
    return {g: float(np.mean(v)) for g, v in errs.items()}


def promotion_decision(current: dict[str, float], candidate: dict[str, float],
                       max_worse: float = MAX_WORSE_PP) -> tuple[bool, str]:
    """Promote only if the held-out mean error improves and no gate is worse by > max_worse pp."""
    gates = sorted(set(current) & set(candidate))
    if not gates:
        return False, "no held-out examples to compare on"
    worse = {g: candidate[g] - current[g] for g in gates if candidate[g] - current[g] > max_worse}
    cur, cand = np.mean([current[g] for g in gates]), np.mean([candidate[g] for g in gates])
    if worse:
        return False, f"gates worse by > {max_worse} pp: {sorted(worse)}"
    if cand >= cur:
        return False, f"held-out mean error {cand:.3f} pp is not below current {cur:.3f} pp"
    return True, f"held-out mean error {cur:.3f} -> {cand:.3f} pp"


def _router_predict(lab: str, version: str | None):
    """The prediction the router would make for each held-out item under `version`."""
    def predict(item, arrays):
        info = registry.version_info(lab, version) if version else None
        gid = item.meta["gate_id"]
        model = None
        if info:
            for d in ([info["gates"].get(gid)] if info["gates"].get(gid) else []) + \
                     ([info["shared"]] if info.get("shared") else []):
                model = registry._predictor(f"{lab}/{version}", registry._abs(d))
                if model:
                    break
        model = model or registry._predictor("base/shared",
                                             registry.models_root() / "base" / "shared")
        if model is None:                                    # template copy
            return arrays["template"].astype(float)
        return model.predict(arrays["density"], arrays["template"], arrays.get("unstim"))
    return predict


# --- retraining -----------------------------------------------------------------------------------


def retrain(lab: str, lr: float = LR, patience: int = PATIENCE, epochs: int = EPOCHS,
            batch: int = 8, threads: int | None = None, graduate: int = GRADUATE,
            shared_arch: dict | None = None, unito_arch: dict | None = None,
            max_minutes: float | None = None, log=print) -> dict:
    """Train a new version on all the lab's examples; promote it only if it is better.
    `max_minutes` caps each training run (the shared model and each graduated gate); `log`
    receives progress lines (one per epoch: loss and held-out Dice)."""
    from flow_er.ml.unet import dataset as ds
    from flow_er.ml.unet import model as unet_model
    from flow_er.ml.unet import train

    evaluation = lab_dataset(lab)
    train_items = ds.select(evaluation, ["train"])
    val_items = ds.select(evaluation, ["val"])
    if not train_items or not val_items:
        raise ValueError(f"lab {lab!r}: {len(train_items)} training and {len(val_items)} held-out "
                         "examples; need both (the held-out share is by individual)")
    current = registry.current_version(lab)
    info = registry.version_info(lab) if current else None
    base = registry.models_root() / "base" / "shared"
    init = (registry._abs(info["shared"]) if info and info.get("shared") else
            base if (base / "model.pt").exists() else None)
    version = registry.next_version_name(lab)
    vdir = registry.lab_dir(lab) / version
    common = {"evaluation": evaluation, "train_roles": ("train",), "val_roles": ("val",),
              "lrs": (lr,), "batches": (batch,), "epochs": epochs, "patience": patience,
              "threads": threads, "max_seconds": max_minutes * 60 if max_minutes else None}

    channels = ds.SHARED_CHANNELS
    arch = shared_arch or unet_model.shared_arch(len(channels), None if init else "imagenet")
    if init:      # fine-tune: the architecture is the starting model's
        arch = json.loads((Path(init) / "recipe.json").read_text())["arch"]
    out = train.train(train.Config(model="_candidate", arch=arch, channels=channels,
                                   init=str(init) if init else None, augment=ds.Augment(),
                                   out_name=version, **common), log=log)
    shared_dir = vdir / "shared"
    shutil.rmtree(shared_dir, ignore_errors=True)
    shutil.move(str(out), shared_dir)

    # Graduation: gates with enough accepted samples also get their own UNITO.
    n_samples = {}
    for it in train_items + val_items:
        n_samples.setdefault(it.meta["gate"], set()).add(it.meta["sample_id"])
    gate_dirs = {}
    for gk, samples in sorted(n_samples.items()):
        if len(samples) < graduate or not any(it.meta["gate"] == gk for it in val_items):
            continue
        gid = cache.gate_id(gk)
        old = registry._abs(info["gates"][gid]) if info and gid in info["gates"] else None
        out = train.train(train.Config(
            model="_candidate", arch=unito_arch or unet_model.unito_arch(),
            channels=ds.UNITO_CHANNELS, gate_keys=[gk], init=str(old) if old else None,
            out_name=f"{version}_{gid}", **{**common, "lrs": (lr if old else 1e-3,)}), log=log)
        gate_dirs[gid] = vdir / "gates" / gid
        shutil.rmtree(gate_dirs[gid], ignore_errors=True)
        shutil.move(str(out), gate_dirs[gid])

    # Held-out comparison: the router's answer now vs the candidate's.
    before = frequency_errors(_router_predict(lab, current), val_items)
    shared_pred = registry.Predictor(f"{lab}/{version}/shared", shared_dir)
    cand_shared = frequency_errors(
        lambda it, a: shared_pred.predict(a["density"], a["template"], a.get("unstim")),
        val_items)
    keep = {}
    for gid, d in gate_dirs.items():
        p = registry.Predictor(f"{lab}/{version}/unito/{gid}", d)
        gate_items = [it for it in val_items if it.meta["gate_id"] == gid]
        err = frequency_errors(lambda it, a, p=p: p.predict(a["density"], a["template"]),
                               gate_items)
        gk = gate_items[0].meta["gate"]
        if err[gk] <= cand_shared.get(gk, np.inf):
            keep[gid] = d
            cand_shared[gk] = err[gk]
        else:
            shutil.rmtree(d)
    promoted, reason = promotion_decision(before, cand_shared)
    registry.register_version(
        lab, gates=keep, shared=shared_dir, parent=current, version=version,
        metrics={"held_out_error_pp": {"current": before, "candidate": cand_shared},
                 "promoted": promoted, "reason": reason, "lr": lr, "patience": patience,
                 "held_out_individuals": ds.individuals_of(val_items)},
        training_samples=sorted({it.meta["sample"] for it in train_items}),
        notes="retrained from " + (str(init) if init else "ImageNet encoder"),
        make_current=promoted)
    log(f"{lab}: {version} {'promoted' if promoted else 'kept as candidate'}: {reason}")
    return {"version": version, "promoted": promoted, "reason": reason,
            "graduated": sorted(keep), "metrics": {"current": before, "candidate": cand_shared}}


def replace_polygons(result: GatingResult) -> dict[str, np.ndarray]:
    """A result's polygons as accepted gates (the user accepted everything unchanged)."""
    return {k: g.polygon_display for k, g in result.gates.items() if g.polygon_display is not None}

