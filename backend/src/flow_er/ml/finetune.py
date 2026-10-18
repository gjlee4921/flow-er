"""Iterative fine-tuning from a lab's accepted gates (called by the worker).

add_labels: a user accepted (or corrected) the gates of one sample -> training examples in the
  cache format (docs/cache-format.md), dataset "lab_<lab>". The sample is gated with the accepted
  polygons (cascade); each accepted gate becomes one example: parent density, accepted polygon
  as the mask, the lab template, and (cytokine gates) the donor's unstimulated well.
  An individual is held out (role "val") when its hash falls in VAL_SHARE, so held-out
  individuals stay held out as labels accumulate; a small lab with none in that share holds out
  the individual with the smallest hash (assign_roles, run by retrain).
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


def _hash(individual: str) -> int:
    return int(hashlib.sha256(individual.encode()).hexdigest()[:8], 16) % 1000


def role_of(individual: str) -> str:
    """'val' for a fixed VAL_SHARE of individuals (by hash), else 'train'."""
    return "val" if _hash(individual) < VAL_SHARE * 1000 else "train"


def held_out(individuals: set[str]) -> set[str]:
    """The held-out individuals: those in the hash share; if none of them is (small labs), the
    one with the smallest hash, so there is always something to compare on with >= 2 people."""
    val = {i for i in individuals if role_of(i) == "val"}
    if not val and len(individuals) >= 2:
        val = {min(individuals, key=_hash)}
    return val


def assign_roles(lab: str) -> set[str]:
    """Write each lab example's role (train / val) from held_out(); returns the held-out set."""
    from flow_er.ml.unet import dataset as ds

    dataset = lab_dataset(lab)
    items = ds.select(dataset, ["train", "val"])
    val = held_out({it.meta["individual"] for it in items})
    for it in items:
        role = "val" if it.meta["individual"] in val else "train"
        if it.meta["roles"].get(dataset) != role:
            it.meta["roles"][dataset] = role
            it.path.with_suffix(".json").write_text(json.dumps(it.meta, indent=1))
    return val


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
    """The prediction the router would make for each held-out item under `version`: the gate's
    per-gate UNITO if the version has one, else template copy."""
    info = registry.version_info(lab, version) if version else None
    return _gates_predict(f"{lab}/{version}", {g: registry._abs(d) for g, d in (info or {}).get("gates", {}).items()})


def _gates_predict(tag: str, gates: dict[str, Path]):
    """Predict each held-out item with its gate's model in `gates` (gate id -> model folder), or
    template copy for a gate without one."""
    def predict(item, arrays):
        gid = item.meta["gate_id"]
        model = registry._predictor(f"{tag}/unito/{gid}", gates[gid]) if gid in gates else None
        if model is None:                                    # template copy
            return arrays["template"].astype(float)
        return model.predict(arrays["density"], arrays["template"], arrays.get("unstim"))
    return predict


# --- retraining -----------------------------------------------------------------------------------


def gates_to_retrain(items, trained_on: set[str], gates: list[str] | None = None) -> set[str]:
    """The gate keys a retrain covers: `gates` (keys or gate ids) if given, else every gate with an
    accepted sample the current version has not seen; plus every gate below those."""
    keys = {it.meta["gate"] for it in items}
    if gates:
        by_id = {cache.gate_id(k): k for k in keys}
        todo = {g if g in keys else by_id[g] for g in gates if g in keys or g in by_id}
    else:
        todo = {it.meta["gate"] for it in items if it.meta["sample"] not in trained_on}
    children: dict[str, set[str]] = {}
    for it in items:
        children.setdefault(it.meta.get("parent") or "", set()).add(it.meta["gate"])
    stack = list(todo)
    while stack:
        for c in children.get(stack.pop(), ()):
            if c not in todo:
                todo.add(c)
                stack.append(c)
    return todo


def retrain(lab: str, gates: list[str] | None = None, lr: float = LR, patience: int = PATIENCE,
            epochs: int = EPOCHS, batch: int = 8, threads: int | None = None,
            graduate: int = GRADUATE, unito_arch: dict | None = None,
            max_minutes: float | None = None, log=print, on: str = "cpu",
            wait_minutes: float | None = None) -> dict:
    """A new version of the lab's per-gate UNITO models; made current only if it is better.

    Retrains `gates` (keys or ids), or by default the gates with accepted samples the current
    version has not seen, plus the gates below them. A gate that has a model warm-starts from it at
    `lr`; a gate without one needs >= `graduate` accepted samples and starts fresh. A gate's new
    model is kept only if its held-out error beats both its current model and template copy on
    that gate (the baseline). Promotion is unchanged: the held-out mean error must improve and no
    gate may get worse by more than MAX_WORSE_PP. No shared model is trained. `max_minutes` caps
    each gate's training; `log` receives one line per epoch (loss and held-out Dice).

    `on`: "cpu" trains in this process (on its CUDA GPU if it has one); "colab" sends the job to
    the notebook through settings.colab_jobs_dir and waits up to `wait_minutes` (ml.remote);
    "auto" uses Colab when the notebook's heartbeat is fresh, else the CPU."""
    if on not in ("cpu", "colab", "auto"):
        raise ValueError(f"on must be cpu, colab or auto, not {on!r}")
    if on == "auto":
        from flow_er.ml import remote
        on = "colab" if remote.notebook_alive() else "cpu"
        log(f"{lab}: retraining on {'the Colab GPU' if on == 'colab' else 'the CPU (no fresh notebook heartbeat)'}")
    if on == "colab":
        from flow_er.ml import remote
        params = {"gates": gates, "lr": lr, "patience": patience, "epochs": epochs, "batch": batch,
                  "threads": threads, "graduate": graduate, "unito_arch": unito_arch,
                  "max_minutes": max_minutes, "on": "cpu"}
        job = remote.submit(lab, params)
        log(f"{lab}: sent to the GPU as {job.name}")
        out = remote.wait(job, timeout=wait_minutes * 60 if wait_minutes else None, log=log)
        return remote.collect(lab, job, out, log=log)
    from flow_er.ml.unet import dataset as ds
    from flow_er.ml.unet import model as unet_model
    from flow_er.ml.unet import train

    evaluation = lab_dataset(lab)
    assign_roles(lab)
    train_items = ds.select(evaluation, ["train"])
    val_items = ds.select(evaluation, ["val"])
    if not train_items or not val_items:
        raise ValueError(f"lab {lab!r}: {len(train_items)} training and {len(val_items)} held-out "
                         "examples; retraining needs accepted samples from at least 2 individuals")
    items = train_items + val_items
    current = registry.current_version(lab)
    info = registry.version_info(lab) if current else None
    current_gates = {g: registry._abs(d) for g, d in (info or {}).get("gates", {}).items()}
    seen = set((info or {}).get("metrics", {}).get("accepted_samples") or (info or {}).get("training_samples") or [])
    todo = gates_to_retrain(items, seen, gates)
    if not todo:
        reason = f"no accepted samples since {current}" if current else "nothing to retrain"
        log(f"{lab}: {reason}")
        return {"version": None, "promoted": False, "reason": reason, "graduated": [], "gates": {},
                "metrics": {}}
    version = registry.next_version_name(lab)
    vdir = registry.lab_dir(lab) / version
    common = {"evaluation": evaluation, "train_roles": ("train",), "val_roles": ("val",),
              "batches": (batch,), "epochs": epochs, "patience": patience, "threads": threads,
              "max_seconds": max_minutes * 60 if max_minutes else None}
    n_samples: dict[str, set[str]] = {}
    for it in items:
        n_samples.setdefault(it.meta["gate"], set()).add(it.meta["sample_id"])

    new_gates, decisions = dict(current_gates), {}
    template = _gates_predict("template", {})
    for gk in sorted(todo):
        gid = cache.gate_id(gk)
        old = current_gates.get(gid)
        gate_val = [it for it in val_items if it.meta["gate"] == gk]
        if not gate_val or not any(it.meta["gate"] == gk for it in train_items):
            decisions[gk] = {"kept": False, "why": "needs accepted samples from training and held-out individuals"}
            continue
        if old is None and len(n_samples[gk]) < graduate:
            decisions[gk] = {"kept": False, "why": f"{len(n_samples[gk])} accepted samples; a new gate needs {graduate}"}
            continue
        arch = json.loads((old / "recipe.json").read_text())["arch"] if old else (unito_arch or unet_model.unito_arch())
        log(f"{lab}: training {gk} ({'from its current model' if old else 'new'}, {len(n_samples[gk])} samples)")
        out = train.train(train.Config(
            model="_candidate", arch=arch, channels=ds.UNITO_CHANNELS, gate_keys=[gk],
            init=str(old) if old else None, out_name=f"{version}_{gid}",
            **{**common, "lrs": (lr if old else 1e-3,)}), log=log)
        d = vdir / "gates" / gid
        shutil.rmtree(d, ignore_errors=True)
        shutil.move(str(out), d)
        err = frequency_errors(_gates_predict(f"{lab}/{version}", {gid: d}), gate_val)[gk]
        base_err = frequency_errors(template, gate_val)[gk]
        cur_err = frequency_errors(_gates_predict(f"{lab}/{current}", {gid: old}), gate_val)[gk] if old else None
        kept = err < base_err and (cur_err is None or err < cur_err)
        decisions[gk] = {"kept": kept, "held_out_error_pp": err, "template_copy_pp": base_err,
                         "current_pp": cur_err, "warm_start": old is not None}
        if kept:
            new_gates[gid] = d
        else:
            shutil.rmtree(d)

    # Held-out comparison over every gate: the router's answer now vs the candidate's.
    before = frequency_errors(_router_predict(lab, current), val_items)
    after = frequency_errors(_gates_predict(f"{lab}/{version}", new_gates), val_items)
    promoted, reason = promotion_decision(before, after)
    graduated = sorted(g for g, d in new_gates.items() if current_gates.get(g) != d)
    registry.register_version(
        lab, gates=new_gates, shared=None, parent=current, version=version,
        metrics={"held_out_error_pp": {"current": before, "candidate": after},
                 "promoted": promoted, "reason": reason, "lr": lr, "patience": patience,
                 "gates": decisions, "accepted_samples": sorted({it.meta["sample"] for it in items}),
                 "held_out_individuals": ds.individuals_of(val_items)},
        training_samples=sorted({it.meta["sample"] for it in train_items}),
        notes=f"per-gate UNITO retrain of {len(todo)} gate(s) from {current or 'template copy'}",
        make_current=promoted)
    log(f"{lab}: {version} {'promoted' if promoted else 'kept as candidate'}: {reason}")
    return {"version": version, "promoted": promoted, "reason": reason, "graduated": graduated,
            "gates": decisions, "metrics": {"current": before, "candidate": after}}


def replace_polygons(result: GatingResult) -> dict[str, np.ndarray]:
    """A result's polygons as accepted gates (the user accepted everything unchanged)."""
    return {k: g.polygon_display for k, g in result.gates.items() if g.polygon_display is not None}

