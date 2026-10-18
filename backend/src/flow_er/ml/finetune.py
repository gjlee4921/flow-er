"""Iterative fine-tuning from a lab's accepted gates (called by the worker).

add_labels: a user accepted (or corrected) the gates of one sample -> training examples in the
  cache format (docs/cache-format.md), dataset "lab_<lab>". The sample is gated with the accepted
  polygons (cascade); each accepted gate becomes one example: parent density, accepted polygon
  as the mask, the lab template, and (cytokine gates) the donor's unstimulated well.
  An individual is held out (role "val") when its hash falls in VAL_SHARE; a small lab with none
  in that share holds out the newest individual nobody has trained on. Held-out individuals are
  recorded in models/labs/<lab>/held_out.json and never switch, so every version is compared on
  the same people and nobody already trained on becomes a held-out check (assign_roles).
retrain: per-gate UNITO only, warm-started from the gate's model, trained on all the lab's
  examples at LR with early stopping on the held-out individuals; gates without a model graduate
  at >= GRADUATE accepted samples. Each gate is then answered by the best of its current model,
  the new model and template copy on the held-out examples (choose_route): a gate on a model
  falls back to template copy only when template copy is better by >= TEMPLATE_MARGIN_PP on
  >= MIN_HELD_OUT examples, and its model stays on disk ("parked") for the next retrain.
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
OUTDATED_AFTER = 3             # kept files the current version has not learned from
TEMPLATE_MARGIN_PP = 0.5       # template copy must win by this much to replace a gate's model
MIN_HELD_OUT = 2               # ... on at least this many held-out examples of that gate
HELD_OUT_FILE = "held_out.json"


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


def frozen_held_out(lab: str) -> set[str]:
    p = registry.lab_dir(lab) / HELD_OUT_FILE
    return set(json.loads(p.read_text())["individuals"]) if p.exists() else set()


def assign_roles(lab: str) -> set[str]:
    """Write each lab example's role (train / val); returns the held-out set.

    Held-out individuals are fixed once chosen (HELD_OUT_FILE). A newcomer, someone no version has
    trained on, is held out when its hash is in the share; if the lab then still holds out nobody
    and has >= 2 individuals, the newest newcomer is held out. Nobody switches later."""
    from flow_er.ml.unet import dataset as ds

    dataset = lab_dataset(lab)
    items = ds.select(dataset, ["train", "val"])
    first: dict[str, tuple[str, str]] = {}                 # individual -> (first accepted, sample)
    for it in items:
        key = (it.meta.get("accepted_at") or "", it.meta["sample"])
        first[it.meta["individual"]] = min(first.get(it.meta["individual"], key), key)
    trained_samples = {s for v in registry.list_versions(lab) for s in v.get("training_samples") or []}
    trained = {it.meta["individual"] for it in items if it.meta["sample"] in trained_samples}
    val = frozen_held_out(lab) & set(first)
    newcomers = [p for p in first if p not in val and p not in trained]
    val |= {p for p in newcomers if role_of(p) == "val"}
    if not val and len(first) >= 2 and newcomers:
        val = {max(newcomers, key=lambda p: (first[p], p))}
    if val:
        p = registry.lab_dir(lab) / HELD_OUT_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"individuals": sorted(val | frozen_held_out(lab))}, indent=1))
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


def choose_route(new: float, current: float | None, template: float, n_held_out: int,
                 margin: float = TEMPLATE_MARGIN_PP, min_held_out: int = MIN_HELD_OUT) -> str:
    """Which answers a gate after retraining: "new" (the new model), "current" (its model now) or
    "template" (template copy), from each one's held-out error (pp).

    A gate answered by template copy (current None) takes the new model only if it beats template
    copy. A gate on a model keeps the better of its current and new model, and falls back to
    template copy only if template copy is better by >= `margin` on >= `min_held_out` examples, so
    one odd held-out donor cannot take away a good model."""
    if current is None:
        return "new" if new < template else "template"
    best, err = ("new", new) if new < current else ("current", current)
    if n_held_out >= min_held_out and template + margin <= err:
        return "template"
    return best


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
    version has not seen, plus the gates below them. A gate that has a model (current or parked)
    warm-starts from it at `lr`; a gate without one needs >= `graduate` accepted samples and starts
    fresh. Each gate is then answered by its current model, the new model or template copy, as
    choose_route picks on the held-out examples. Promotion: the held-out mean error must improve
    and no gate may get worse by more than MAX_WORSE_PP. No shared model is trained. `max_minutes` caps
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

    current_parked = {g: registry._abs(d) for g, d in (info or {}).get("parked", {}).items()}
    new_gates, parked, decisions = dict(current_gates), dict(current_parked), {}
    template = _gates_predict("template", {})
    for gk in sorted(todo):
        gid = cache.gate_id(gk)
        old, old_parked = current_gates.get(gid), current_parked.get(gid)
        init = old or old_parked                              # a parked model is still a start
        gate_val = [it for it in val_items if it.meta["gate"] == gk]
        if not gate_val or not any(it.meta["gate"] == gk for it in train_items):
            decisions[gk] = {"kept": False, "why": "needs accepted samples from training and held-out individuals"}
            continue
        if init is None and len(n_samples[gk]) < graduate:
            decisions[gk] = {"kept": False, "why": f"{len(n_samples[gk])} accepted samples; a new gate needs {graduate}"}
            continue
        arch = json.loads((init / "recipe.json").read_text())["arch"] if init else (unito_arch or unet_model.unito_arch())
        log(f"{lab}: training {gk} ({'from its current model' if old else 'from its parked model' if init else 'new'}, "
            f"{len(n_samples[gk])} samples)")
        out = train.train(train.Config(
            model="_candidate", arch=arch, channels=ds.UNITO_CHANNELS, gate_keys=[gk],
            init=str(init) if init else None, out_name=f"{version}_{gid}",
            **{**common, "lrs": (lr if init else 1e-3,)}), log=log)
        d = vdir / "gates" / gid
        shutil.rmtree(d, ignore_errors=True)
        shutil.move(str(out), d)
        err = frequency_errors(_gates_predict(f"{lab}/{version}", {gid: d}), gate_val)[gk]
        base_err = frequency_errors(template, gate_val)[gk]
        cur_err = frequency_errors(_gates_predict(f"{lab}/{current}", {gid: old}), gate_val)[gk] if old else None
        route = choose_route(err, cur_err, base_err, len(gate_val))
        decisions[gk] = {"kept": route == "new", "route": route, "held_out_error_pp": err,
                         "template_copy_pp": base_err, "current_pp": cur_err, "n_held_out": len(gate_val),
                         "warm_start": init is not None}
        if route == "new":
            new_gates[gid] = d
            parked.pop(gid, None)
        elif route == "template":                             # keep the better model on disk
            new_gates.pop(gid, None)
            if old is not None and cur_err <= err:
                parked[gid] = old
                shutil.rmtree(d)
            else:
                parked[gid] = d
        else:
            shutil.rmtree(d)

    # Held-out comparison over every gate: the router's answer now vs the candidate's.
    before = frequency_errors(_router_predict(lab, current), val_items)
    after = frequency_errors(_gates_predict(f"{lab}/{version}", new_gates), val_items)
    promoted, reason = promotion_decision(before, after)
    graduated = sorted(g for g, d in new_gates.items() if current_gates.get(g) != d)
    registry.register_version(
        lab, gates=new_gates, shared=None, parent=current, version=version, parked=parked,
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


def kept_samples(lab: str) -> dict[str, str]:
    """Every sample the lab has accepted gates for -> the individual it came from."""
    out = {}
    d = cache.cache_dir() / lab_dataset(lab)
    for f in sorted(d.glob("*/*.json")) if d.exists() else []:
        m = json.loads(f.read_text())
        if m.get("sample"):
            out[m["sample"]] = m.get("individual") or ""
    return out


def lab_status(lab: str, outdated_after: int = OUTDATED_AFTER) -> dict:
    """Whether the lab's current models are live, outdated or offline, and why.

    offline: a model of the current version cannot run here (no weights, or another image recipe).
    outdated: the lab has kept >= `outdated_after` files the current version was not trained on
    (about one donor's wells), so a retrain has new material. Otherwise live; a lab with no
    version gates by template copy."""
    info = registry.version_info(lab)
    version = info["version"] if info else None
    broken = []
    for gid, d in (info or {}).get("gates", {}).items():
        d = registry._abs(d)
        try:
            ok = (d / "model.pt").exists() and                 json.loads((d / "recipe.json").read_text())["recipe_id"] == images.recipe_id()
        except (OSError, ValueError, KeyError):
            ok = False
        if not ok:
            broken.append(gid)
    learned = set((info or {}).get("training_samples") or []) |         set((info or {}).get("metrics", {}).get("accepted_samples") or [])
    new = {s: i for s, i in kept_samples(lab).items() if s not in learned}
    out = {"version": version, "gates": len((info or {}).get("gates", {})), "new_files": len(new),
           "new_individuals": len(set(new.values()) - {""}), "outdated_after": outdated_after}
    if broken:
        return {**out, "state": "offline",
                "why": f"{len(broken)} of {version}'s gate models cannot run here (missing weights "
                       "or another image recipe)"}
    if len(new) >= outdated_after:
        return {**out, "state": "outdated",
                "why": f"{len(new)} kept file(s) from {out['new_individuals']} individual(s) are "
                       f"not in {version or 'any version'} yet; retrain to use them"}
    why = (f"{version} answers {out['gates']} gate(s); the rest copy the template" if version
           else "no trained version yet; every gate copies the template")
    return {**out, "state": "live", "why": why + (f"; {len(new)} kept file(s) waiting" if new else "")}


def replace_polygons(result: GatingResult) -> dict[str, np.ndarray]:
    """A result's polygons as accepted gates (the user accepted everything unchanged)."""
    return {k: g.polygon_display for k, g in result.gates.items() if g.polygon_display is not None}

