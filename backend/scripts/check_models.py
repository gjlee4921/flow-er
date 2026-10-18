"""Install downloaded model weights and check them against the predictions they were trained with.

    # 1. copy a downloaded models folder (or .zip of it) into MODELS_DIR; existing files are kept
    #    unless identical, so a partial download can be added to
    uv run python scripts/check_models.py install ~/Downloads/flower_models.zip
    # 2. every model folder: are the weights there, and do they load into the recipe's model?
    uv run python scripts/check_models.py verify
    # 3. run each model on this CPU and compare with the per-cell probabilities saved by the run
    #    that trained it (data/_results/<evaluation>/<method>/)
    uv run python scripts/check_models.py reproduce [--roles test] [--methods unito_gpu ...]
    # 4. make the checked ALS models the router's: models/base/shared + lab "als" version
    uv run python scripts/check_models.py register
    # 5. gate the ALS test donors' peptide wells end to end (cascade, unstimulated well) with the
    #    lab's models and with template copy; F1 and frequency error against the expert
    uv run python scripts/check_models.py runtime [--lab als]
    # 6. does retraining help? teach a throwaway copy of the lab the expert's gates on half the
    #    test donors (+ the fine-tune donors), retrain, gate the other half; two folds, figures
    uv run python scripts/check_models.py finetune --out <folder> [--max-minutes 6] [--folds A B]

Model folders: models/<evaluation>/<model>/<gate_id>/ (per gate) or .../shared/ (shared), and
models/pretrain/shared{_1ch}/shared/ for the pretrained base (scripts/train_gate.py).
"""

import argparse
import filecmp
import hashlib
import json
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

from flow_er.config import settings
from flow_er.ml import cache, registry
from flow_er.ml.gatetree import GateTree, from_workspace
from flow_er.ml.unet import dataset as ds
from flow_er.ml.unet import infer

# Methods whose results come from something other than one of our U-Net folders.
NOT_UNETS = {"template_copy", "classical", "headline", "template_perfect"}
# Agreement needed to call a model reproduced: every per-cell probability within PROB_TOL AND at
# most MAX_FLIPS of cells on the other side of 0.5. GPU and CPU convolutions differ only in the last
# bits; another model of the same gate differs by > 0.5 somewhere while flipping < 0.1% of cells,
# so the flip share alone cannot tell the weights apart (checked on the 4 CPU-trained ALS gates).
PROB_TOL = 1e-2
MAX_FLIPS = 1e-4


def model_dirs() -> list[Path]:
    """Every folder holding a recipe.json under MODELS_DIR (lab versions excluded)."""
    root = settings.models_dir
    return sorted(p.parent for p in root.rglob("recipe.json")
                  if "labs" not in p.relative_to(root).parts)


def weights_for(evaluation: str, method: str) -> tuple[str, Path] | None:
    """('per_gate', root) or ('shared', folder) for a results method, if its recipe is here."""
    root = settings.models_dir
    if method.startswith("shared_zero_shot"):
        suffix = method.removeprefix("shared_zero_shot")
        d = root / "pretrain" / f"shared{suffix}" / "shared"
        return ("shared", d) if (d / "recipe.json").exists() else None
    d = root / evaluation / method
    if (d / "shared" / "recipe.json").exists():
        return "shared", d / "shared"
    if any(d.glob("*/recipe.json")):
        return "per_gate", d
    return None


def cmd_install(src: Path) -> None:
    """Copy model files from a folder or zip into MODELS_DIR (found by their recipe.json)."""
    tmp = None
    if src.suffix == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="models_"))
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp)
        src = tmp
    # the folder that plays the role of models/: the parent of the first <evaluation>/ level
    recipes = sorted(src.rglob("recipe.json"))
    if not recipes:
        sys.exit(f"no recipe.json under {src}")
    anchors = [p for p in recipes if p.parent.parent.parent.name in {"pretrain", "als",
                                                                   "mo280_round1"}]
    if not anchors:
        sys.exit("cannot find the models/<evaluation>/<model>/<name>/ layout in the download")
    base = anchors[0].parent.parent.parent.parent
    added = same = kept = 0
    for f in sorted(p for p in base.rglob("*") if p.is_file()):
        rel = f.relative_to(base)
        out = settings.models_dir / rel
        if out.exists():
            if filecmp.cmp(f, out, shallow=False):
                same += 1
                continue
            if out.suffix == ".json":         # our recipe copy vs the one written with the weights
                print(f"  replaced {rel} with the downloaded copy")
            else:
                print(f"  kept the existing {rel}; the download differs")
                kept += 1
                continue
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
        added += 1
    print(f"installed from {src}: {added} new, {same} already here, {kept} kept as they were")
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


def cmd_verify() -> bool:
    """Each model folder: weights present and loading strictly into the recipe's architecture."""
    ok = True
    print(f"{'model folder':64s} {'weights':>9s}  status")
    for d in model_dirs():
        rel = d.relative_to(settings.models_dir).as_posix()
        w = d / "model.pt"
        if not w.exists():
            print(f"{rel:64s} {'-':>9s}  NO WEIGHTS")
            ok = False
            continue
        try:
            model, recipe = infer.load(d)
            n = sum(p.numel() for p in model.parameters())
            print(f"{rel:64s} {w.stat().st_size / 1e6:7.0f}MB  ok "
                  f"({recipe['arch']['name']}, {n / 1e6:.1f}M params, {len(recipe['channels'])}ch)")
        except Exception as e:      # noqa: BLE001 - report every broken folder, not just the first
            print(f"{rel:64s} {w.stat().st_size / 1e6:7.0f}MB  FAILED: {e}")
            ok = False
    return ok


def compare(model_dir: Path, items: list[ds.Item], evaluation: str, method: str) -> dict:
    """Predict `items` with the model here and compare with the saved per-cell probabilities."""
    model, recipe = infer.load(model_dir)
    diff, flips, cells, files = 0.0, 0, 0, 0
    for item, pmap in zip(items, infer.probability_maps(model, recipe, items)):
        saved = infer.result_path(evaluation, method, item.meta)
        if not saved.exists():
            continue
        with np.load(item.path, allow_pickle=False) as z:
            prob = infer.cell_probabilities(pmap, z["cell_bins"])
        with np.load(saved, allow_pickle=False) as z:
            ref = z["prob"]
        if len(ref) != len(prob):
            raise ValueError(f"{saved}: {len(ref)} saved cells, {len(prob)} in the cache")
        if len(prob):
            diff = max(diff, float(np.abs(prob - ref).max()))
        flips += int(((prob > 0.5) != (ref > 0.5)).sum())
        cells += len(prob)
        files += 1
    return {"files": files, "cells": cells, "max_prob_diff": diff, "flips": flips}


def cmd_reproduce(roles: list[str], methods: list[str] | None) -> bool:
    ok = True
    results = settings.data_dir / "_results"
    rows = []
    for evaluation in sorted(p.name for p in results.iterdir() if p.is_dir()):
        for method in sorted(p.name for p in (results / evaluation).iterdir() if p.is_dir()):
            if method in NOT_UNETS or (methods and method not in methods):
                continue
            found = weights_for(evaluation, method)
            name = f"{evaluation}/{method}"
            if found is None:
                rows.append((name, "no model folder here (not one of our U-Nets, or not downloaded)", None))
                continue
            kind, where = found
            t = time.time()
            items = ds.select(evaluation, roles)
            if kind == "shared":
                if not (where / "model.pt").exists():
                    rows.append((name, "NO WEIGHTS", None))
                    ok = False
                    continue
                r = compare(where, items, evaluation, method)
            else:
                r = {"files": 0, "cells": 0, "max_prob_diff": 0.0, "flips": 0}
                missing = []
                for gk in ds.gate_keys_of(items):
                    d = where / cache.gate_id(gk)
                    if not (d / "recipe.json").exists():
                        continue
                    if not (d / "model.pt").exists():
                        missing.append(d.name)
                        continue
                    g = compare(d, [it for it in items if it.meta["gate"] == gk], evaluation,
                                method)
                    r = {k: max(r[k], g[k]) if k == "max_prob_diff" else r[k] + g[k] for k in r}
                if missing:
                    rows.append((name, f"NO WEIGHTS for {len(missing)} gates", None))
                    ok = False
                    if not r["files"]:
                        continue
            if not r["files"]:
                rows.append((name, f"no saved predictions for roles {roles}", None))
                continue
            share = r["flips"] / max(r["cells"], 1)
            good = r["max_prob_diff"] <= PROB_TOL and share <= MAX_FLIPS
            ok &= good
            rows.append((name, "REPRODUCED" if good else "DIFFERS", r | {
                "flip_share": share, "seconds": round(time.time() - t, 1)}))
    print(f"{'evaluation/method':36s} {'result':12s} {'files':>6s} {'cells':>11s} "
          f"{'max |dp|':>9s} {'flipped':>8s} {'time':>6s}")
    for name, status, r in rows:
        if r is None:
            print(f"{name:36s} {status}")
        else:
            print(f"{name:36s} {status:12s} {r['files']:6d} {r['cells']:11,d} "
                  f"{r['max_prob_diff']:9.2e} {r['flip_share']:8.2e} {r['seconds']:5.0f}s")
    return ok


def cmd_register() -> None:
    """models/base/shared <- the pretrained base; lab 'als' <- GPU UNITO + shared fine-tuned."""
    root = settings.models_dir
    src, base = root / "pretrain" / "shared" / "shared", root / "base" / "shared"
    if not (src / "model.pt").exists():
        sys.exit(f"{src} has no weights; install them first")
    if base.exists() and _digest(base / "model.pt") != _digest(src / "model.pt"):
        sys.exit(f"{base} already holds different weights; move it away to replace it")
    shutil.copytree(src, base, dirs_exist_ok=True)
    print(f"base model: {base}")
    gates = {p.name: p for p in sorted((root / "als" / "unito_gpu").iterdir())
             if (p / "model.pt").exists()}
    shared = root / "als" / "shared_finetuned" / "shared"
    if not (shared / "model.pt").exists():
        sys.exit(f"{shared} has no weights; install them first")
    for v in registry.list_versions("als"):
        if v.get("notes", "").startswith("GPU UNITO") and len(v["gates"]) == len(gates):
            print(f"lab als: already registered as {v['version']} (current: {v['current']})")
            return
    version = registry.register_version(
        "als", gates=gates, shared=shared, make_current=True,
        notes=f"GPU UNITO ({len(gates)} gates) + shared fine-tuned; checked by check_models.py")
    print(f"lab als: {version} is current ({len(gates)} per-gate models + shared fine-tuned)")


def als_wells(roles: list[str]) -> dict[str, dict[str, dict]]:
    """individual -> condition -> registry sample, for the ALS individuals with these roles."""
    people = {it.meta["individual"] for it in ds.select("als", roles)}
    reg = json.loads((settings.splits_dir / "datasets_v1.json").read_text())
    out: dict[str, dict[str, dict]] = {}
    for smp in reg["samples"]:
        if smp["individual"] in people and smp["usable"] and smp["dataset"].startswith("als_ics"):
            out.setdefault(smp["individual"], {})[smp["condition"]] = smp
    return out


def expert_tree(smp: dict, keys: list[str]) -> GateTree:
    from flow_er.cytometry.io import FcsFile
    wsp = settings.data_dir / smp["dataset"] / smp["workspace"]
    return from_workspace(wsp, smp["sample"], FcsFile(settings.data_dir / smp["file"]), keys)


def cmd_runtime(lab: str, template: Path) -> None:
    from flow_er.cytometry.gating import workspace_labels
    from flow_er.cytometry.io import FcsFile
    from flow_er.ml.runtime import autogate

    tree = GateTree.from_json(template)
    rows = []                                  # (method, gate, f1, frequency error pp, model)
    for person, wells in sorted(als_wells(["test"]).items()):
        smp, unstim = wells.get("peptide"), wells.get("unstimulated")
        if smp is None:
            continue
        fcs = settings.data_dir / smp["file"]
        labels, _, _ = workspace_labels(FcsFile(fcs), settings.data_dir / smp["dataset"] /
                                        smp["workspace"], smp["sample"], tree.keys, [])
        truth = dict(zip(tree.keys, labels))
        unstim_fcs = settings.data_dir / unstim["file"] if unstim else None
        for method, kw in ((f"lab {lab}", {"lab": lab}), ("template copy", {"use_models": False})):
            t = time.time()
            res = autogate(fcs, tree, unstim=unstim_fcs, **kw)
            secs = time.time() - t
            for k, g in res.gates.items():
                e = truth[k]
                ep = truth[tree[k].parent] if tree[k].parent else np.ones_like(e)
                ap = (res.gates[tree[k].parent].membership if tree[k].parent
                      else np.ones_like(e))
                f1 = 2 * (g.membership & e).sum() / max(g.membership.sum() + e.sum(), 1)
                err = abs(100 * g.membership[ap].mean() - 100 * e[ep].mean())
                rows.append((method, k, f1, err, g.model))
            print(f"  donor {person[-4:]}  {method:14s} {len(res.gates)} gates in {secs:4.1f}s "
                  f"(version {res.version})")
    names = [k.split("/")[-1] for k in tree.keys]
    print(f"\n{'gate':34s} {'model used':30s} {'F1 lab':>7s} {'F1 tmpl':>8s} "
          f"{'err lab':>8s} {'err tmpl':>9s}   mean over test donors; error in pp")
    for k, n in zip(tree.keys, names):
        lab_rows = [r for r in rows if r[1] == k and r[0] != "template copy"]
        tpl_rows = [r for r in rows if r[1] == k and r[0] == "template copy"]
        if not lab_rows:
            continue
        label = n if names.count(n) == 1 else f"{n} [{k.split('/')[-2]}]"
        model = lab_rows[0][4].split("/")[-1][-30:]
        print(f"{label[:34]:34s} {model:30s} {np.mean([r[2] for r in lab_rows]):7.3f} "
              f"{np.mean([r[2] for r in tpl_rows]):8.3f} {np.mean([r[3] for r in lab_rows]):8.2f} "
              f"{np.mean([r[3] for r in tpl_rows]):9.2f}")


CONDITIONS = ("unstimulated", "peptide", "positive control")


def _wells(person: str, wells: dict[str, dict], data_dir: Path) -> list[dict]:
    """A donor's wells in plate order, with absolute paths (read before the folders switch)."""
    out = []
    for cond in CONDITIONS:
        if cond in wells:
            smp = wells[cond]
            out.append({"person": person, "condition": cond, "sample": smp["sample"],
                        "fcs": data_dir / smp["file"],
                        "wsp": data_dir / smp["dataset"] / smp["workspace"]})
    return out


def _gate_name(key: str) -> str:
    parts = key.split("/")
    return f"{parts[-2].split()[0]} {parts[-1]}" if parts[-1].endswith("+") else parts[-1]


def _score_well(well: dict, unstim: Path | None, tree: GateTree, versions: dict[str, str | None],
                lab: str) -> tuple[list[dict], dict]:
    """Gate one well with each version (None = template copy); per gate F1 and % error against the
    expert, plus the densities and polygons for the figures."""
    from flow_er.cytometry.gating import workspace_labels
    from flow_er.cytometry.io import FcsFile
    from flow_er.ml.runtime import autogate

    keys = [g.key for g in tree.gates if not g.qc]
    labels, _, _ = workspace_labels(FcsFile(well["fcs"]), well["wsp"], well["sample"], tree.keys, [])
    truth = dict(zip(tree.keys, labels))
    expert = from_workspace(well["wsp"], well["sample"], FcsFile(well["fcs"]), tree.keys)
    rows, figs = [], {"expert": {k: expert[k].template_array() for k in keys}}
    for name, version in versions.items():
        if version:
            registry.set_current(lab, version)
        res = autogate(well["fcs"], tree, lab=lab if version else None, unstim=unstim,
                       use_models=version is not None)
        figs[name] = {k: (res.gates[k].density, res.gates[k].polygon_display) for k in keys}
        for k in keys:
            g, e, parent = res.gates[k], truth[k], tree[k].parent
            ep = truth[parent] if parent else np.ones_like(e)
            ap = res.gates[parent].membership if parent else np.ones_like(e)
            rows.append({"person": well["person"], "condition": well["condition"], "gate": k,
                         "method": name, "model": g.model,
                         "f1": float(2 * (g.membership & e).sum() / max(g.membership.sum() + e.sum(), 1)),
                         "pct": float(100 * g.membership[ap].mean()) if ap.any() else 0.0,
                         "expert_pct": float(100 * e[ep].mean()) if ep.any() else 0.0,
                         "confidence": g.confidence, "low_confidence": g.low_confidence})
            rows[-1]["error_pp"] = abs(rows[-1]["pct"] - rows[-1]["expert_pct"])
    return rows, figs


def cmd_finetune(lab: str, max_minutes: float, out_dir: Path, folds: list[str]) -> None:
    """Does retraining on a lab's accepted gates improve its gating on donors it never saw?

    Two folds over the 6 locked ALS test donors: teach the 2 fine-tune donors + 3 test donors (the
    expert's gates on all 3 wells, as a scientist keeping each file would), retrain with
    ml.retrain, then gate the other 3 test donors with the lab's current version, the retrained
    one and template copy. Throwaway data/models folders; the lab's real version is untouched."""
    from flow_er.cytometry.io import FcsFile
    from flow_er.ml import finetune
    from flow_er.ml.runtime import autogate

    real_data, real_models = Path(settings.data_dir), Path(settings.models_dir)
    tree = GateTree.from_json(real_models / "labs" / lab / "template.json")
    info = registry.version_info(lab)
    current_gates = {g: registry._abs(d) for g, d in info["gates"].items()}
    teach_always = [w for p, ws in sorted(als_wells(["finetune"]).items()) for w in _wells(p, ws, real_data)]
    test = sorted(als_wells(["test"]).items())
    halves = {"A": (test[:3], test[3:]), "B": (test[3:], test[:3])}
    out_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        taught, held = halves[fold]
        teach = teach_always + [w for p, ws in taught for w in _wells(p, ws, real_data)]
        evaluate = [w for p, ws in held for w in _wells(p, ws, real_data)]
        tmp = Path(tempfile.mkdtemp(prefix="finetune_check_"))
        settings.data_dir, settings.models_dir = tmp / "data", tmp / "models"
        registry._cache.clear()
        (registry.lab_dir("check")).mkdir(parents=True)
        shutil.copy(real_models / "labs" / lab / "template.json", registry.lab_dir("check"))
        before = registry.register_version("check", gates=current_gates, make_current=True,
                                           notes=f"copy of {lab}/{info['version']}")
        print(f"\nfold {fold}: teach {len({w['person'] for w in teach})} donors "
              f"({len(teach)} wells), evaluate on {[p[-4:] for p, _ in held]}; lab copy in {tmp}")
        t = time.time()
        for w in teach:
            unstim_well = next((u for u in teach if u["person"] == w["person"]
                                and u["condition"] == "unstimulated"), None)
            unstim = None
            if unstim_well and unstim_well is not w:     # the app passes the unstimulated well's result
                unstim = autogate(unstim_well["fcs"], from_workspace(
                    unstim_well["wsp"], unstim_well["sample"], FcsFile(unstim_well["fcs"]),
                    tree.keys), use_models=False)
            expert = from_workspace(w["wsp"], w["sample"], FcsFile(w["fcs"]), tree.keys)
            accepted = {g.key: expert[g.key].template_array() for g in tree.gates if not g.qc}
            finetune.add_labels("check", w["fcs"], tree, accepted, w["person"],
                                f"{w['person'].replace(':', '_')}_{w['condition'].replace(' ', '_')}",
                                condition=w["condition"], unstim=unstim)
        print(f"  add_labels: {len(teach)} wells x {len(accepted)} gates in {time.time() - t:.0f}s")
        t = time.time()
        log_file = (out_dir / f"train_fold{fold}.log").open("w", encoding="utf-8")

        def log(line, log_file=log_file):
            log_file.write(str(line).rstrip() + "\n")
            log_file.flush()
            if not str(line).lstrip().startswith("epoch"):
                print("   ", str(line).strip())
        summary = finetune.retrain("check", max_minutes=max_minutes, log=log)
        log_file.close()
        minutes = (time.time() - t) / 60
        print(f"  retrain: {minutes:.1f} min; {summary['version']} promoted {summary['promoted']}: "
              f"{summary['reason']}")
        t = time.time()
        versions = {"current": before, "retrained": summary["version"], "template copy": None}
        rows, figs = [], {}
        for w in evaluate:
            unstim = next((u["fcs"] for u in evaluate if u["person"] == w["person"]
                           and u["condition"] == "unstimulated" and u is not w), None)
            r, f = _score_well(w, unstim, tree, versions, "check")
            rows += r
            figs[f"{w['person'][-4:]} {w['condition']}"] = f
        print(f"  evaluated {len(evaluate)} wells x 3 methods in {time.time() - t:.0f}s")
        _print_scores(rows, tree)
        (out_dir / f"fold{fold}.json").write_text(json.dumps(
            {"fold": fold, "taught": [p for p, _ in taught], "evaluated": [p for p, _ in held],
             "retrain_minutes": minutes, "max_minutes": max_minutes,
             "summary": {k: v for k, v in summary.items()}, "rows": rows}, indent=1, default=float))
        np.save(out_dir / f"fold{fold}_figs.npy", np.array(figs, dtype=object), allow_pickle=True)
        settings.data_dir, settings.models_dir = real_data, real_models
        registry._cache.clear()
        shutil.rmtree(tmp, ignore_errors=True)
    plot_finetune(out_dir, tree)


def _print_scores(rows: list[dict], tree: GateTree) -> None:
    print(f"  {'gate':28s} {'err current':>11s} {'err retrained':>13s} {'err template':>12s} "
          f"{'F1 current':>10s} {'F1 retrained':>12s}   mean over evaluated wells, pp")
    for k in (g.key for g in tree.gates if not g.qc):
        m = {name: [r for r in rows if r["gate"] == k and r["method"] == name]
             for name in ("current", "retrained", "template copy")}
        e = {n: np.mean([r["error_pp"] for r in v]) for n, v in m.items()}
        f = {n: np.mean([r["f1"] for r in v]) for n, v in m.items()}
        print(f"  {_gate_name(k)[:28]:28s} {e['current']:11.2f} {e['retrained']:13.2f} "
              f"{e['template copy']:12.2f} {f['current']:10.3f} {f['retrained']:12.3f}")
    for n in ("current", "retrained", "template copy"):
        print(f"  mean error {n:14s} {np.mean([r['error_pp'] for r in rows if r['method'] == n]):.3f} pp")


def plot_finetune(out_dir: Path, tree: GateTree) -> None:
    """Figures from the folds saved in out_dir: per-gate error, paired wells and gate overlays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folds = sorted(out_dir.glob("fold?.json"))
    rows = [r for f in folds for r in json.loads(f.read_text())["rows"]]
    figs = {}
    for f in folds:
        figs.update(np.load(f.with_name(f.stem + "_figs.npy"), allow_pickle=True).item())
    keys = [g.key for g in tree.gates if not g.qc]
    methods = ("current", "retrained", "template copy")
    colours = {"current": "#3b6fb6", "retrained": "#e07b28", "template copy": "#9a9a9a",
               "expert": "black"}

    def err(k, m, cond=None):
        return [r["error_pp"] for r in rows if r["gate"] == k and r["method"] == m
                and (cond is None or r["condition"] == cond)]

    # 1. per-gate error by well type
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
    y = np.arange(len(keys))
    for ax, cond in zip(axes, CONDITIONS):
        for i, m in enumerate(methods):
            ax.barh(y + (i - 1) * 0.27, [np.mean(err(k, m, cond)) for k in keys], 0.27,
                    color=colours[m], label=m)
        ax.set_title(f"{cond} wells (n={len(err(keys[0], 'current', cond))})")
        ax.set_xlabel("|model % - expert %| of parent, pp (mean)")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y, [_gate_name(k) for k in keys])
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right")
    fig.suptitle("Frequency error on held-out test donors before and after retraining")
    fig.tight_layout()
    fig.savefig(out_dir / "1_error_per_gate.png", dpi=130)
    plt.close(fig)

    # 2. paired: every (well, gate), current vs retrained
    fig, ax = plt.subplots(figsize=(7, 7))
    for cond, mk in zip(CONDITIONS, "o^s"):
        a = [r for r in rows if r["method"] == "current" and r["condition"] == cond]
        b = {(r["person"], r["gate"]): r["error_pp"] for r in rows
             if r["method"] == "retrained" and r["condition"] == cond}
        ax.scatter([r["error_pp"] + 0.01 for r in a], [b[(r["person"], r["gate"])] + 0.01 for r in a],
                   marker=mk, alpha=0.6, label=cond)
    lim = [0.01, max(r["error_pp"] for r in rows) * 1.5 + 0.01]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("error with current models (pp, +0.01)")
    ax.set_ylabel("error after retraining (pp, +0.01)")
    ax.set_title("Each well x gate: below the line = retraining helped")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "2_paired_wells.png", dpi=130)
    plt.close(fig)

    # 3. per gate, the well where retraining changed the error most: density + the three polygons
    def overlay(ax, well, k, title):
        f = figs[well]
        dens = f["retrained"][k][0]
        ax.imshow(np.log1p(dens), origin="lower", extent=(0, 1, 0, 1), cmap="Greys", aspect="auto")
        for name in ("expert", "current", "retrained"):
            poly = f[name][k] if name == "expert" else f[name][k][1]
            if poly is not None and len(poly) >= 3:
                p = np.vstack([poly, poly[:1]])
                ax.plot(p[:, 0], p[:, 1], color=colours[name], lw=1.6 if name == "expert" else 1.3,
                        ls="--" if name == "expert" else "-")
        ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.set_xticks([]), ax.set_yticks([])
        ax.set_xlabel(tree[k].x, fontsize=7, labelpad=1)
        ax.set_ylabel(tree[k].y, fontsize=7, labelpad=1)
        ax.set_title(title, fontsize=8)

    def gallery(name, pick):
        cols = 5
        fig, axes = plt.subplots(int(np.ceil(len(keys) / cols)), cols, figsize=(3.4 * cols, 3.6 * 3))
        for ax, k in zip(axes.flat, keys):
            pairs = [(r, next(s for s in rows if s["method"] == "retrained" and s["gate"] == k
                              and s["person"] == r["person"] and s["condition"] == r["condition"]))
                     for r in rows if r["method"] == "current" and r["gate"] == k]
            r, s = pick(pairs)
            well = f"{r['person'][-4:]} {r['condition']}"
            overlay(ax, well, k, f"{_gate_name(k)}\n{well}: expert {r['expert_pct']:.2f}%\n"
                                 f"current {r['pct']:.2f}%  retrained {s['pct']:.2f}%")
        for ax in list(axes.flat)[len(keys):]:
            ax.axis("off")
        handles = [plt.Line2D([], [], color=colours[n], ls="--" if n == "expert" else "-", label=n)
                   for n in ("expert", "current", "retrained")]
        fig.legend(handles=handles, loc="lower right", fontsize=10)
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(out_dir / f"3_{name.split(':')[0].replace(' ', '_')}.png", dpi=120)
        plt.close(fig)

    gallery("gates_most_improved: per gate, the held-out well retraining helped most",
            lambda p: max(p, key=lambda rs: rs[0]["error_pp"] - rs[1]["error_pp"]))
    gallery("gates_most_worsened: per gate, the held-out well retraining hurt most",
            lambda p: min(p, key=lambda rs: rs[0]["error_pp"] - rs[1]["error_pp"]))

    # 4. one well, the full cascade: the positive control with the largest current error
    worst = max((r for r in rows if r["method"] == "current" and r["condition"] == "positive control"),
                key=lambda r: r["error_pp"])
    well = f"{worst['person'][-4:]} positive control"
    fig, axes = plt.subplots(3, 5, figsize=(17, 11))
    for ax, k in zip(axes.flat, keys):
        rs = {m: next(r for r in rows if r["method"] == m and r["gate"] == k and
                      f"{r['person'][-4:]} {r['condition']}" == well) for m in methods}
        overlay(ax, well, k, f"{_gate_name(k)}: expert {rs['current']['expert_pct']:.2f}%\n"
                             f"current {rs['current']['pct']:.2f}%  retrained {rs['retrained']['pct']:.2f}%")
    fig.legend(handles=[plt.Line2D([], [], color=colours[n], ls="--" if n == "expert" else "-", label=n)
                        for n in ("expert", "current", "retrained")], loc="lower right")
    fig.suptitle(f"Donor {well}: every gate, current vs retrained vs expert")
    fig.tight_layout()
    fig.savefig(out_dir / "4_cascade_one_well.png", dpi=120)
    plt.close(fig)
    print(f"figures in {out_dir}")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("install").add_argument("src", type=Path)
    sub.add_parser("verify")
    r = sub.add_parser("reproduce")
    r.add_argument("--roles", nargs="+", default=["test"])
    r.add_argument("--methods", nargs="*")
    sub.add_parser("register")
    rt = sub.add_parser("runtime")
    rt.add_argument("--lab", default="als")
    rt.add_argument("--template", type=Path, help="GateTree JSON (default: data/_demo/"
                    "template_2008.json, written by scripts/autogate.py --save-template)")
    ft = sub.add_parser("finetune")
    ft.add_argument("--lab", default="als")
    ft.add_argument("--max-minutes", type=float, default=6, help="training cap per gate")
    ft.add_argument("--out", type=Path, required=True, help="folder for the results and figures")
    ft.add_argument("--folds", nargs="+", default=["A", "B"], choices=["A", "B"])
    ft.add_argument("--plot-only", action="store_true")
    a = p.parse_args()
    if a.cmd == "install":
        cmd_install(a.src)
    elif a.cmd == "verify":
        sys.exit(0 if cmd_verify() else 1)
    elif a.cmd == "reproduce":
        sys.exit(0 if cmd_reproduce(a.roles, a.methods) else 1)
    elif a.cmd == "register":
        cmd_register()
    elif a.cmd == "runtime":
        cmd_runtime(a.lab, a.template or settings.data_dir / "_demo" / "template_2008.json")
    elif a.plot_only:
        plot_finetune(a.out, GateTree.from_json(settings.models_dir / "labs" / a.lab / "template.json"))
    else:
        cmd_finetune(a.lab, a.max_minutes, a.out, a.folds)


if __name__ == "__main__":
    main()
