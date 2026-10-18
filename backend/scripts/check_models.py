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
    # 6. fine-tune a throwaway lab from the base model on screen-4 donors (temporary folders)
    uv run python scripts/check_models.py finetune [--base <model dir>] [--max-minutes 15]

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


def cmd_finetune(base: Path, donors: int, max_minutes: float) -> None:
    from flow_er.ml import finetune
    from flow_er.ml.runtime import autogate

    if not (base / "model.pt").exists():
        sys.exit(f"{base} has no weights")
    tree = GateTree.from_json(settings.data_dir / "_demo" / "template_2008.json")
    wells = {p: w for p, w in als_wells(["finetune", "test"]).items() if "peptide" in w}
    labelled = []
    for person in sorted(wells)[:donors]:     # expert gates, read before switching folders
        w = wells[person]
        unstim = None
        if "unstimulated" in w:
            u = w["unstimulated"]
            unstim = autogate(settings.data_dir / u["file"], expert_tree(u, tree.keys),
                              use_models=False)
        e = expert_tree(w["peptide"], tree.keys)
        labelled.append((person, settings.data_dir / w["peptide"]["file"],
                         {k: e[k].template_array() for k in tree.keys}, unstim))
    tmp = Path(tempfile.mkdtemp(prefix="finetune_check_"))
    settings.data_dir, settings.models_dir = tmp / "data", tmp / "models"
    shutil.copytree(base, settings.models_dir / "base" / "shared")
    print(f"temporary lab in {tmp}; base model {base}")
    t = time.time()
    for person, fcs, accepted, unstim in labelled:
        finetune.add_labels("check", fcs, tree, accepted, person, fcs.stem, unstim=unstim)
    print(f"add_labels: {len(labelled)} donors x {len(tree.keys)} gates in {time.time() - t:.0f}s")
    t = time.time()
    out = finetune.retrain("check", max_minutes=max_minutes,
                           log=lambda m: print("   ", m.strip()) if "best" in m else None)
    print(f"retrain: {(time.time() - t) / 60:.1f} min; version {out['version']}, promoted "
          f"{out['promoted']}: {out['reason']}")
    before, after = out["metrics"]["current"], out["metrics"]["candidate"]
    print(f"{'gate (held-out donors)':40s} {'base model':>11s} {'fine-tuned':>11s}")
    for g in sorted(after):
        name = "/".join(g.split("/")[-2:]) if g.split("/")[-1].endswith("+") else g.split("/")[-1]
        print(f"{name[:40]:40s} {before[g]:9.2f}pp {after[g]:9.2f}pp")
    print(f"{'mean':40s} {np.mean(list(before.values())):9.2f}pp "
          f"{np.mean(list(after.values())):9.2f}pp")
    shutil.rmtree(tmp, ignore_errors=True)


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
    ft.add_argument("--base", type=Path, help="starting model (default: models/base/shared)")
    ft.add_argument("--donors", type=int, default=8)
    ft.add_argument("--max-minutes", type=float, default=15)
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
    else:
        cmd_finetune(a.base or settings.models_dir / "base" / "shared", a.donors, a.max_minutes)


if __name__ == "__main__":
    main()
