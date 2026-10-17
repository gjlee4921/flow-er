"""Train/validation/test splits by person or mouse across all datasets, never by cell.

The registry (datasets_v1.json, written by scripts/build_registry.py) says who each sample is;
the split (split_v1.json) says what each sample is used for. Both live in settings.splits_dir,
outside git. validate() fails if any individual is on both the training side and the held-out
side of an evaluation, or if a control, unusable or excluded sample is used anywhere.

    uv run python -m flow_er.ml.splits freeze   # write split_v1.json from the rules below
    uv run python -m flow_er.ml.splits check    # validate it, including every used file's MD5
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

from flow_er.config import settings

REGISTRY = "datasets_v1.json"
SPLIT = "split_v1.json"

# --- the rules ----------------------------------------------------------------------------------

# ALS benchmark: ICS screens 3/4, split by donor (approved 6/2/2/6, HC/ALS balanced).
ALS_DATASETS = ("als_ics_screen3", "als_ics_screen4")
ALS_ROLES = {
    "train": ["2008", "2238", "3551", "7182", "7183", "7184"],
    "val": ["4191", "7207"],
    "finetune": ["3417", "6936"],
    "test": ["3740", "4595", "6934", "7181", "8710", "8830"],
}
# Stretch: 4-fold CV by donor; fold k = the k-th donor of each plate (2 HC + 2 ALS, 2 per screen).
# Validation = the screen-3 HC and ALS donors of the next fold. Run only after the locked test.
ALS_CV_FOLDS = [["2008", "7182", "3417", "6936"], ["2238", "7183", "3740", "7181"],
                ["3551", "7184", "4595", "8710"], ["4191", "7207", "6934", "8830"]]

# MO280 unseen tissue: halves balanced by diet group (5 WD + 3 CD mice each).
MO280_HALVES = {"A": [1, 3, 5, 7, 9, 11, 13, 15], "B": [2, 4, 6, 8, 10, 12, 14, 16]}
MO280_VAL = {"A": [7, 15], "B": [10, 16]}       # one CD and one WD mouse of the training half
MO280_TRAIN_TISSUES, MO280_TEST_TISSUES = ("blood", "spleen"), ("liver", "aorta")
# A shared gate "changes between tissues" when at least this share of test-tissue samples use a
# gate version that no blood or spleen sample uses.
MO280_CHANGED_SHARE = 0.5

# Shared pretraining pool: never scored. About 10% of its individuals stop training early.
POOL_DATASETS = ("figshare_spleen", "macrophages_peritoneal", "als_spectral")
POOL_VAL = {"mouse:figshare_spleen:young 5", "mouse:figshare_spleen:young 6",
            "?mouse:macrophages_peritoneal:SM_PM_2:C", "?mouse:macrophages_peritoneal:SM_PM_4:A",
            "human:3553", "human:8075"}

# Used nowhere (decisions of 2026-10-17).
EXCLUDED = {
    "datasets": {"als_ics_screen1": "donor identity unknown (may be benchmark donors)",
                 "als_ics_screen2": "donor identity unknown (may be benchmark donors)"},
    "cohorts": {"MO CVR Aorta (Aurora)": "2 of 28 gates agree with FlowJo; dropped everywhere"},
    "individuals_in": {("als_spectral", "human:6936"): "6936 is an ALS fine-tune/CV donor"},
}


# --- registry -----------------------------------------------------------------------------------


def key(s: dict) -> str:
    return f"{s['dataset']}/{s['workspace']}/{s['sample']}"


def load_registry(splits_dir: Path | None = None) -> dict:
    return json.loads(((splits_dir or settings.splits_dir) / REGISTRY).read_text())


def load_split(splits_dir: Path | None = None, check: bool = True) -> dict:
    """The frozen split; validated against the registry unless check=False."""
    d = splits_dir or settings.splits_dir
    split = json.loads((d / SPLIT).read_text())
    if check:
        validate(split, load_registry(d))
    return split


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def excluded_reason(s: dict) -> str:
    if s["control"]:
        return f"control ({s['control_reason']})"
    if not s["usable"]:
        return f"unusable ({s['notes']})"
    if s["dataset"] in EXCLUDED["datasets"]:
        return EXCLUDED["datasets"][s["dataset"]]
    if s["cohort"] in EXCLUDED["cohorts"]:
        return EXCLUDED["cohorts"][s["cohort"]]
    return EXCLUDED["individuals_in"].get((s["dataset"], s["individual"]), "")


# --- building the split ---------------------------------------------------------------------------


def _mo280_round(samples: list[dict], train_half: str, test_half: str) -> dict:
    def pick(mice, tissues):
        ids = {f"mouse:aorta_olfr2:MO280-{n}" for n in mice}
        return sorted(key(s) for s in samples
                      if s["individual"] in ids and s["tissue"] in tissues)

    val = MO280_VAL[train_half]
    train = [n for n in MO280_HALVES[train_half] if n not in val]
    return {"stages": {"train": pick(train, MO280_TRAIN_TISSUES),
                       "val": pick(val, MO280_TRAIN_TISSUES),
                       "test": pick(MO280_HALVES[test_half], MO280_TEST_TISSUES)},
            "train_mice": train, "val_mice": val, "test_mice": MO280_HALVES[test_half]}


def _mo280_gates(samples: list[dict], versions_csv: Path) -> dict:
    """Gates present in both the training and the test tissues, split into the ones whose expert
    gate changes between tissues (headline) and the ones that are one template everywhere."""
    import pandas as pd

    by_tissue = defaultdict(set)
    for s in samples:
        by_tissue[s["tissue"]] |= set(s["usable_gates"])
    shared = (set().union(*(by_tissue[t] for t in MO280_TRAIN_TISSUES))
              & set().union(*(by_tissue[t] for t in MO280_TEST_TISSUES)))
    v = pd.read_csv(versions_csv)
    v = v[v["sample"].isin({s["sample"] for s in samples})]
    v["tissue"] = v["sample"].str.split("_").str[0].str.lower()
    changed = []
    for gate in shared:
        d = v[v["gate"] == gate]
        seen = set(d.loc[d["tissue"].isin(MO280_TRAIN_TISSUES), "version"])
        test = d[d["tissue"].isin(MO280_TEST_TISSUES)]
        if len(test) and (~test["version"].isin(seen)).mean() >= MO280_CHANGED_SHARE:
            changed.append(gate)
    return {"headline_changes_between_tissues": sorted(changed),
            "template_copy_is_perfect_here": sorted(shared - set(changed))}


def build_split(registry: dict, inspect_dir: Path) -> dict:
    samples = [s for s in registry["samples"] if not excluded_reason(s)]

    def by_individual(datasets, individuals):
        ids = set(individuals)
        return sorted(key(s) for s in samples if s["dataset"] in datasets and s["individual"] in ids)

    def donors(ids):
        return [f"human:{d}" for d in ids]

    als = {"datasets": list(ALS_DATASETS),
           "individuals": {role: donors(ids) for role, ids in ALS_ROLES.items()},
           "stages": {role: by_individual(ALS_DATASETS, donors(ids))
                      for role, ids in ALS_ROLES.items()},
           "scored_gates": "every usable gate except the Time QC gate; FlowKit memberships are "
                           "the truth, FlowJo counts are reported as the noise floor"}
    cv = []
    for k, fold in enumerate(ALS_CV_FOLDS):
        nxt = ALS_CV_FOLDS[(k + 1) % len(ALS_CV_FOLDS)]
        val = [nxt[0], nxt[1]]                     # screen-3 HC and ALS donors of the next fold
        train = [d for f in ALS_CV_FOLDS for d in f if d not in fold and d not in val]
        cv.append({"stages": {"train": by_individual(ALS_DATASETS, donors(train)),
                              "val": by_individual(ALS_DATASETS, donors(val)),
                              "test": by_individual(ALS_DATASETS, donors(fold))},
                   "test_donors": fold, "val_donors": val})

    mo280 = [s for s in samples if s["dataset"] == "aorta_olfr2"
             and s["cohort"].startswith("MO280")]
    gates = _mo280_gates(mo280, inspect_dir / "aorta_olfr2" / "27-Feb-2023.gate_versions.csv")
    rounds = {name: {**_mo280_round(mo280, a, b), "scored_gates": gates}
              for name, (a, b) in {"mo280_round1": ("A", "B"), "mo280_round2": ("B", "A")}.items()}
    rounds["mo280_round2"]["stretch"] = True

    pool = [s for s in samples if s["dataset"] in POOL_DATASETS]
    excluded = defaultdict(list)
    for s in registry["samples"]:
        if why := excluded_reason(s):
            excluded[why].append(key(s))

    return {
        "version": "split_v1", "created": date.today().isoformat(), "registry": REGISTRY,
        "registry_sha256": None,  # filled in by freeze()
        "unit": "individual (person or mouse); every sample of an individual has one role per "
                "evaluation, and no individual is on the training side (pretraining pool, "
                "train, fine-tune) and the held-out side (val, test) of any evaluation",
        "pretrain": {
            "stages": {"train": sorted(key(s) for s in pool if s["individual"] not in POOL_VAL),
                       "val": sorted(key(s) for s in pool if s["individual"] in POOL_VAL)},
            "gates": "gates with agrees_with_flowjo true, Time QC gates excluded; each dataset's "
                     "gates are separate tasks; datasets weighted equally per epoch",
            "never_scored": True},
        "evaluations": {
            "als": als,
            "als_cv": {"stretch": True, "run_after": "the locked ALS test result is recorded",
                       "folds": cv},
            **rounds},
        "excluded": {why: sorted(keys) for why, keys in excluded.items()},
    }


def freeze(splits_dir: Path | None = None, inspect_dir: Path | None = None) -> Path:
    d = splits_dir or settings.splits_dir
    registry = load_registry(d)
    split = build_split(registry, inspect_dir or settings.data_dir / "_inspect")
    split["registry_sha256"] = sha256(d / REGISTRY)
    validate(split, registry)
    out = d / SPLIT
    if out.exists():
        raise FileExistsError(f"{out} is frozen; write a new version instead of overwriting")
    out.write_text(json.dumps(split, indent=1))
    return out


# --- checks ---------------------------------------------------------------------------------------


class SplitError(AssertionError):
    pass


TRAIN_SIDE, HELD_OUT = ("train", "finetune"), ("val", "test")


def _evaluations(split: dict):
    """(name, stages) for every evaluation, each CV fold counted as its own evaluation."""
    for name, e in split["evaluations"].items():
        if "folds" in e:
            for k, fold in enumerate(e["folds"]):
                yield f"{name}[{k}]", fold["stages"]
        else:
            yield name, e["stages"]


def validate(split: dict, registry: dict, splits_dir: Path | None = None) -> None:
    """Raise SplitError on the first rule a split breaks."""
    by_key = {key(s): s for s in registry["samples"]}
    if splits_dir is not None and split.get("registry_sha256") != sha256(splits_dir / REGISTRY):
        raise SplitError("registry changed since the split was frozen")

    def samples(keys):
        missing = [k for k in keys if k not in by_key]
        if missing:
            raise SplitError(f"not in the registry: {missing[:3]}")
        return [by_key[k] for k in keys]

    pool = {role: samples(keys) for role, keys in split["pretrain"]["stages"].items()}
    pool_individuals = {s["individual"] for ss in pool.values() for s in ss}
    if {s["individual"] for s in pool["train"]} & {s["individual"] for s in pool["val"]}:
        raise SplitError("pretraining pool: an individual is in both its train and val parts")

    used = [s for ss in pool.values() for s in ss]
    for name, stages in _evaluations(split):
        roles = {role: samples(keys) for role, keys in stages.items()}
        used += [s for ss in roles.values() for s in ss]
        ind = {role: {s["individual"] for s in ss} for role, ss in roles.items()}
        train_side = pool_individuals.union(*(ind.get(r, set()) for r in TRAIN_SIDE))
        for r in HELD_OUT:
            if leak := train_side & ind.get(r, set()):
                raise SplitError(f"{name}: {sorted(leak)[:3]} on the training side and in {r}")
        if both := ind.get("val", set()) & ind.get("test", set()):
            raise SplitError(f"{name}: {sorted(both)[:3]} in both val and test")
        for r in HELD_OUT:
            if unknown := [i for i in ind.get(r, set()) if i.startswith("?") or not i]:
                raise SplitError(f"{name}: individuals of unknown identity in {r}: {unknown[:3]}")
        # Every used sample of an individual has one role (e.g. a donor's three wells together).
        seen = defaultdict(set)
        for role, ss in roles.items():
            for s in ss:
                seen[(s["dataset"], s["individual"])].add(role)
        if split_up := [k for k, rs in seen.items() if len(rs) > 1]:
            raise SplitError(f"{name}: individual split across roles: {split_up[:3]}")
        # The same file content must not sit on both sides (e.g. a renamed copy).
        sides = defaultdict(set)
        for role, ss in roles.items():
            for s in ss:
                sides[s["md5"]].add("held_out" if role in HELD_OUT else "train")
        for s in (s for ss in pool.values() for s in ss):
            sides[s["md5"]].add("train")
        if dup := [m for m, sd in sides.items() if len(sd) > 1]:
            raise SplitError(f"{name}: identical file content on both sides: {dup[:3]}")

    for s in used:
        if why := excluded_reason(s):
            raise SplitError(f"{key(s)} is used but excluded: {why}")


def verify_files(split: dict, registry: dict, data_dir: Path | None = None,
                 max_mb: float | None = None) -> int:
    """Check that every used file exists and has its registry MD5; returns files checked.
    max_mb skips the MD5 of larger files (existence is still checked)."""
    root = data_dir or settings.data_dir
    by_key = {key(s): s for s in registry["samples"]}
    all_stages = [split["pretrain"]["stages"]] + [st for _, st in _evaluations(split)]
    keys = {k for stages in all_stages for keys in stages.values() for k in keys}
    checked = 0
    for k in sorted(keys):
        s = by_key[k]
        path = root / s["file"]
        if not path.is_file():
            raise SplitError(f"missing file for {k}: {path}")
        if max_mb is None or path.stat().st_size <= max_mb * 2**20:
            h = hashlib.md5()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(2**20), b""):
                    h.update(chunk)
            if h.hexdigest() != s["md5"]:
                raise SplitError(f"MD5 mismatch for {k}")
            checked += 1
    return checked


def summary(split: dict, registry: dict) -> str:
    by_key = {key(s): s for s in registry["samples"]}
    lines = []

    def line(name, stages):
        parts = []
        for role, keys in stages.items():
            ss = [by_key[k] for k in keys]
            parts.append(f"{role} {len(ss)} samples/{len({s['individual'] for s in ss})} ind.")
        lines.append(f"  {name:16s} " + " | ".join(parts))

    line("pretrain", split["pretrain"]["stages"])
    for name, stages in _evaluations(split):
        line(name, stages)
    g = split["evaluations"]["mo280_round1"]["scored_gates"]
    lines.append(f"  MO280 scored gates: {len(g['headline_changes_between_tissues'])} headline + "
                 f"{len(g['template_copy_is_perfect_here'])} template-copy-is-perfect")
    lines.append("  excluded: " + "; ".join(f"{len(v)} ({w[:50]})" for w, v in
                                            sorted(split["excluded"].items(), key=lambda x: -len(x[1]))))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["freeze", "check"])
    parser.add_argument("--max-mb", type=float, default=None,
                        help="check: skip the MD5 of files larger than this")
    args = parser.parse_args()
    if args.command == "freeze":
        print(f"wrote {freeze()}")
    split, registry = load_split(check=False), load_registry()
    validate(split, registry, settings.splits_dir)
    print("split is valid:\n" + summary(split, registry))
    if args.command == "check":
        print(f"MD5 verified for {verify_files(split, registry, max_mb=args.max_mb)} files")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
