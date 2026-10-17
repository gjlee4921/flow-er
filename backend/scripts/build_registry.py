"""inspect_wsp.py outputs -> backend/splits/datasets_v1.json, the registry every split reads.

One row per workspace sample: dataset, file, MD5, species, tissue, individual (person or mouse),
how sure that identity is, condition, control flag and usable gates; plus per-dataset gate
summaries with FlowKit-vs-FlowJo agreement. Identity rules are per dataset and documented below.

    uv run python scripts/build_registry.py   # reads <data>/_inspect, writes <data>/_splits
"""

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from flow_er.config import settings

DATA = settings.data_dir
OUT = settings.splits_dir / "datasets_v1.json"

DATASETS = {
    "figshare_spleen": {
        "species": "mouse", "tissue": "spleen", "source": "Figshare project 210475",
        "license": "CC BY 4.0", "instrument": "BD FACSymphony A5 SE (header: SE (LSRFortessa))",
        "identity_basis": "one FCS file = one mouse (young 1-6, Old 1, 3-6; Old 2 has no file)"},
    "als_ics_screen1": {
        "species": "human", "tissue": "PBMC", "source": "Zenodo (ALS ICS study, screen 1)",
        "license": "CC BY 4.0", "instrument": "BD LSRFortessa X-20",
        "identity_basis": "workspace groups 'Donor 1-4' only; real donor IDs unknown, so these "
                          "donors may be the same people as named donors elsewhere"},
    "als_ics_screen2": {
        "species": "human", "tissue": "PBMC", "source": "Zenodo (ALS ICS study, screen 2)",
        "license": "CC BY 4.0", "instrument": "BD LSRFortessa X-20",
        "identity_basis": "workspace groups 'Donor 1-6 (ALS/HC)' only; real donor IDs unknown"},
    "als_ics_screen3": {
        "species": "human", "tissue": "PBMC", "source": "Zenodo 16754591",
        "license": "CC BY 4.0", "instrument": "BD LSRFortessa X-20",
        "identity_basis": "4-digit workspace group owning every gate of wells C1-3/C4-6/C7-9/C10-12"},
    "als_ics_screen4": {
        "species": "human", "tissue": "PBMC", "source": "Zenodo 16754609",
        "license": "CC BY 4.0", "instrument": "BD LSRFortessa X-20",
        "identity_basis": "4-digit workspace group owning every gate of wells C1-3/C4-6/C7-9/C10-12"},
    "als_spectral": {
        "species": "human", "tissue": "PBMC", "source": "Zenodo (ALS spectral panel)",
        "license": "CC BY 4.0", "instrument": "spectral (see inspection)",
        "identity_basis": "4-digit donor ID in the sample name, same numbering as the ICS groups"},
    "aorta_olfr2": {
        "species": "mouse", "tissue": "aorta/liver/spleen/blood", "source": "Figshare 32996177",
        "license": "CC BY 4.0", "instrument": "BD LSRFortessa (MO280) and Cytek Aurora (MO CVR)",
        "identity_basis": "MO280: Tissue_N with TUBE NAME N; the lab's Analysis.xlsx gives tube N "
                          "the same diet group in all four tissues, so N is taken to be the mouse "
                          "(inferred, not stated). MO CVR files: mouse unknown"},
    "macrophages_peritoneal": {
        "species": "mouse", "tissue": "peritoneal macrophages", "source": "Figshare 12546686",
        "license": "CC0", "instrument": "BD FACSDiva cytometer",
        "identity_basis": "specimens labelled only A/C per experiment (SM_PM_k); mice unknown"},
}

# A gate "agrees with FlowJo" when, over wells where FlowJo counted >= 100 events, FlowKit's count
# is within this median and worst-case relative difference. Pretraining uses agreeing gates only.
AGREE_MEDIAN_PCT, AGREE_MAX_PCT = 1.0, 5.0

# Workspaces whose labels FlowKit cannot reproduce: samples stay registered (so overlap checks see
# the individuals) but are not usable.
UNRELIABLE_LABELS = {
    ("als_spectral", "#4 Spectral.wsp"): "FlowKit 10-36% below FlowJo from cd45+ down "
                                         "(biex top 100000 below the Aurora data range); "
                                         "excluded until fixed",
    ("als_spectral", "#5 Spectral.wsp"): "FlowKit 10-36% below FlowJo from cd45+ down "
                                         "(biex top 100000 below the Aurora data range); "
                                         "excluded until fixed",
}

# Single samples that cannot be used, with the reason.
UNUSABLE_SAMPLES = {
    ("macrophages_peritoneal", "SM_PM_2.wsp", "Specimen_001_C_unstained_012.fcs"):
        "Figshare holds Specimen_001_C_unstained_012.fcs and ..._C_Unstained_012.fcs, which are "
        "one path on Windows; the file on disk is the other one. Control, not re-downloaded",
}

# MO280 diet group per tube N, from the lab's Analysis.xlsx (identical for all four tissues).
MO280_DIET = {**{n: "WD 4W" for n in range(1, 5)}, **{n: "CD 4W" for n in range(5, 8)},
              **{n: "CD 2W" for n in range(8, 11)}, **{n: "WD 2W" for n in range(11, 14)},
              **{n: "WD 10W" for n in range(14, 17)}}
ICS_CONDITION = {"negative": "unstimulated", "negatice": "unstimulated", "peptide": "peptide",
                 "positive": "positive control"}


def ics_condition(text: str) -> str:
    m = re.search(r"(?i)(negati[cv]e|peptide|positive)", text)
    return ICS_CONDITION[m.group(1).lower()] if m else ""


def identify(dataset: str, wsp: str, s: pd.Series) -> dict:
    """species, tissue, individual, identity ('confirmed'|'inferred'|'unknown'), cohort,
    condition for one sample. An individual starting with '?' has no reliable identity."""
    name, file, groups = s["sample"], str(s["file"]), str(s["groups"])
    out = {"species": DATASETS[dataset]["species"], "tissue": DATASETS[dataset]["tissue"],
           "individual": "", "identity": "unknown", "cohort": "", "condition": ""}
    if dataset == "figshare_spleen":
        mouse = name.removeprefix("Spleen samples_").removesuffix(".fcs")
        out.update(individual=f"mouse:figshare_spleen:{mouse}", identity="confirmed",
                   cohort="aged" if mouse.lower().startswith("old") else "young")
    elif dataset in ("als_ics_screen3", "als_ics_screen4"):
        donor = re.match(r"(\d{4})", groups)
        out.update(individual=f"human:{donor.group(1)}" if donor else "", identity="confirmed",
                   cohort="ALS" if "ALS" in wsp else "HC", condition=ics_condition(file))
    elif dataset == "als_ics_screen1":
        cohort = "ALS" if wsp.startswith("ALS") else "HC"
        donor = re.search(r"Donor \d+", groups)
        out.update(individual=f"?human:als_ics_screen1:{cohort}:{donor.group(0) if donor else name}",
                   cohort=cohort, condition=ics_condition(name))
    elif dataset == "als_ics_screen2":
        donor = re.search(r"Donor \d+", groups)
        out.update(individual=f"?human:als_ics_screen2:{donor.group(0) if donor else name}",
                   cohort="ALS" if re.search(r"\bALS\b", groups) else "HC",
                   condition=ics_condition(name))
    elif dataset == "als_spectral":
        m = re.search(r"\b(\d{4})\b(?:\s+(ALS|HC))?", name)
        out.update(individual=f"human:{m.group(1)}" if m else f"?human:als_spectral:{name}",
                   identity="confirmed" if m else "unknown", cohort=(m.group(2) or "") if m else "",
                   condition="phenotyping")
    elif dataset == "aorta_olfr2":
        m = re.match(r"(Aorta|Blood|Liver|Spleen)_(\d+)\.fcs$", name)
        if m:
            n = int(m.group(2))
            out.update(tissue=m.group(1).lower(), individual=f"mouse:aorta_olfr2:MO280-{n}",
                       identity="inferred", cohort="MO280 diet time course",
                       condition=MO280_DIET.get(n, ""))
        elif str(s.get("$CYT", "")).lower().startswith("aurora"):
            # MO CVR Aorta (Cytek Aurora): SpectroFlo GROUPNAME/settings say aorta; mice unknown.
            out.update(tissue="aorta", individual=f"?mouse:aorta_olfr2:CVR:{s['md5'][:8]}",
                       cohort="MO CVR Aorta (Aurora)",
                       condition="Olfr2 KO" if re.search(r"(?i)\bko\b", name) else "")
        else:
            # MO280 compensation controls and spleen FMOs (controls, never used for training).
            out.update(tissue="spleen" if name.lower().startswith("spleen") else "",
                       individual=f"?mouse:aorta_olfr2:MO280-control:{s['md5'][:8]}",
                       cohort="MO280 diet time course")
    elif dataset == "macrophages_peritoneal":
        m = re.match(r"Specimen_001_([AC])_", name)
        out.update(individual=f"?mouse:macrophages_peritoneal:{wsp}:{m.group(1) if m else 'all'}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inspect", type=Path, default=DATA / "_inspect")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    samples, datasets, unregistered = [], {}, {}
    for dataset, meta in DATASETS.items():
        folder = args.inspect / dataset
        if not folder.is_dir():
            print(f"{dataset}: no inspection output in {folder}, skipped")
            continue
        gate_rows, used_files = [], set()
        # A download manifest (name, path, md5) is the authoritative file list where present;
        # files left over from an earlier download are not part of the dataset.
        manifest_path = DATA / dataset / "manifest.json"
        manifest = {e["path"].replace("\\", "/"): e["md5"]
                    for e in json.loads(manifest_path.read_text())} if manifest_path.exists() else {}
        for sfile in sorted(folder.glob("*.samples.csv")):
            stem = sfile.name.removesuffix(".samples.csv")
            info = json.loads((folder / f"{stem}.json").read_text())
            wsp = Path(info["wsp"]).name
            sdf = pd.read_csv(sfile, dtype=str).fillna("")
            gates = pd.read_csv(folder / f"{stem}.gates.csv", dtype=str).fillna("")
            counts = pd.read_csv(folder / f"{stem}.counts.csv")
            counts["rel"] = 100 * (counts["flowkit"] - counts["flowjo"]).abs() / counts[
                "flowjo"].clip(lower=1)
            counts["big"] = counts["flowjo"] >= 100
            time_gates = set(gates.loc[(gates["x"] == "Time") | (gates["y"] == "Time"),
                                       "gate_key"])
            for _, g in gates.iterrows():
                c = counts[counts["gate"] == g["gate_key"]].dropna(subset=["flowkit"])
                big = c.loc[c["big"], "rel"]
                med, worst = (round(float(big.median()), 2), round(float(big.max()), 2)) \
                    if len(big) else (None, None)
                gate_rows.append({
                    "workspace": wsp, "gate": g["gate_key"], "type": g["type"],
                    "x": g["x"], "y": g["y"], "kind": g["kind"], "versions": int(g["versions"]),
                    "samples": int(g["samples"]), "qc_time_gate": g["gate_key"] in time_gates,
                    "checked": len(c), "exact": int((c["flowkit"] == c["flowjo"]).sum()),
                    "median_rel_diff_pct": med, "max_rel_diff_pct": worst,
                    "max_abs_diff": int((c["flowkit"] - c["flowjo"]).abs().max()) if len(c) else None,
                    # FlowKit reproduces FlowJo on this gate (wells with >= 100 events).
                    "agrees_with_flowjo": None if med is None else bool(
                        med <= AGREE_MEDIAN_PCT and worst <= AGREE_MAX_PCT)})
            failures = info.get("load_failures", {})
            for _, s in sdf.iterrows():
                has_file = bool(s["md5"])
                control = s["control"] not in ("", "-")
                usable = sorted(set(counts.loc[counts["sample"] == s["sample"], "gate"]) - time_gates)
                notes = []
                rel = (str(Path(s["file_path"]).relative_to(DATA)).replace("\\", "/")
                       if has_file else "")
                in_dataset = rel.removeprefix(f"{dataset}/")
                if not has_file:
                    notes.append(f"no FCS: {s['file']}")
                if s["label"] in failures:
                    notes.append(f"FlowKit failed: {failures[s['label']]}")
                if manifest and has_file and manifest.get(in_dataset) != s["md5"]:
                    notes.append("MD5 does not match the download manifest")
                unreliable = (UNRELIABLE_LABELS.get((dataset, wsp), "")
                              or UNUSABLE_SAMPLES.get((dataset, wsp, s["sample"]), ""))
                if unreliable:
                    notes.append(f"labels unreliable: {unreliable}")
                row = {"dataset": dataset, "workspace": wsp, "sample": s["sample"],
                       "file": rel, "md5": s["md5"], "events": int(s["events"]),
                       **identify(dataset, Path(wsp).stem, s),
                       "control": control, "control_reason": s["control"] if control else "",
                       "usable": (has_file and not control and s["label"] not in failures
                                  and not unreliable
                                  and (not manifest or manifest.get(in_dataset) == s["md5"])),
                       "usable_gates": usable if has_file and not control else [],
                       "notes": "; ".join(notes)}
                samples.append(row)
                if has_file:
                    used_files.add(row["file"])
        on_disk = sorted(f"{dataset}/{p}" for p in manifest if p.lower().endswith(".fcs")) or \
            sorted(str(p.relative_to(DATA)).replace("\\", "/")
                   for p in (DATA / dataset).rglob("*.fcs"))
        unregistered[dataset] = [f for f in on_disk if f not in used_files]
        datasets[dataset] = {**meta, "gates": gate_rows,
                             "fcs_files_not_in_any_workspace": unregistered[dataset]}

    # The same file content registered twice (e.g. a Dropbox "conflicted copy") is one sample.
    by_md5 = {}
    for r in samples:
        if r["md5"] and r["md5"] in by_md5 and by_md5[r["md5"]] != (r["dataset"], r["sample"]):
            r["notes"] = "; ".join(filter(None, [r["notes"], "duplicate content of "
                                                 + "/".join(by_md5[r["md5"]])]))
        by_md5.setdefault(r["md5"], (r["dataset"], r["sample"]))

    registry = {
        "version": "datasets_v1", "created": date.today().isoformat(),
        "data_dir": "paths are relative to ~/code/reinvent-the-wheel/data",
        "individual_ids": "human:<4-digit ID> and mouse:<dataset>:<id> are real identities; a "
                          "leading '?' means identity is unknown (may be anyone) and such "
                          "samples may only be used for pretraining, never for validation/test",
        "datasets": datasets, "samples": samples}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(registry, indent=1))

    df = pd.DataFrame(samples)
    print(df.groupby("dataset").agg(
        samples=("sample", "size"), with_file=("md5", lambda m: (m != "").sum()),
        controls=("control", "sum"), usable=("usable", "sum"),
        individuals=("individual", "nunique"),
        identity=("identity", lambda v: ",".join(sorted(set(v))))).to_string())
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
