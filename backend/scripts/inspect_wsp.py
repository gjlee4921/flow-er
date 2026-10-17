"""Print FlowJo workspaces' samples and gate trees (first check that FlowKit reads our data).

For each workspace: match its samples to FCS files on disk (headers only), flag control samples,
print header keywords, sample groups, the gate tree with which samples share each gate's geometry,
compensation and transforms, the event counts FlowJo saved, and then load FCS files ONE AT A TIME
to check that FlowKit's counts match FlowJo's. With several workspaces, also compare them.
With --out, also write per-workspace CSV/JSON files that build_registry.py reads.

    uv run python scripts/inspect_wsp.py                      # Figshare spleen, smallest file only
    uv run python scripts/inspect_wsp.py --wsp A.wsp --wsp B.wsp --load all --out DIR
"""

import argparse
import ctypes
import hashlib
import json
import os
import re
import sys
import traceback
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import flowkit as fk
import numpy as np
import pandas as pd

DEFAULT_WSP = (Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[3] / "data"))
               / "figshare_spleen" / "Spleen_Y_A_wsp_file.wsp")
# Header keywords that help tell samples, plates and donors apart, when a file has them.
ID_KEYWORDS = ["$FIL", "$SRC", "TUBE NAME", "WELL ID", "$DATE", "$BTIM", "EXPERIMENT NAME",
               "PLATE NAME", "$CYT", "CYTNUM", "SETTINGS"]
# Keywords that differ between any two files for technical reasons; not shown as "varying".
_TECHNICAL = re.compile(r"^(\$?P\d+[A-Z]+|\$(BEGIN|END)\w*|\$NEXTDATA|\$TOT|\$?SPILL\w*|GUID|"
                        r"\$ETIM|EXPORT TIME|.*DATE|.*TIME|\$BTIM|.*ID|.*CHECKSUM|\$CRCS?|"
                        r"#?P\d+\w*|.*VOLTAGE.*|.*DELAY|.*ASF)$")
# Sample or file names of controls that must never be used as training samples.
_CONTROL = re.compile(r"(?i)(\bfmo\b|fmo[ _-]|unstain|bead|compensation|stained control|isotype|"
                      r"single[ _-]?(stain|f480|apc|pe)|^single|_single)")
# Above this many samples, print per-gate summaries instead of gate x sample matrices.
WIDE = 16


def section(title: str) -> None:
    print(f"\n{'=' * 110}\n{title}\n{'=' * 110}")


# --- memory (Windows only; prints nothing elsewhere) -------------------------------------------


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong)
        for name in ("total_phys", "avail_phys", "total_page", "avail_page", "total_virt",
                     "avail_virt", "avail_ext_virt")
    ]


class _ProcessMemory(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
        (name, ctypes.c_size_t)
        for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                     "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                     "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")
    ]


def free_ram_mb() -> float | None:
    if os.name != "nt":
        return None
    status = _MemoryStatus(dwLength=ctypes.sizeof(_MemoryStatus))
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.avail_phys / 2**20


def peak_memory_mb() -> tuple[float, float] | None:
    """(peak working set, peak committed) of this process in MB."""
    if os.name != "nt":
        return None
    counters = _ProcessMemory(cb=ctypes.sizeof(_ProcessMemory))
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters),
                                     counters.cb)
    return counters.PeakWorkingSetSize / 2**20, counters.PeakPagefileUsage / 2**20


# --- FCS header and FlowJo XML, read without loading events ------------------------------------


def read_fcs_text(path: Path) -> dict[str, str]:
    """The FCS TEXT segment as a dict; reads a few KB, never the event data."""
    with open(path, "rb") as f:
        header = f.read(58)
        start, end = int(header[10:18]), int(header[18:26])
        f.seek(start)
        text = f.read(end - start + 1).decode("latin-1")
    delim = text[0]
    # A doubled delimiter is an escaped literal delimiter inside a value.
    parts = text[1:].replace(delim * 2, "\0").rstrip(delim).split(delim)
    parts = [p.replace("\0", delim) for p in parts]
    return {k.upper(): v.strip() for k, v in zip(parts[::2], parts[1::2])}


def flowjo_counts(wsp_path: Path) -> dict[str, dict[tuple[str, ...], int]]:
    """Event counts FlowJo saved per sample and gate path, keyed like FlowKit's gate paths."""
    counts = {}
    for node in ET.parse(wsp_path).getroot().iter("SampleNode"):
        counts[node.get("name")] = {(): int(node.get("count"))}
        _walk_populations(node, ("root",), counts[node.get("name")])
    return counts


def _walk_populations(parent, path: tuple[str, ...], out: dict) -> None:
    subpops = parent.find("Subpopulations")
    for pop in [] if subpops is None else subpops.findall("Population"):
        gate_path = path + (pop.get("name"),)
        out[gate_path] = int(pop.get("count"))
        _walk_populations(pop, gate_path, out)


def wsp_keywords(wsp_path: Path) -> dict[str, dict[str, str]]:
    """FCS keywords FlowJo stored per sample in the workspace (GUID, $BTIM, ...)."""
    out = {}
    for sample in ET.parse(wsp_path).getroot().iter("Sample"):
        node, kws = sample.find("SampleNode"), sample.find("Keywords")
        if node is not None and kws is not None:
            out[node.get("name")] = {k.get("name", "").upper(): (k.get("value") or "").strip()
                                     for k in kws}
    return out


# Keywords that identify one acquisition; compared between the workspace and the FCS header.
_ACQUISITION_KEYS = ("GUID", "$DATE", "$BTIM", "$ETIM")


def match_fcs(sample_ids: list[str], counts: dict, headers: dict[Path, dict],
              keywords: dict[str, dict]) -> tuple[dict[str, Path], dict[str, str]]:
    """Sample ID -> FCS file with the same event count, a matching name and, where FlowJo
    stored them, the same acquisition keywords (GUID, date, start and end time).

    A name matches when the sample ID equals, or contains or is contained in as whole words, the
    file name or $FIL (ignoring case, .fcs and separators): repositories add prefixes
    ("HC ...") and FlowJo adds suffixes. Names alone are not enough: downloads flatten folders,
    so different experiments' files can share a name or $FIL. Returns the matches and a note per
    unmatched sample.
    """
    def norm(s: str) -> str:
        return re.sub(r"[\s_-]+", " ", s.lower().removesuffix(".fcs")).strip()

    def within(a: str, b: str) -> bool:
        return bool(re.search(rf"(^| ){re.escape(a)}( |$)", b))

    def same_name(sid: str, p: Path, h: dict) -> bool:
        a = norm(sid)
        return any(b and (within(a, b) or within(b, a))
                   for b in (norm(p.name), norm(h.get("$FIL", ""))))

    def same_acquisition(sid: str, h: dict) -> bool:
        stored = keywords.get(sid, {})
        return all(stored[k].lower() == h[k].lower()
                   for k in _ACQUISITION_KEYS if stored.get(k) and h.get(k))

    matched, notes = {}, {}
    for sid in sample_ids:
        n = counts[sid][()]
        named = [p for p, h in headers.items() if same_name(sid, p, h)]
        counted = [p for p in named if int(headers[p]["$TOT"]) == n]
        cands = [p for p in counted if same_acquisition(sid, headers[p])]
        if len(cands) > 1 and len({md5(p) for p in cands}) == 1:
            cands = cands[:1]                     # byte-identical copies of one file
        if len(cands) == 1:
            matched[sid] = cands[0]
        elif cands:
            notes[sid] = f"ambiguous: {[p.name for p in cands]}"
        elif counted:
            notes[sid] = (f"name and event count match {[p.name for p in counted]} but the "
                          f"workspace's GUID/date/time keywords differ")
        elif named:
            notes[sid] = (f"name matches {[p.name for p in named]} but event counts "
                          f"{[headers[p]['$TOT'].lstrip('0') for p in named]} != {n}")
        else:
            notes[sid] = "no FCS file"
    return matched, notes


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- descriptions -------------------------------------------------------------------------------


def describe_geometry(gate) -> str:
    if gate.gate_type == "PolygonGate":
        return f"{len(gate.vertices)} vertices"
    if gate.gate_type == "RectangleGate":
        return "  ".join(f"[{_fmt(d.min)}, {_fmt(d.max)}]" for d in gate.dimensions)
    return gate.gate_type


def geometry_key(gate) -> bytes:
    if gate.gate_type == "PolygonGate":
        arr = np.asarray(gate.vertices, dtype=float)
    elif gate.gate_type == "RectangleGate":
        arr = np.asarray([[np.nan if v is None else v for v in (d.min, d.max)]
                          for d in gate.dimensions], dtype=float)
    else:
        return repr(sorted((k, repr(v)) for k, v in vars(gate).items()
                           if k not in ("gate_name", "dimensions"))).encode()
    return str(arr.shape).encode() + np.round(arr, 6).tobytes()


def _fmt(v) -> str:
    return "open" if v is None else f"{v:.4g}"


def channel_label(channel: str, markers: dict[str, str]) -> str:
    raw = channel.removeprefix("Comp-")
    marker = markers.get(raw)
    return f"{channel} ({marker})" if marker and marker != raw else channel


def sample_labels(sample_ids: list[str], headers: dict[str, dict]) -> dict[str, str]:
    """Short label per sample: the WELL ID keyword when every file has a distinct one, else the
    sample ID without its common prefix and .fcs."""
    wells = {sid: headers.get(sid, {}).get("WELL ID") for sid in sample_ids}
    if all(wells.values()) and len(set(wells.values())) == len(wells):
        return wells
    prefix = os.path.commonprefix(sample_ids) if len(sample_ids) > 1 else ""
    return {sid: sid[len(prefix):].removesuffix(".fcs") or sid for sid in sample_ids}


def gate_labels(gate_ids: list) -> dict:
    """Indented gate name, plus the parent when a name occurs more than once in the tree
    (e.g. the same cytokine gate under CD4 and CD8)."""
    n = Counter(name for name, _ in gate_ids)
    return {(name, path): "  " * (len(path) - 1) + name + (f" [{path[-1]}]" if n[name] > 1 else "")
            for name, path in gate_ids}


def gate_key(name: str, path: tuple) -> str:
    return "/".join(path[1:] + (name,))


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def sample_groups(wsp, sample_ids: list[str]) -> dict[str, frozenset]:
    """Sample groups other than 'All Samples'; groups with identical members are merged."""
    by_members = defaultdict(list)
    for g in wsp.get_sample_groups():
        members = frozenset(wsp.get_sample_ids(group_name=g, loaded_only=False)) & set(sample_ids)
        if members and members != frozenset(sample_ids):
            by_members[members].append(g)
    return {" | ".join(names): m for m, names in by_members.items()}


def describe_members(members: frozenset, groups: dict[str, frozenset],
                     labels: dict[str, str]) -> str:
    covered = [name for name, m in groups.items() if m <= members]
    # Prefer the largest groups so an exact cover reads as few names as possible.
    covered.sort(key=lambda n: -len(groups[n]))
    chosen, union = [], frozenset()
    for name in covered:
        if not groups[name] <= union:
            chosen.append(name)
            union |= groups[name]
    if chosen and union == members:
        return " + ".join(f"[{n}]" for n in chosen)
    names = sorted((labels[s] for s in members), key=natural_key)
    return ",".join(names) if len(names) <= 12 else f"{len(names)} samples: {','.join(names[:6])}…"


def control_reason(sid: str, path: Path | None, groups: list[str]) -> str:
    for text in (sid, path.name if path else ""):
        m = _CONTROL.search(text)
        if m:
            return f"name: {m.group(0).strip(' _-')}"
    if any(g.split(" | ")[0].lower().startswith("compensation") for g in groups):
        return "in Compensation group"
    return ""


# --- one workspace --------------------------------------------------------------------------------


@dataclass
class WspInfo:
    path: Path
    sample_ids: list[str]
    files: dict[str, Path]
    headers: dict[str, dict]
    labels: dict[str, str]
    groups: dict[str, frozenset]
    gate_ids: list
    gates: dict = field(default_factory=dict)            # (name, path) -> (type, dims)
    versions: dict = field(default_factory=dict)         # (name, path) -> (kind, n versions)
    comp: np.ndarray | None = None
    comp_detectors: list = field(default_factory=list)
    xforms: dict = field(default_factory=dict)           # gated channel -> repr
    error: str = ""


def inspect_workspace(wsp_path: Path, fcs_paths: list[Path], load: str, max_mb: float,
                      out: Path | None) -> WspInfo:
    folders = sorted({str(p.parent) for p in fcs_paths})
    section(f"WORKSPACE {wsp_path.name}\n  path: {wsp_path}\n"
            f"  FCS candidates: {len(fcs_paths)} files in {folders}")
    wsp = fk.Workspace(str(wsp_path), load_missing_file_data=True)
    counts = flowjo_counts(wsp_path)
    sample_ids = sorted(wsp.get_sample_ids(loaded_only=False), key=natural_key)
    all_headers = {p: read_fcs_text(p) for p in sorted(fcs_paths)}
    files, match_notes = match_fcs(sample_ids, counts, all_headers, wsp_keywords(wsp_path))
    headers = {sid: all_headers[p] for sid, p in files.items()}
    labels = sample_labels(sample_ids, headers)
    groups = sample_groups(wsp, sample_ids)
    group_of = {sid: [g for g, m in groups.items() if sid in m] for sid in sample_ids}
    controls = {sid: control_reason(sid, files.get(sid), group_of[sid]) for sid in sample_ids}
    stem = re.sub(r"[^\w.-]+", "_", wsp_path.stem).strip("_")

    # 1. Samples vs files ----------------------------------------------------------------------
    print("\n--- 1. Samples vs FCS files (same event count AND same file name or $FIL)")
    print(wsp.summary().to_string(), "\n")
    rows = []
    for sid in sample_ids:
        p, h = files.get(sid), headers.get(sid, {})
        rows.append({"label": labels[sid], "sample": sid,
                     "file": p.name if p else match_notes.get(sid, "MISSING"),
                     "MB": round(p.stat().st_size / 2**20, 1) if p else None,
                     "events": counts[sid][()], "channels": h.get("$PAR"),
                     "groups": "; ".join(group_of[sid]) or "-", "control": controls[sid] or "-"})
    samples_df = pd.DataFrame(rows)
    print(samples_df.to_string(index=False))
    unmatched = [p.name for p in all_headers if p not in files.values()]
    print(f"\nsamples without an FCS file: {len(match_notes)}"
          + "".join(f"\n  {labels[s]}: {n}" for s, n in match_notes.items()))
    print(f"FCS files in the folder not matched to this workspace: {len(unmatched)}")
    print(f"control samples (excluded from training): "
          f"{sum(bool(c) for c in controls.values())}/{len(sample_ids)}")
    if files:
        ev = pd.Series({labels[s]: counts[s][()] for s in files})
        print(f"events per matched file: min {ev.min():,}  median {int(ev.median()):,}  "
              f"max {ev.max():,}  (largest: {ev.idxmax()}, "
              f"{max(p.stat().st_size for p in files.values()) / 2**20:.0f} MB)")

    # 2. Header keywords -----------------------------------------------------------------------
    print("\n--- 2. FCS header keywords (identity)")
    markers: dict[str, str] = {}
    if not headers:
        print("no matched FCS files")
    else:
        all_keys = set().union(*map(set, headers.values()))
        present = [k for k in ID_KEYWORDS if k in all_keys]
        varying = sorted(k for k in all_keys
                         if not _TECHNICAL.match(k) and k not in present
                         and len({h.get(k) for h in headers.values()}) > 1)
        donorish = sorted(k for k in all_keys
                          if re.search(r"DONOR|SUBJECT|PATIENT|PARTICIPANT|ANIMAL|MOUSE|SAMPLE ID|"
                                       r"TISSUE|ORGAN|GROUP|CONDITION|TREATMENT|STRAIN", k))
        shown = present + [k for k in varying if k not in present][:8]
        table = pd.DataFrame([{"label": labels[sid], **{k: headers[sid].get(k, "")[:40]
                                                        for k in shown}}
                              for sid in sample_ids if sid in headers])
        if len(table) > WIDE:
            for k in shown:
                vals = Counter(table[k])
                print(f"  {k:16s} {len(vals)} value(s): "
                      + "; ".join(f"{v!r}×{c}" for v, c in vals.most_common(8))
                      + (" …" if len(vals) > 8 else ""))
        else:
            print(table.to_string(index=False))
        print(f"\nother non-technical keywords that vary between files: {varying or 'none'}")
        print(f"keywords that may hold identity/condition: "
              f"{ {k: sorted({h.get(k, '') for h in headers.values()})[:6] for k in donorish} or 'none'}")

        channels = {sid: [h[f"$P{i}N"] for i in range(1, int(h["$PAR"]) + 1)]
                    for sid, h in headers.items()}
        for sid, h in headers.items():
            for i, ch in enumerate(channels[sid], start=1):
                if h.get(f"$P{i}S") and ch not in markers:
                    markers[ch] = h[f"$P{i}S"]
        panels = Counter(tuple(c) for c in channels.values())
        print(f"\ndistinct channel lists: {len(panels)}"
              + "".join(f"\n  {n} file(s), {len(c)} channels: {list(c)}"
                        for c, n in panels.most_common(4)))
        print(f"marker names ($PnS): {markers or 'none in any file'}")

    # 3. Gate tree ------------------------------------------------------------------------------
    print("\n--- 3. Gate tree")
    trees = {sid: wsp.get_gate_ids(sid) for sid in sample_ids}
    tree_kinds = Counter(tuple(sorted(t)) for t in trees.values())
    biggest = max(sample_ids, key=lambda s: len(trees[s]))
    print(wsp.get_gate_hierarchy(biggest, output="ascii"), "\n")
    print(f"distinct gate trees: {len(tree_kinds)}")
    for t, n in tree_kinds.most_common():
        holders = [labels[s] for s in sample_ids if tuple(sorted(trees[s])) == t]
        ctrl = sum(bool(controls[s]) for s in sample_ids if tuple(sorted(trees[s])) == t)
        print(f"  {len(t):3d} gates: {n} samples ({ctrl} controls), e.g. {holders[:5]}")
    seen, gate_ids = set(), []
    for s in sorted(sample_ids, key=lambda s: -len(trees[s])):
        for g in trees[s]:
            if g not in seen:
                seen.add(g)
                gate_ids.append(g)
    glabel = gate_labels(gate_ids)
    info = WspInfo(wsp_path, sample_ids, files, headers, labels, groups, gate_ids)

    rows, version_rows = [], []
    for name, path in gate_ids:
        holders = [s for s in sample_ids if (name, path) in trees[s]]
        gate = wsp.get_gate(holders[0], name, path)
        dims = gate.dimensions
        info.gates[(name, path)] = (gate.gate_type, tuple(d.id for d in dims))
        # Which samples share this gate's geometry? One version = a template copied to all.
        by_geom = defaultdict(set)
        for sid in holders:
            by_geom[geometry_key(wsp.get_gate(sid, name, path))].add(sid)
        versions = sorted((frozenset(v) for v in by_geom.values()), key=len, reverse=True)
        for k, v in enumerate(versions):
            version_rows += [{"sample": s, "gate": gate_key(name, path), "version": k} for s in v]
        holder_groups = {g: m & set(holders) for g, m in groups.items() if m & set(holders)}
        aligned = all(describe_members(v, holder_groups, labels).startswith("[")
                      for v in versions)
        if len(versions) == 1:
            kind = "template"
        elif len(versions) == len(holders):
            kind = "per sample"
        elif aligned:
            kind = "by group"
        else:
            kind = "mixed"
        info.versions[(name, path)] = (kind, len(versions))
        if len(versions) > 1 and len(versions[0]) > len(holders) / 2:
            shown = [describe_members(v, holder_groups, labels) for v in versions[1:6]]
            detail = (f"rest share one; differs: {' / '.join(shown)}"
                      + (f" / … {len(versions) - 6} more" if len(versions) > 6 else ""))
        elif len(versions) > 1:
            detail = " / ".join(describe_members(v, holder_groups, labels) for v in versions[:6])
            detail += f" / … {len(versions) - 6} more" if len(versions) > 6 else ""
        else:
            detail = ""
        rows.append({
            "gate": glabel[(name, path)],
            "type": gate.gate_type.replace("Gate", "") + ("(compl.)" if getattr(
                gate, "use_complement", False) else ""),
            "x": channel_label(dims[0].id, markers) if dims else "-",
            "y": channel_label(dims[1].id, markers) if len(dims) > 1 else "-",
            "comp": ",".join(sorted({str(d.compensation_ref) for d in dims})),
            "geometry (first holder)": describe_geometry(gate),
            "samples": len(holders),
            "versions": len(versions),
            "kind": kind,
            "which samples share": detail[:120],
        })
    gates_df = pd.DataFrame(rows)
    print(gates_df.to_string(index=False))
    print("\ngate types:", gates_df["type"].value_counts().to_dict(),
          "| 1D gates:", gates_df[gates_df["y"] == "-"]["gate"].str.strip().tolist() or "none")

    # 4. Compensation ---------------------------------------------------------------------------
    print("\n--- 4. Compensation")
    mats = {}
    for sid in sample_ids:
        try:
            mats[sid] = wsp.get_comp_matrix(sid)
        except Exception as e:  # noqa: BLE001 - report and continue
            mats[sid] = None
            print(f"  {labels[sid]}: cannot read matrix: {e}")
    distinct = []
    for sid, m in mats.items():
        for d in distinct:
            a = d["m"]
            if (a is None and m is None) or (a is not None and m is not None and list(
                    a.detectors) == list(m.detectors) and np.allclose(a.matrix, m.matrix)):
                d["samples"].append(sid)
                break
        else:
            distinct.append({"m": m, "samples": [sid]})
    print(f"distinct matrices: {len(distinct)}")
    for d in distinct:
        m = d["m"]
        ctrl = sum(bool(controls[s]) for s in d["samples"])
        if m is None:
            print(f"  none: {len(d['samples'])} samples ({ctrl} controls)")
            continue
        arr = np.asarray(m.matrix, dtype=float)
        off = arr - np.eye(len(arr))
        print(f"  {arr.shape[0]} detectors, identity={np.allclose(off, 0)}, "
              f"max |off-diag|={np.abs(off).max():.3g}: {len(d['samples'])} samples "
              f"({ctrl} controls), e.g. {[labels[s] for s in d['samples'][:4]]}")
    main = max(distinct, key=lambda d: len(d["samples"]))["m"]
    if main is not None and headers:
        info.comp, info.comp_detectors = np.asarray(main.matrix, float), list(main.detectors)
        chans = next(iter(headers.values()))
        pnn = [chans[f"$P{i}N"] for i in range(1, int(chans["$PAR"]) + 1)]
        missing = [d for d in main.detectors if d not in pnn]
        if missing:
            print(f"  detectors not among the FCS $PnN names: {missing} "
                  "(FlowJo writes '/' as '_'; loaded with use_flowjo_labels=True)")

    # 5. Transforms -----------------------------------------------------------------------------
    print("\n--- 5. Transforms of channels used by gates")
    xforms = {sid: wsp.get_transforms(sid) for sid in sample_ids}
    gated_ch = {d for _, dims in info.gates.values() for d in dims}
    rows = []
    for ch in sorted(gated_ch, key=natural_key):
        reprs = Counter(repr(xforms[s].get(ch)) for s in sample_ids if ch in xforms[s])
        info.xforms[ch] = reprs.most_common(1)[0][0] if reprs else "none"
        rows.append({"channel": channel_label(ch, markers), "transform": info.xforms[ch],
                     "versions": len(reprs)})
    print(pd.DataFrame(rows).to_string(index=False))

    # 6. FlowJo's counts ------------------------------------------------------------------------
    print("\n--- 6. Event counts FlowJo saved (gate x sample)")
    fj = pd.DataFrame({labels[s]: {glabel[(n, p)]: counts[s].get(tuple(p) + (n,))
                                   for n, p in gate_ids} for s in sample_ids}).reindex(
        [glabel[g] for g in gate_ids])
    data_cols = [labels[s] for s in sample_ids if not controls[s]]
    if len(sample_ids) > WIDE:
        print(f"(summary over {len(data_cols)} non-control samples)")
        print(fj[data_cols].T.describe().T[["count", "min", "50%", "max"]]
              .round().astype("Int64").to_string())
    else:
        print(fj.to_string())

    # 7. FlowKit vs FlowJo ----------------------------------------------------------------------
    loadable = [s for s in sample_ids if s in files]
    to_load = {"none": [], "all": loadable,
               "samples": [s for s in loadable if not controls[s]],
               "smallest": sorted(loadable, key=lambda s: files[s].stat().st_size)[:1]
               }.get(load, [load] if load in files else [])
    print(f"\n--- 7. FlowKit counts minus FlowJo counts, loading {len(to_load)} file(s) one at a "
          f"time (free RAM {free_ram_mb() or float('nan'):.0f} MB)")
    diffs, kit, failures = {}, {}, {}
    for sid in to_load:
        if files[sid].stat().st_size > max_mb * 2**20:
            failures[labels[sid]] = f"skipped: larger than --max-mb {max_mb}"
            continue
        try:
            sample = fk.Sample(str(files[sid]), sample_id=sid, use_flowjo_labels=True)
            one = fk.Workspace(str(wsp_path), fcs_samples=[sample], load_missing_file_data=True)
            one.analyze_samples(sample_id=sid, cache_events=False, use_mp=False)
            report = one.get_gating_results(sid).report
        except Exception as e:  # noqa: BLE001 - report and continue with the next file
            failures[labels[sid]] = f"{type(e).__name__}: {str(e)[:200]}"
            continue
        finally:
            sample = one = None
        kit[labels[sid]] = {glabel[(n, tuple(p))]: c for p, n, c in
                            zip(report["gate_path"], report["gate_name"], report["count"])}
        diffs[labels[sid]] = {g: c - fj.loc[g, labels[sid]] for g, c in kit[labels[sid]].items()}
    for lab, why in failures.items():
        print(f"  {lab}: {why}")
    if diffs:
        diff_df = pd.DataFrame(diffs).reindex(fj.index)
        rel = 100 * diff_df.abs() / fj[diff_df.columns].clip(lower=1)
        if diff_df.shape[1] > WIDE:
            summary = pd.DataFrame({
                "samples": diff_df.notna().sum(1),
                "exact": (diff_df == 0).sum(1),
                "median |diff|": diff_df.abs().median(1),
                "max |diff|": diff_df.abs().max(1),
                "max |diff| % (>=100 ev)": rel.where(fj[diff_df.columns] >= 100).max(1).round(2),
                "worst sample": diff_df.abs().idxmax(1)})
            print(summary.to_string())
        else:
            print(diff_df.to_string())
        flagged = [(g.strip(), s, int(diff_df.loc[g, s]), int(fj.loc[g, s]))
                   for g in diff_df.index for s in diff_df.columns
                   if pd.notna(diff_df.loc[g, s]) and rel.loc[g, s] > 1
                   and abs(diff_df.loc[g, s]) > 2]
        big = rel.where(fj[diff_df.columns] >= 100)
        print(f"\nexact matches: {int((diff_df == 0).sum().sum())}/{int(diff_df.notna().sum().sum())};"
              f" max |diff| on gates with >=100 events: {np.nanmax(big.to_numpy()):.3f}%")
        print(f"gate/sample pairs off by >1% AND >2 events: {len(flagged)}"
              + "".join(f"\n  {f}" for f in flagged[:15])
              + (f"\n  … {len(flagged) - 15} more" if len(flagged) > 15 else ""))

    # Files for build_registry.py ---------------------------------------------------------------
    if out:
        out.mkdir(parents=True, exist_ok=True)
        samples_df["md5"] = [md5(files[s]) if s in files else "" for s in sample_ids]
        samples_df["file_path"] = [str(files[s]) if s in files else "" for s in sample_ids]
        samples_df["tree_gates"] = [len(trees[s]) for s in sample_ids]
        for k in ID_KEYWORDS:
            samples_df[k] = [headers.get(s, {}).get(k, "") for s in sample_ids]
        samples_df.to_csv(out / f"{stem}.samples.csv", index=False)
        gates_df.assign(gate_key=[gate_key(n, p) for n, p in gate_ids]).to_csv(
            out / f"{stem}.gates.csv", index=False)
        pd.DataFrame(version_rows).assign(label=lambda d: d["sample"].map(labels)).to_csv(
            out / f"{stem}.gate_versions.csv", index=False)
        long = [{"sample": s, "label": labels[s], "gate": gate_key(n, p),
                 "flowjo": counts[s].get(tuple(p) + (n,)),
                 "flowkit": kit.get(labels[s], {}).get(glabel[(n, p)])}
                for s in sample_ids for n, p in gate_ids if (n, p) in trees[s]]
        pd.DataFrame(long).to_csv(out / f"{stem}.counts.csv", index=False)
        (out / f"{stem}.json").write_text(json.dumps(
            {"wsp": str(wsp_path), "groups": {g: sorted(m) for g, m in groups.items()},
             "load_failures": failures, "match_notes": match_notes,
             "transforms": info.xforms}, indent=1))
        print(f"\nwrote {out / stem}.*")
    return info


# --- across workspaces ------------------------------------------------------------------------------


def compare_workspaces(infos: list[WspInfo], fcs_paths: list[Path]) -> None:
    section(f"ACROSS ALL {len(infos)} WORKSPACES")
    names = [i.path.name for i in infos]

    all_files = sorted(set(fcs_paths))
    claimed = defaultdict(list)
    for i in infos:
        for sid, p in i.files.items():
            claimed[p].append(f"{i.path.name}:{sid}")
    print(f"FCS files in the folders: {len(all_files)}; matched exactly once: "
          f"{sum(len(claimed[p]) == 1 for p in all_files)}; matched twice or more: "
          f"{[(p.name, claimed[p]) for p in all_files if len(claimed[p]) > 1] or 'none'}")
    unmatched = [p for p in all_files if not claimed.get(p)]
    print(f"unmatched files: {len(unmatched)}" + "".join(f"\n  {p.name}" for p in unmatched))

    gates = sorted({g for i in infos for g in i.gate_ids}, key=lambda g: (len(g[1]), g[1], g[0]))
    print(f"\ngate trees: {Counter(len(i.gate_ids) for i in infos)} gates per workspace; "
          f"{len(gates)} distinct gate paths overall")
    all_chans = Counter(tuple(h[f"$P{k}N"] for k in range(1, int(h["$PAR"]) + 1))
                        for i in infos for h in i.headers.values())
    print(f"distinct FCS channel lists across all files: {len(all_chans)}")
    print(f"transforms of gated channels identical in all workspaces: "
          f"{all(i.xforms == infos[0].xforms for i in infos)}")
    with_comp = [i for i in infos if i.comp is not None]
    if len(with_comp) > 1:
        comp = pd.DataFrame(index=names, columns=names, dtype=float)
        for a in with_comp:
            for b in with_comp:
                if a.comp_detectors == b.comp_detectors:
                    comp.loc[a.path.name, b.path.name] = np.abs(a.comp - b.comp).max()
        print("compensation, max |difference| between workspaces' main matrices:")
        print(comp.to_string(float_format=lambda v: f"{v:.3g}"))

    glabel = gate_labels(gates)
    table = pd.DataFrame({i.path.name: {
        glabel[g]: (f"{i.versions[g][0]} ({i.versions[g][1]})" if g in i.versions else "-")
        for g in gates} for i in infos}).reindex([glabel[g] for g in gates])
    print("\nhow each gate varies, per workspace:")
    print(table.to_string())

    guids, sums = defaultdict(list), defaultdict(list)
    for i in infos:
        for sid, h in i.headers.items():
            if h.get("GUID"):
                guids[h["GUID"]].append(f"{i.path.name}:{i.labels[sid]}")
            sums[md5(i.files[sid])].append(f"{i.path.name}:{i.labels[sid]}")
    print(f"\nduplicate FCS GUIDs: {[v for v in guids.values() if len(v) > 1] or 'none'}; "
          f"files with identical content: {[v for v in sums.values() if len(v) > 1] or 'none'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wsp", type=Path, action="append",
                        help=f"workspace file (repeatable); default {DEFAULT_WSP}")
    parser.add_argument("--fcs-dir", type=Path, action="append",
                        help="FCS folder per --wsp; default <wsp folder>/fcs if it exists, "
                             "else the wsp folder")
    parser.add_argument("--manifest", type=Path,
                        help="manifest.json listing every file of a download (name, path, md5); "
                             "its FCS files are the candidates for every --wsp")
    parser.add_argument("--load", default="smallest",
                        help="FCS files to load for the count check: smallest | all | samples "
                             "(all but controls) | none | <sample id>. Always one at a time.")
    parser.add_argument("--max-mb", type=float, default=300,
                        help="never load an FCS file larger than this")
    parser.add_argument("--out", type=Path, help="folder for per-workspace CSV/JSON files")
    args = parser.parse_args()

    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 500,
                  "display.max_colwidth", 120)
    wsps = [p.resolve() for p in (args.wsp or [DEFAULT_WSP])]
    fcs_dirs = args.fcs_dir or [p.parent / "fcs" if (p.parent / "fcs").is_dir() else p.parent
                                for p in wsps]
    if len(fcs_dirs) != len(wsps):
        parser.error("give one --fcs-dir per --wsp, or none")
    if args.manifest:
        entries = json.loads(args.manifest.read_text())
        listed = [(args.manifest.parent / e["path"].replace(chr(92), "/")).resolve()
                  for e in entries if e["name"].lower().endswith(".fcs")]
        candidates = [listed for _ in wsps]
    else:
        candidates = [sorted(d.resolve().glob("*.fcs")) for d in fcs_dirs]
    print(f"FlowKit {fk.__version__}")

    infos = []
    for w, c in zip(wsps, candidates):
        try:
            infos.append(inspect_workspace(w, c, args.load, args.max_mb, args.out))
        except Exception:  # noqa: BLE001 - one unreadable workspace must not hide the others
            print(f"\n!!! FAILED on {w.name}:\n{traceback.format_exc()}")
    if len(infos) > 1:
        compare_workspaces(infos, sorted({p for c in candidates for p in c}))
    peak = peak_memory_mb()
    if peak:
        print(f"\npeak working set: {peak[0]:.0f} MB, peak committed: {peak[1]:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
