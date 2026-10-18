"""The data an experiment reads: FCS files, and a FlowJo workspace if there is one.

Reading goes through FlowKit here for now. `flow_er.cytometry` (io, gating, export) is where
reading FCS files and workspaces will live; when it gains its loaders, `_open()` is the one
place to switch over. Nothing in this module trains or places gates: gates come from the
FlowJo workspace, from the scientist, or later from `flow_er.ml`.

Display space: every channel goes through a transform so values sit in roughly 0..1, and the data
lab draws in that space. With one workspace whose transforms already give 0..1, that is the
workspace's own (FlowJo's gate vertices then need no change). Otherwise (several workspaces, or
FlowJo's biex, which runs to about 4096) it is the gate models' display for every file, and
FlowJo's shapes are moved into it.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", module="flowkit")
import flowkit as fk

from flow_er.config import settings

DISPLAY_N = 20000  # events kept per sample for plotting; counts always use every event
SEED = 7  # the display subset is fixed, so the same file always looks the same
SCATTER = re.compile(r"^(FSC|SSC)|time", re.IGNORECASE)
TIME = re.compile(r"^time$", re.IGNORECASE)
# the usual live/dead dyes, so a viability channel reads as "Viability" rather than its dye
VIABILITY = re.compile(r"aqua|amine|live.?dead|zombie|viab|7-?aad|dapi|propidium|fixable|ghost dye|efluor ?780", re.IGNORECASE)
log = logging.getLogger(__name__)


FCS_VERSION = re.compile(rb"^FCS[23]\.\d")
# DATASETS_DIR/uploads/<name> holds files brought in through the start page; DATASETS_DIR/linked/<name> is a
# link to a folder elsewhere on this machine, read in place. Any other folder is an example dataset.
CONTAINERS = {"uploads": "upload", "linked": "folder"}


def fcs_header(path: Path) -> dict | None:
    """The TEXT segment of an FCS file, read without loading its events (a few KB, even for a 1 GB
    file). None when the file is not FCS."""
    try:
        with open(path, "rb") as f:
            head = f.read(58)
            if not FCS_VERSION.match(head):
                return None
            start, end = int(head[10:18].strip() or 0), int(head[18:26].strip() or 0)
            if end <= start:
                return None
            f.seek(start)
            text = f.read(end - start + 1).decode("latin-1")
    except (OSError, ValueError):
        return None
    delim = text[0]
    parts = text[1:].replace(delim * 2, "\0").split(delim)
    keys = {parts[i].upper(): parts[i + 1].replace("\0", delim) for i in range(0, len(parts) - 1, 2)}
    n = int(keys.get("$PAR", 0) or 0)
    params = [{"n": keys.get(f"$P{i}N", ""), "s": keys.get(f"$P{i}S", "")} for i in range(1, n + 1)]
    return {"version": head[:6].decode(), "events": int(keys.get("$TOT", 0) or 0), "params": params, "keywords": keys}


@lru_cache(maxsize=256)
def _markers(path: str, mtime: float) -> tuple[str, ...]:
    h = fcs_header(Path(path))
    if not h:
        return ()
    return tuple(short_label(p["n"], p["s"]) for p in h["params"] if p["n"] and channel_group(p["n"]) == "marker")


def describe_dataset(d: Path, name: str, kind: str) -> dict | None:
    """A folder as the start page shows it: files, size, workspace, and the panel's markers."""
    fcs = sorted(d.rglob("*.fcs"))
    if not fcs:
        return None
    wsp = sorted(d.rglob("*.wsp"))
    label = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", d.name) if kind == "upload" else d.name
    return {"name": name, "label": re.sub(r"[_-]+", " ", label), "kind": kind, "path": str(d), "files": len(fcs),
            "workspace": wsp[0].name if wsp else None, "markers": list(_markers(str(fcs[0]), fcs[0].stat().st_mtime)),
            "size_mb": round(sum(f.stat().st_size for f in fcs) / 1048576, 1)}


def find_datasets() -> list[dict]:
    """Folders under DATASETS_DIR that hold FCS files: examples, uploads and linked folders."""
    base = Path(os.path.abspath(settings.datasets_dir.expanduser()))
    if not base.exists():
        return []
    brought: list[tuple[float, dict]] = []
    examples: list[dict] = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        if d.name in CONTAINERS:
            for c in (p for p in d.iterdir() if p.is_dir()):
                if (x := describe_dataset(c, f"{d.name}/{c.name}", CONTAINERS[d.name])):
                    brought.append((c.lstat().st_mtime, x))
        elif (x := describe_dataset(d, d.name, "example")):
            examples.append(x)
    # what was brought in comes first, newest first, so a new upload is at the front
    return [x for _, x in sorted(brought, key=lambda t: t[0], reverse=True)] + examples


def short_label(pnn: str, pns: str = "") -> str:
    """'CD3 APC-H7 FLR-A' -> 'CD3'; scatter channels keep their name."""
    if pns:
        return pns.split()[0]
    if SCATTER.match(pnn):
        return pnn
    return pnn.split()[0] if " " in pnn else pnn


def channel_group(pnn: str) -> str:
    """scatter, time or marker: how the axis pickers group a channel."""
    if TIME.match(pnn):
        return "time"
    return "scatter" if SCATTER.match(pnn) else "marker"


def channel_name(pnn: str, pns: str = "") -> str:
    """The name a person reads: 'CD3 (APC-H7)', 'Viability (Aqua Amine)', 'FSC-A'."""
    if channel_group(pnn) != "marker":
        return pnn
    words = (pns or pnn).replace(" FLR-A", "").replace("-A", "").split()
    if VIABILITY.search(pnn) or VIABILITY.search(pns):
        return f"Viability ({' '.join(words)})"
    marker, fluor = words[0], " ".join(words[1:])
    return f"{marker} ({fluor})" if fluor else marker


@dataclass
class Gate:
    key: str            # path-like id, e.g. "Time/Singlets/aAmine-/CD3+"
    name: str
    parent: str | None  # parent key; None for the root's children
    source: str         # flowjo | user | model
    dims: list[str] = field(default_factory=list)  # channel ids (pnn) for x, y
    vertices: list[list[float]] = field(default_factory=list)
    kind: str = "polygon"
    reviewed_on: str | None = None  # a model gate put up for review: the file it was reviewed on


def _words(text: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", text.lower()) if w}


def _named(f: Path, fil: str, names) -> str | None:
    """The name a workspace knows this file by: its name at acquisition ($FIL), its file name, or the
    file name less words added in front of it later (FlowJo kept "Specimen_001_C2_C02 Peptide.fcs",
    the file was then saved as "ALS Specimen_001_C2_C02 Peptide.fcs")."""
    if fil in names:
        return fil
    if f.name in names:
        return f.name
    return next((n for n in names if len(f.name) > len(n) and f.name.endswith(n) and f.name[-len(n) - 1] in " _-"), None)


def _assign(fcs: list[Path], wsps: list[Path]) -> dict[Path, list[tuple[Path, str]]]:
    """Which FCS files each FlowJo workspace gates, and by what name inside it. A workspace names its
    samples by the file name they had at acquisition ($FIL), so two groups (say ALS and healthy
    controls) can each have a "Specimen_001_C1_C01.fcs"; a file then goes to the workspace whose own
    name and folders share the most words with the file's name (ALS with ALS, HC with HC), counting
    only the words those workspaces don't all share (both may sit under an "ALS study" folder)."""
    from urllib.parse import unquote

    from flowkit._utils import wsp_utils  # the sample names and file paths a .wsp lists

    listed = {}
    for w in wsps:
        try:
            samples = wsp_utils.parse_wsp(str(w))["samples"]
        except Exception:
            log.warning("could not read %s", w, exc_info=True)
            continue
        listed[w] = {name: _words(w.stem + " " + unquote(str(d.get("sample_uri", "")))) for name, d in samples.items()}
    out: dict[Path, list[tuple[Path, str]]] = {w: [] for w in listed}
    for f in fcs:
        kw = (fcs_header(f) or {}).get("keywords", {})
        fil = kw.get("$FIL") or f.name
        named = {w: n for w, names in listed.items() if (n := _named(f, fil, names))}
        if not named:
            continue
        # the words the file gained after acquisition, and the plate it was run on
        own = (_words(f.stem) - _words(Path(fil).stem)) | _words(kw.get("PLATE NAME", ""))
        shared = set.intersection(*(listed[w][n] for w, n in named.items()))
        best = max(named, key=lambda w: len(own & (listed[w][named[w]] - shared)))
        out[best].append((f, named[best]))
    return out


class Dataset:
    """One folder of FCS files, read once and kept in memory. A sample is known by its file name;
    the folder may hold several FlowJo workspaces, each gating its own files."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.lock = threading.Lock()
        self.fcs = sorted(folder.rglob("*.fcs"))
        self.wsps = sorted(folder.rglob("*.wsp"))
        self.wsp_path = self.wsps[0] if self.wsps else None
        self.workspaces: list[tuple[Path, fk.Workspace]] = []
        self._where: dict[str, tuple[fk.Workspace, str]] = {}  # file name -> (its workspace, its name there)
        if self.wsps and self.fcs:
            for w, files in _assign(self.fcs, self.wsps).items():
                if not files:
                    continue
                try:
                    # FlowKit knows a file by its $FIL; one FlowJo renamed is loaded under FlowJo's name,
                    # with FlowJo's channel names ("/" as "_") as FlowKit does for the others
                    ws = fk.Workspace(str(w), fcs_samples=[
                        str(f) if name == ((fcs_header(f) or {}).get("keywords", {}).get("$FIL") or f.name)
                        else fk.Sample(str(f), sample_id=name, use_flowjo_labels=True) for f, name in files])
                except Exception:
                    log.warning("could not read %s; its files are used without gates", w, exc_info=True)
                    continue
                self.workspaces.append((w, ws))
                loaded = set(ws.get_sample_ids())
                for f, name in files:
                    if name in loaded:
                        self._where[f.name] = (ws, name)
        self.ws = self.workspaces[0][1] if self.workspaces else None
        if self.workspaces:
            self.wsp_path = self.workspaces[0][0]
        # One display for every file when there is no workspace to take one from, or when the
        # workspaces' own would disagree: several workspaces (each plate its own transforms) or
        # FlowJo's biex scale (not 0-1). Then every file is drawn
        # and gated in the gate models' display units (linear x/$PnR, logicle T=$PnR W=0.5 M=4.5), so a
        # gate means the same cells in every file, and model polygons need no conversion.
        self.uniform = not self.workspaces or len(self.workspaces) > 1 or any(
            isinstance(t, fk.transforms.WSPBiexTransform)
            for ws, name in {id(w): (w, n) for w, n in self._where.values()}.values()
            for t in ws.get_transforms(name).values())
        self._samples: dict[str, fk.Sample] = {}
        self._events: dict[tuple[str, str], np.ndarray] = {}
        self._subset: dict[str, np.ndarray] = {}
        self._analysed: set[str] = set()

    # ---- samples and channels --------------------------------------------------------

    @property
    def sample_ids(self) -> list[str]:
        """Files with gates first, in file order, then any the workspaces don't list."""
        names = [p.name for p in self.fcs]
        return [n for n in names if n in self._where] + [n for n in names if n not in self._where]

    def workspace_for(self, sid: str) -> Path | None:
        """The FlowJo workspace that gates a file, if any."""
        ws = self._where.get(sid, (None, None))[0]
        return next((p for p, w in self.workspaces if w is ws), None)

    def sample_path(self, sid: str) -> Path:
        """The FCS file a sample id names."""
        return next(f for f in self.fcs if f.name == sid)

    def _sample(self, sid: str) -> fk.Sample:
        if sid not in self._samples:
            if sid in self._where:
                ws, name = self._where[sid]
                self._samples[sid] = ws.get_sample(name)
            else:
                p = next(f for f in self.fcs if f.name == sid)
                s = fk.Sample(str(p))
                # channels named as FlowJo and the gate models name them ("LIVE/DEAD Aqua-A" is
                # "LIVE_DEAD Aqua-A"), so a lab's gates find their channels in raw files too
                s.pnn_labels = [n.replace("/", "_") for n in s.pnn_labels]
                s.channels["pnn"] = s.pnn_labels
                if "spill" in s.metadata:
                    s.metadata["spill"] = s.metadata["spill"].replace("/", "_")
                self._samples[sid] = s
        return self._samples[sid]

    def samples(self) -> list[dict]:
        out = []
        for sid in self.sample_ids:
            s = self._sample(sid)
            out.append({"id": sid, "file": sid, "events": int(s.event_count)})
        return out

    def channels(self) -> list[dict]:
        s = self._sample(self.sample_ids[0])
        out = []
        for pnn, pns in zip(s.pnn_labels, s.pns_labels):
            out.append({"id": pnn, "label": pns or pnn, "short": short_label(pnn, pns),
                        "scatter": bool(SCATTER.match(pnn)), "group": channel_group(pnn), "name": channel_name(pnn, pns)})
        return out

    # ---- events in display space ------------------------------------------------------

    def recipe_transforms(self, sid: str) -> dict:
        """The gate models' display transforms for one file: linear x/$PnR for scatter and time,
        logicle T=$PnR W=0.5 M=4.5 for fluorescence (recipe 1bd7b6e6), each from 0 to about 1."""
        s = self._sample(sid)
        out = {}
        for n, pnn in enumerate(s.pnn_labels, 1):
            t = float(s.metadata.get(f"p{n}r") or 262144)
            if SCATTER.match(pnn) or pnn.lower() == "time":
                out[pnn] = fk.transforms.LinearTransform(param_t=t, param_a=0)
            else:
                out[pnn] = fk.transforms.LogicleTransform(param_t=t, param_w=0.5, param_m=4.5, param_a=0)
        return out

    def _transforms(self, sid: str) -> dict:
        if self.uniform:
            return self.recipe_transforms(sid)
        if sid in self._where:
            ws, name = self._where[sid]
            return ws.get_transforms(name)
        s = self._sample(sid)
        out = {}
        for pnn in s.pnn_labels:
            if SCATTER.match(pnn):
                out[pnn] = fk.transforms.LinearTransform(param_t=262144, param_a=0)
            else:
                out[pnn] = fk.transforms.LogicleTransform(param_t=262144, param_w=0.5, param_m=4.5, param_a=0)
        return out

    def _compensate(self, sid: str, raw: bool = False) -> np.ndarray:
        s = self._sample(sid)
        cm = None
        if not raw:
            cm = self._where[sid][0].get_comp_matrix(self._where[sid][1]) if sid in self._where else None
            if cm is None and "spill" in s.metadata:
                try:
                    # the file's own spillover matrix, over the detectors it lists (not every channel)
                    parts = s.metadata["spill"].split(",")
                    cm = fk.Matrix(s.metadata["spill"], parts[1:int(parts[0]) + 1])
                except Exception:
                    log.warning("could not use the spillover matrix in %s", sid, exc_info=True)
                    cm = None
        return cm.apply(s) if cm is not None else s.get_events(source="raw")

    def compensated(self, sid: str) -> np.ndarray:
        """Every event of a sample, compensated but not display-scaled: the values medians are
        reported on, as FlowJo does. Not cached; it is only read for statistics."""
        with self.lock:
            return np.asarray(self._compensate(sid), dtype=np.float64)

    def events(self, sid: str, stage: str = "processed") -> np.ndarray:
        """Every event of a sample, transformed; 'raw' skips compensation."""
        key = (sid, stage)
        if key not in self._events:
            with self.lock:
                s = self._sample(sid)
                comp = self._compensate(sid, raw=stage == "raw")
                xf = self._transforms(sid)
                ev = np.array(comp, dtype=np.float64, copy=True)
                for j, pnn in enumerate(s.pnn_labels):
                    t = xf.get(pnn)
                    if t is not None:
                        ev[:, j] = t.apply(ev[:, j])
                self._events[key] = ev
        return self._events[key]

    def subset(self, sid: str) -> np.ndarray:
        """A fixed random set of event indices for drawing."""
        if sid not in self._subset:
            n = self._sample(sid).event_count
            rng = np.random.default_rng(SEED)
            self._subset[sid] = np.sort(rng.choice(n, size=min(DISPLAY_N, n), replace=False))
        return self._subset[sid]

    # ---- gates ------------------------------------------------------------------------

    def _analyse(self, sid: str) -> None:
        if sid in self._where and sid not in self._analysed:
            ws, name = self._where[sid]
            with self.lock:
                ws.analyze_samples(sample_id=name)
                self._analysed.add(sid)

    def flowjo_gates(self, sid: str) -> list[Gate]:
        if sid not in self._where:
            return []
        self._analyse(sid)
        ws, name = self._where[sid]
        df = ws.get_gating_results(name).report
        gates = []
        for _, row in df.iterrows():
            path = tuple(row.gate_path)
            key = "/".join(list(path[1:]) + [row.gate_name])
            parent = "/".join(path[1:]) or None
            g = Gate(key=key, name=row.gate_name, parent=parent, source="flowjo", kind=str(row.gate_type))
            try:
                gd = ws.get_gate(name, row.gate_name, gate_path=path)
                g.dims = [d.id for d in gd.dimensions]
                if hasattr(gd, "vertices"):
                    g.vertices = [[float(a), float(b)] for a, b in gd.vertices]
                    g.kind = "polygon"
                    if self.uniform:
                        self._to_uniform(sid, g)
                elif hasattr(gd, "dimensions") and len(gd.dimensions) == 2:
                    d0, d1 = gd.dimensions
                    if d0.min is not None or d0.max is not None:
                        (lo0, hi0), (lo1, hi1) = self._bounds(sid, d0), self._bounds(sid, d1)
                        g.vertices = [[lo0, lo1], [hi0, lo1], [hi0, hi1], [lo0, hi1]]
                        g.kind = "rect"
            except Exception:
                log.info("no drawable shape for gate %s", key, exc_info=True)
            gates.append(g)
        return gates

    def _move(self, sid: str, dim: str, v: np.ndarray) -> np.ndarray:
        """Values on one axis of a FlowJo shape, from the workspace's display into the shared one:
        back to channel values with its transform, forward with the models'. A value at the very
        bottom or top of FlowJo's scale means "everything past here" (FlowJo piles events beyond its
        scale on the edge), so it lands well past the edge of the plot (the plot pins it to the edge)."""
        ws, name = self._where[sid]
        theirs, ours = ws.get_transforms(name).get(dim), self.recipe_transforms(sid).get(dim)
        v = np.asarray(v, dtype=float)
        if theirs is None:
            return ours.apply(v) if ours is not None else v
        out = ours.apply(theirs.inverse(v)) if ours is not None else theirs.inverse(v)
        floor, top = theirs.apply(np.array([-1e9, 1e9], float))
        out = np.where(v <= floor + 1e-6, -1.0, out)
        return np.where(v >= top - 1e-6, 2.0, out)

    def _bounds(self, sid: str, d) -> tuple[float, float]:
        """A rectangle's sides on one axis, in the display the lab draws; an open side sits well past
        the edge of the plot."""
        if not self.uniform:
            return (d.min if d.min is not None else 0, d.max if d.max is not None else 1)
        lo = float(self._move(sid, d.id, [d.min])[0]) if d.min is not None else -1.0
        hi = float(self._move(sid, d.id, [d.max])[0]) if d.max is not None else 2.0
        return lo, hi

    def _to_uniform(self, sid: str, g: Gate) -> None:
        """A FlowJo polygon moved into the shared display. Each edge is split first, since a straight
        edge in one display is a curve in another."""
        v = np.asarray(g.vertices, dtype=float)
        t = np.linspace(0, 1, 24, endpoint=False)[:, None]
        v = np.concatenate([a + t * (b - a) for a, b in zip(v, np.roll(v, -1, axis=0))])
        for j, c in enumerate(g.dims):
            v[:, j] = self._move(sid, c, v[:, j])
        g.vertices = v.tolist()

    def flowjo_membership(self, sid: str, key: str) -> np.ndarray:
        self._analyse(sid)
        ws, name = self._where[sid]
        parts = key.split("/")
        path = tuple(["root"] + parts[:-1])
        return np.asarray(ws.get_gate_membership(name, parts[-1], gate_path=path), dtype=bool)


@lru_cache(maxsize=8)
def open_dataset(folder: str) -> Dataset:
    return Dataset(Path(folder))


def in_polygon(x: np.ndarray, y: np.ndarray, poly: list[list[float]]) -> np.ndarray:
    """Even-odd rule, vectorised over events."""
    inside = np.zeros(x.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        crosses = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= crosses & (x < xint)
    return inside
