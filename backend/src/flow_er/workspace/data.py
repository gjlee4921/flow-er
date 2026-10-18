"""The data an experiment reads: FCS files, and a FlowJo workspace if there is one.

Reading goes through FlowKit here for now. `flow_er.cytometry` (io, gating, export) is where
reading FCS files and workspaces will live; when it gains its loaders, `_open()` is the one
place to switch over. Nothing in this module trains or places gates: gates come from the
FlowJo workspace, from the scientist, or later from `flow_er.ml`.

Display space: every channel goes through its transform (the workspace's own, or logicle for
fluorescence and linear for scatter) so values sit in roughly 0..1, the same space FlowJo's
gate vertices are stored in. The data lab draws in that space.
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
# DATA_DIR/uploads/<name> holds files brought in through the start page; DATA_DIR/linked/<name> is a
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
    """Folders under DATA_DIR that hold FCS files: examples, uploads and linked folders."""
    base = Path(os.path.abspath(settings.data_dir.expanduser()))
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


class Dataset:
    """One folder of FCS files, read once and kept in memory."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.lock = threading.Lock()
        self.fcs = sorted(folder.rglob("*.fcs"))
        wsps = sorted(folder.rglob("*.wsp"))
        self.wsp_path = wsps[0] if wsps else None
        self.ws = None
        if self.wsp_path:
            try:
                self.ws = fk.Workspace(str(self.wsp_path), fcs_samples=str(self.fcs[0].parent))
            except Exception:
                log.warning("could not read %s; using the FCS files alone", self.wsp_path, exc_info=True)
                self.ws = None
        self._samples: dict[str, fk.Sample] = {}
        self._events: dict[tuple[str, str], np.ndarray] = {}
        self._subset: dict[str, np.ndarray] = {}
        self._analysed: set[str] = set()

    # ---- samples and channels --------------------------------------------------------

    @property
    def sample_ids(self) -> list[str]:
        if self.ws:
            return list(self.ws.get_sample_ids())
        return [p.name for p in self.fcs]

    def sample_path(self, sid: str) -> Path:
        """The FCS file a sample id names."""
        return next(f for f in self.fcs if f.name == sid)

    def _sample(self, sid: str) -> fk.Sample:
        if sid not in self._samples:
            if self.ws:
                self._samples[sid] = self.ws.get_sample(sid)
            else:
                p = next(f for f in self.fcs if f.name == sid)
                self._samples[sid] = fk.Sample(str(p))
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

    def _transforms(self, sid: str) -> dict:
        if self.ws:
            return self.ws.get_transforms(sid)
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
            cm = self.ws.get_comp_matrix(sid) if self.ws else None
            if cm is None and "spill" in s.metadata:
                try:
                    cm = fk.Matrix(s.metadata["spill"], s.pnn_labels)
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
        if self.ws and sid not in self._analysed:
            with self.lock:
                self.ws.analyze_samples(sample_id=sid)
                self._analysed.add(sid)

    def flowjo_gates(self, sid: str) -> list[Gate]:
        if not self.ws:
            return []
        self._analyse(sid)
        df = self.ws.get_gating_results(sid).report
        gates = []
        for _, row in df.iterrows():
            path = tuple(row.gate_path)
            key = "/".join(list(path[1:]) + [row.gate_name])
            parent = "/".join(path[1:]) or None
            g = Gate(key=key, name=row.gate_name, parent=parent, source="flowjo", kind=str(row.gate_type))
            try:
                gd = self.ws.get_gate(sid, row.gate_name, gate_path=path)
                g.dims = [d.id for d in gd.dimensions]
                if hasattr(gd, "vertices"):
                    g.vertices = [[float(a), float(b)] for a, b in gd.vertices]
                    g.kind = "polygon"
                elif hasattr(gd, "dimensions") and len(gd.dimensions) == 2:
                    d0, d1 = gd.dimensions
                    if d0.min is not None or d0.max is not None:
                        lo0, hi0 = d0.min if d0.min is not None else 0, d0.max if d0.max is not None else 1
                        lo1, hi1 = d1.min if d1.min is not None else 0, d1.max if d1.max is not None else 1
                        g.vertices = [[lo0, lo1], [hi0, lo1], [hi0, hi1], [lo0, hi1]]
                        g.kind = "rect"
            except Exception:
                log.info("no drawable shape for gate %s", key, exc_info=True)
            gates.append(g)
        return gates

    def flowjo_membership(self, sid: str, key: str) -> np.ndarray:
        self._analyse(sid)
        parts = key.split("/")
        path = tuple(["root"] + parts[:-1])
        return np.asarray(self.ws.get_gate_membership(sid, parts[-1], gate_path=path), dtype=bool)


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
