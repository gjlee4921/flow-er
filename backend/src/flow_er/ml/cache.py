"""Where cached samples live and how to read them back (format: docs/cache-format.md).

Two layers, both written by scripts/build_dataset.py:
- per-sample source files (_samples/...): compensated values, FlowKit labels and gate geometry,
  used by the baselines and to derive the model files;
- per-(sample, gate) model files: density, expert mask, template, unstimulated-well density and
  the parent cells' pixels and labels, the only thing models read.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flow_er.config import settings
from flow_er.ml import images

FORMAT = "cache-v1"


def cache_dir() -> Path:
    return settings.data_dir / "_cache" / "v1"


def sample_id(sample: dict) -> str:
    return Path(sample["file"]).stem


def gate_id(gate_key: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", gate_key.split("/")[-1]).strip("_")
    return f"{slug}__{hashlib.sha256(gate_key.encode()).hexdigest()[:8]}"


def sample_path(sample: dict) -> Path:
    return cache_dir() / "_samples" / sample["dataset"] / f"{sample_id(sample)}.npz"


def gate_path(sample: dict, gate_key: str) -> Path:
    return cache_dir() / sample["dataset"] / gate_id(gate_key) / f"{sample_id(sample)}.npz"


def results_path(evaluation: str, method: str, sample: dict, gate_key: str) -> Path:
    return (settings.data_dir / "_results" / evaluation / method / sample["dataset"]
            / gate_id(gate_key) / f"{sample_id(sample)}.npz")


@dataclass
class Cached:
    """One per-sample source file: compensated values of the gated channels, FlowKit membership
    per gate, and per gate its parent, its two channel columns, the parent density and mask."""
    key: str
    raw: np.ndarray            # (n events, channels) compensated values
    columns: list[str]         # channel per column ("|raw" = uncompensated dimension)
    pnr: np.ndarray            # $PnR per column
    labels: np.ndarray         # (gates, n events) bool, FlowKit = truth
    parent: np.ndarray         # (gates,) index of the parent gate, -1 = all events
    dims: np.ndarray           # (gates, 2) column of the gate's x and y channel
    images: np.ndarray         # (gates, 101, 101) parent-population counts
    masks: np.ndarray          # (gates, 101, 101) expert mask
    gate_keys: list[str]
    meta: dict

    @property
    def channels(self) -> list[str]:
        return [c.split("|")[0] for c in self.columns]

    def gate(self, key: str) -> int:
        return self.gate_keys.index(key)

    def in_parent(self, g: int) -> np.ndarray:
        p = self.parent[g]
        return self.labels[p] if p >= 0 else np.ones(self.labels.shape[1], bool)

    def gate_raw(self, g: int, cells: np.ndarray | None = None) -> np.ndarray:
        x, y = self.dims[g]
        raw = self.raw[:, [x, y]]
        return raw if cells is None else raw[cells]

    def gate_channels(self, g: int) -> tuple[list[str], list[float]]:
        x, y = self.dims[g]
        return [self.channels[x], self.channels[y]], [float(self.pnr[x]), float(self.pnr[y])]

    def gate_display(self, g: int, cells: np.ndarray | None = None) -> np.ndarray:
        chans, pnrs = self.gate_channels(g)
        return images.to_display(self.gate_raw(g, cells), chans, pnrs)


def load_sample(sample: dict) -> Cached:
    with np.load(sample_path(sample), allow_pickle=False) as z:
        return Cached(key=f"{sample['dataset']}/{sample['workspace']}/{sample['sample']}",
                      raw=z["raw"], columns=list(z["columns"]), pnr=z["pnr"], labels=z["labels"],
                      parent=z["parent"], dims=z["dims"], images=z["images"], masks=z["masks"],
                      gate_keys=list(z["gate_keys"]), meta=json.loads(str(z["meta"])))


def load_gate(sample: dict, gate_key: str) -> tuple[dict, dict]:
    """(arrays, meta) of one per-(sample, gate) model file."""
    path = gate_path(sample, gate_key)
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return arrays, json.loads(path.with_suffix(".json").read_text())
