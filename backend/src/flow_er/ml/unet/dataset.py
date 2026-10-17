"""Torch datasets over the per-(sample, gate) cache files (docs/cache-format.md).

Selection is by evaluation and role, read from the meta JSON next to each file. Every count
image (density, unstim) is normalised the same way: log1p(count) / max over the image. The
template channel is the 0/1 template mask. Augmentation (shared model only, training only):
drift = random shift and scaling of each display axis, applied identically to density, unstim
and mask (the template stays put); event re-subsampling = each event kept with probability p,
p drawn per item from [0.5, 1], done as binomial thinning of the counts.
"""

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from flow_er.ml import cache, images

NORMALISATION = "log1p(count) / max(log1p(count)) per image and channel; all-zero image stays 0"
UNITO_CHANNELS = ("density",)
SHARED_CHANNELS = ("density", "template", "unstim")
SHARED_1CH_CHANNELS = ("density", "template")


@dataclass(frozen=True)
class Augment:
    shift: float = 5.0          # max shift per axis, pixels (= 0.05 display units)
    scale: float = 0.1          # max relative scaling per axis, about the image centre
    keep: tuple[float, float] = (0.5, 1.0)   # fraction of events kept

    def describe(self) -> dict:
        return {"drift": {"shift_px": self.shift, "scale": self.scale,
                          "applies_to": ["density", "unstim", "mask"]},
                "resample": {"keep_fraction": list(self.keep), "how": "binomial thinning"}}


def normalise(counts: np.ndarray) -> np.ndarray:
    img = np.log1p(np.asarray(counts, np.float32))
    top = img.max()
    return img / top if top > 0 else img


@dataclass
class Item:
    path: Path      # the .npz model file
    meta: dict


def scan(gate_keys: Iterable[str] | None = None) -> list[Item]:
    """Every model file (with its meta), optionally only these gates."""
    root = cache.cache_dir()
    if gate_keys is None:
        metas = [p for p in root.glob("*/*/*.json") if not p.parts[-3].startswith("_")]
    else:
        metas = [p for g in gate_keys for p in root.glob(f"*/{cache.gate_id(g)}/*.json")]
    items = [Item(p.with_suffix(".npz"), json.loads(p.read_text())) for p in sorted(metas)]
    return [it for it in items if it.meta.get("format") == cache.FORMAT]


def select(evaluation: str, roles: Iterable[str], gate_keys: Iterable[str] | None = None,
           exclude: str | None = None, keep: Callable[[dict], bool] | None = None) -> list[Item]:
    """Model files whose sample has one of `roles` in `evaluation` (e.g. "als", "pretrain").
    `exclude` is a regex on the gate key; `keep` an extra filter on the meta."""
    roles = set(roles)
    items = [it for it in scan(gate_keys) if it.meta["roles"].get(evaluation) in roles]
    if exclude:
        items = [it for it in items if not re.search(exclude, it.meta["gate"])]
    if keep:
        items = [it for it in items if keep(it.meta)]
    return items


def gate_keys_of(items: Iterable[Item]) -> list[str]:
    return sorted({it.meta["gate"] for it in items})


def individuals_of(items: Iterable[Item]) -> list[str]:
    return sorted({it.meta["individual"] for it in items})


def _axis_map(rng: np.random.Generator, aug: Augment) -> np.ndarray:
    """Source pixel of each output pixel along one axis: out = c + s * (src - c) + t."""
    n, c = images.BINS, (images.BINS - 1) / 2
    s = 1 + rng.uniform(-aug.scale, aug.scale)
    t = rng.uniform(-aug.shift, aug.shift)
    return np.rint(c + (np.arange(n) - c - t) / s).astype(int)


def _remap(img: np.ndarray, sy: np.ndarray, sx: np.ndarray, fill_edge: bool) -> np.ndarray:
    n = images.BINS
    out = img[np.clip(sy, 0, n - 1)][:, np.clip(sx, 0, n - 1)]
    if not fill_edge:
        out = out.copy()
        out[(sy < 0) | (sy >= n)] = 0
        out[:, (sx < 0) | (sx >= n)] = 0
    return out


class GateDataset(Dataset):
    """(input [C, 101, 101], target [1, 101, 101]) per model file. Channels are any of
    "density", "template", "unstim" (zeros when the file has none)."""

    def __init__(self, items: list[Item], channels: Iterable[str] = UNITO_CHANNELS,
                 augment: Augment | None = None, seed: int = 0):
        self.items = list(items)
        self.channels = tuple(channels)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.items)

    def arrays(self, i: int) -> dict[str, np.ndarray]:
        with np.load(self.items[i].path, allow_pickle=False) as z:
            out = {k: z[k] for k in ("density", "mask", "template") if k in z.files}
            out["unstim"] = (z["unstim"] if "unstim" in z.files
                             else np.zeros_like(out["density"]))
        return out

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.arrays(i)
        density, unstim, mask = a["density"], a["unstim"], a["mask"].astype(np.float32)
        if self.augment is not None:
            rng = self.rng
            p = rng.uniform(*self.augment.keep)
            density = rng.binomial(np.rint(density).astype(np.int64), p).astype(np.float32)
            unstim = rng.binomial(np.rint(unstim).astype(np.int64), p).astype(np.float32)
            sy, sx = _axis_map(rng, self.augment), _axis_map(rng, self.augment)
            density = _remap(density, sy, sx, fill_edge=False)
            unstim = _remap(unstim, sy, sx, fill_edge=False)
            mask = _remap(mask, sy, sx, fill_edge=True)
        planes = {"density": normalise(density), "unstim": normalise(unstim),
                  "template": a["template"].astype(np.float32)}
        x = np.stack([planes[c] for c in self.channels])
        return torch.from_numpy(x), torch.from_numpy(mask[None].astype(np.float32))

    def inputs(self, i: int) -> torch.Tensor:
        """The un-augmented input of item i (for inference)."""
        aug, self.augment = self.augment, None
        try:
            return self[i][0]
        finally:
            self.augment = aug
