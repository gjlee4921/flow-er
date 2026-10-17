"""Predict gates: model + recipe -> probability map -> per-cell membership.

Each cell takes the value of the pixel it falls in: prob[i] = map[cell_bins[i, 1], cell_bins[i, 0]],
pred = prob > 0.5. Predictions go to data/_results/<evaluation>/<method>/... (docs/cache-format.md).
Methods: unito, shared_zero_shot, shared_finetuned, shared_finetuned_1ch.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flow_er.ml import cache, images
from flow_er.ml.unet import dataset as ds
from flow_er.ml.unet.model import build


def load(model_dir: Path) -> tuple[nn.Module, dict]:
    recipe = json.loads((Path(model_dir) / "recipe.json").read_text())
    if recipe["recipe_id"] != images.recipe_id():
        raise ValueError(f"{model_dir}: image recipe {recipe['recipe_id']} != current "
                         f"{images.recipe_id()}")
    model = build({**recipe["arch"], "encoder_weights": None})
    model.load_state_dict(torch.load(Path(model_dir) / "model.pt", map_location="cpu"))
    return model.eval(), recipe


@torch.no_grad()
def probability_maps(model: nn.Module, recipe: dict, items: list[ds.Item],
                     batch: int = 16) -> Iterator[np.ndarray]:
    """(101, 101) sigmoid probability map of each model file, in order."""
    data = ds.GateDataset(items, recipe["channels"])
    for start in range(0, len(data), batch):
        x = torch.stack([data.inputs(i) for i in range(start, min(start + batch, len(data)))])
        yield from torch.sigmoid(model(x))[:, 0].numpy()


def cell_probabilities(prob_map: np.ndarray, cell_bins: np.ndarray) -> np.ndarray:
    bins = cell_bins.astype(np.intp)
    return prob_map[bins[:, 1], bins[:, 0]].astype(np.float32)


def result_path(evaluation: str, method: str, meta: dict) -> Path:
    sample = {"dataset": meta["dataset"], "file": f"{meta['sample_id']}.fcs"}
    return cache.results_path(evaluation, method, sample, meta["gate"])


def predict(model_dir: Path, items: list[ds.Item], evaluation: str, method: str) -> list[Path]:
    """Write pred (and prob) for every item; returns the written paths."""
    model, recipe = load(model_dir)
    written = []
    for item, pmap in zip(items, probability_maps(model, recipe, items)):
        with np.load(item.path, allow_pickle=False) as z:
            prob = cell_probabilities(pmap, z["cell_bins"])
        out = result_path(evaluation, method, item.meta)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, pred=prob > 0.5, prob=prob)
        written.append(out)
    return written


def predict_roles(evaluation: str, method: str, roles: list[str], model_dir: Path | None = None,
                  per_gate_root: Path | None = None, gate_keys: list[str] | None = None,
                  exclude: str | None = None, log=print) -> list[Path]:
    """Predict every file of `roles` in `evaluation`: with one shared model (`model_dir`), or
    per gate with <per_gate_root>/<gate_id>/ (gates without a model are skipped)."""
    items = ds.select(evaluation, roles, gate_keys, exclude)
    if model_dir is not None:
        written = predict(model_dir, items, evaluation, method)
    else:
        written = []
        for gk in ds.gate_keys_of(items):
            d = per_gate_root / cache.gate_id(gk)
            if not (d / "model.pt").exists():
                log(f"  no model for {gk}, skipped")
                continue
            written += predict(d, [it for it in items if it.meta["gate"] == gk], evaluation,
                               method)
    log(f"{evaluation}/{method}: {len(written)} prediction files")
    return written
