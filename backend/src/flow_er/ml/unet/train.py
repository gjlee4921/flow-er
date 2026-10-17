"""Training: BCEWithLogitsLoss + Adam, early stopping on the validation individuals' Dice.

Optional grid over learning rate {1e-3, 3e-4} x batch size {4, 8} (UNITO tunes only these two);
the grid point with the best validation Dice is kept. Each run saves model.pt and recipe.json to
settings.models_dir/<evaluation>/<model>/<gate_id or "shared">/. The shared model is pretrained
on the pool (evaluation "pretrain") and fine-tuned from that checkpoint (`init`).
"""

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from flow_er.config import settings
from flow_er.ml import cache, images
from flow_er.ml.unet import dataset as ds
from flow_er.ml.unet.model import build

GRID_LRS, GRID_BATCHES = (1e-3, 3e-4), (4, 8)
GRID = [(lr, b) for lr in GRID_LRS for b in GRID_BATCHES]
DEFAULT = (1e-3, 8)
SPLIT_VERSION = "split_v1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    evaluation: str
    model: str                          # directory name, e.g. "unito", "shared_finetuned"
    arch: dict
    channels: tuple[str, ...]
    train_roles: tuple[str, ...]
    val_roles: tuple[str, ...]
    gate_keys: list[str] | None = None  # None = every gate in the selected files
    exclude: str | None = None          # regex on gate keys to leave out
    grid: bool = False
    lrs: tuple[float, ...] | None = None      # override the learning rates (grid or fixed)
    batches: tuple[int, ...] | None = None    # override the batch sizes (grid or fixed)
    epochs: int = 100
    patience: int = 10
    max_seconds: float | None = None    # per grid point
    augment: ds.Augment | None = None
    balance_datasets: bool = False      # weight datasets equally per epoch
    init: str | None = None             # model dir to start from (fine-tuning)
    threads: int | None = None
    seed: int = 0
    out_name: str = "shared"            # last path part; per-gate runs use the gate_id
    extra: dict = field(default_factory=dict)


def points(cfg: Config) -> list[tuple[float, int]]:
    """(lr, batch) settings to train: the grid or the default, with either list overridden."""
    lrs = cfg.lrs or (GRID_LRS if cfg.grid else (DEFAULT[0],))
    batches = cfg.batches or (GRID_BATCHES if cfg.grid else (DEFAULT[1],))
    return [(lr, b) for lr in lrs for b in batches]


def model_dir(evaluation: str, model: str, name: str) -> Path:
    return settings.models_dir / evaluation / model / name


def dice_per_image(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Dice of sigmoid > 0.5 vs the expert mask per image; 1 when both are empty."""
    pred = (logits > 0).float().flatten(1)
    tgt = (target > 0.5).float().flatten(1)
    inter = (pred * tgt).sum(1)
    total = pred.sum(1) + tgt.sum(1)
    return torch.where(total > 0, 2 * inter / total.clamp(min=1), torch.ones_like(total))


@torch.no_grad()
def evaluate(model: nn.Module, data: ds.GateDataset, batch: int = 16) -> float:
    model.eval()
    dev = next(model.parameters()).device
    scores = []
    for x, y in DataLoader(data, batch_size=batch):
        scores.append(dice_per_image(model(x.to(dev)), y.to(dev)))
    return float(torch.cat(scores).mean()) if scores else float("nan")


def _loader(data: ds.GateDataset, batch: int, balance: bool, seed: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    drop_last = len(data) % batch == 1          # BatchNorm cannot train on a batch of one
    if balance:
        names = [it.meta["dataset"] for it in data.items]
        counts = {n: names.count(n) for n in set(names)}
        w = torch.tensor([1.0 / counts[n] for n in names], dtype=torch.double)
        sampler = WeightedRandomSampler(w, len(names), replacement=True, generator=g)
        return DataLoader(data, batch_size=batch, sampler=sampler, drop_last=drop_last)
    return DataLoader(data, batch_size=batch, shuffle=True, generator=g, drop_last=drop_last)


def fit(model: nn.Module, train: ds.GateDataset, val: ds.GateDataset, lr: float, batch: int,
        epochs: int, patience: int, max_seconds: float | None = None, balance: bool = False,
        seed: int = 0, log=print) -> dict:
    """Train in place; returns the best state (by val Dice) and the per-epoch history."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = _loader(train, batch, balance, seed)
    dev = next(model.parameters()).device
    best, best_epoch, best_state, history, stale = -1.0, -1, None, [], 0
    start = time.monotonic()
    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        score = evaluate(model, val)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_dice": score})
        log(f"  epoch {epoch:3d} loss {history[-1]['loss']:.4f} val Dice {score:.4f}")
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= patience or (max_seconds and time.monotonic() - start > max_seconds):
            break
    return {"state": best_state, "best_val_dice": best, "best_epoch": best_epoch,
            "history": history, "seconds": time.monotonic() - start}


def _new_model(cfg: Config) -> nn.Module:
    model = build(cfg.arch)
    if cfg.init:
        init = Path(cfg.init)
        recipe = json.loads((init / "recipe.json").read_text())
        if tuple(recipe["channels"]) != tuple(cfg.channels):
            raise ValueError(f"{init}: channels {recipe['channels']} != {list(cfg.channels)}")
        model.load_state_dict(torch.load(init / "model.pt", map_location="cpu"))
    return model.to(DEVICE)


def train(cfg: Config, log=print) -> Path:
    """Select the files, train (grid or fixed), save the best model and its recipe."""
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    train_items = ds.select(cfg.evaluation, cfg.train_roles, cfg.gate_keys, cfg.exclude)
    val_items = ds.select(cfg.evaluation, cfg.val_roles, cfg.gate_keys, cfg.exclude)
    if not train_items or not val_items:
        raise ValueError(f"{cfg.evaluation}: {len(train_items)} training and {len(val_items)} "
                         f"validation files for {cfg.gate_keys or 'all gates'}")
    train_ds = ds.GateDataset(train_items, cfg.channels, cfg.augment, seed=cfg.seed)
    val_ds = ds.GateDataset(val_items, cfg.channels)
    log(f"{cfg.evaluation}/{cfg.model}/{cfg.out_name}: {len(train_items)} train, "
        f"{len(val_items)} val files")

    runs = []
    for lr, batch in points(cfg):
        log(f" lr {lr:g} batch {batch}")
        model = _new_model(cfg)
        r = fit(model, train_ds, val_ds, lr, batch, cfg.epochs, cfg.patience, cfg.max_seconds,
                cfg.balance_datasets, cfg.seed, log)
        runs.append({"lr": lr, "batch": batch, **r})
    best = max(runs, key=lambda r: r["best_val_dice"])

    out = model_dir(cfg.evaluation, cfg.model, cfg.out_name)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], out / "model.pt")
    recipe = {
        "format": "unet-recipe-v1", "created": datetime.now().isoformat(timespec="seconds"),
        "model": cfg.model, "arch": cfg.arch, "channels": list(cfg.channels),
        "image_recipe": images.RECIPE, "recipe_id": images.recipe_id(),
        "transform": {k: images.RECIPE[k] for k in ("scatter", "fluorescence", "pixel")},
        "normalisation": ds.NORMALISATION,
        "augmentation": cfg.augment.describe() if cfg.augment else None,
        "gate_keys": ds.gate_keys_of(train_items), "gate_ids": sorted(
            {cache.gate_id(g) for g in ds.gate_keys_of(train_items)}),
        "exclude": cfg.exclude, "split": SPLIT_VERSION, "evaluation": cfg.evaluation,
        "train_roles": list(cfg.train_roles), "val_roles": list(cfg.val_roles),
        "train_individuals": ds.individuals_of(train_items),
        "val_individuals": ds.individuals_of(val_items),
        "n_train": len(train_items), "n_val": len(val_items),
        "init": str(cfg.init) if cfg.init else None,
        "hyperparameters": {"lr": best["lr"], "batch": best["batch"], "max_epochs": cfg.epochs,
                            "patience": cfg.patience, "max_seconds": cfg.max_seconds,
                            "grid": cfg.grid, "seed": cfg.seed, "threads": cfg.threads, "device": str(DEVICE),
                            "balance_datasets": cfg.balance_datasets, "loss": "BCEWithLogitsLoss",
                            "optimiser": "Adam"},
        "grid": [{k: r[k] for k in ("lr", "batch", "best_val_dice", "best_epoch", "seconds")}
                 for r in runs],
        "val_dice": [h["val_dice"] for h in best["history"]],
        "history": best["history"], "best_val_dice": best["best_val_dice"],
        "best_epoch": best["best_epoch"],
        "training_seconds": sum(r["seconds"] for r in runs), **cfg.extra,
    }
    (out / "recipe.json").write_text(json.dumps(recipe, indent=1, default=_json_default))
    log(f" best: lr {best['lr']:g} batch {best['batch']} val Dice {best['best_val_dice']:.4f} "
        f"-> {out}")
    return out


def _json_default(o):
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    raise TypeError(type(o))
