"""End to end on a synthetic cache: dataset -> train -> saved recipe -> infer -> predictions.

Each synthetic sample is two Gaussian blobs of cells; the expert gate is a disc on one blob.
"""

import json
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flow_er.config import settings
from flow_er.ml import cache
from flow_er.ml.unet import dataset as ds
from flow_er.ml.unet import infer, train
from flow_er.ml.unet.model import shared_arch, unito_arch

GATE = "Lymphocytes/CD4 Single Positive/IFNy FITC +"
ROLES = {"train": range(24), "val": range(24, 27), "test": range(27, 30)}


def disc(cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[0:101, 0:101]
    return (np.hypot(xx - cx, yy - cy) <= r).astype(np.uint8)


def write_sample(i: int, role: str, rng: np.random.Generator, unstim: bool) -> None:
    a = np.array([30, 35]) + rng.normal(0, 4, 2)
    b = np.array([68, 65]) + rng.normal(0, 4, 2)
    pts = np.vstack([rng.normal(a, 5, (1500, 2)), rng.normal(b, 6, (2500, 2))])
    bins = np.clip(np.rint(pts), 0, 100).astype(np.uint8)
    density = np.zeros((101, 101), np.float32)
    np.add.at(density, (bins[:, 1], bins[:, 0]), 1)
    mask = disc(*a, 14)
    arrays = {"density": density, "mask": mask, "template": disc(30, 35, 14),
              "cell_bins": bins, "cell_label": mask[bins[:, 1], bins[:, 0]].astype(bool),
              "cell_index": np.arange(len(bins), dtype=np.uint32)}
    if unstim:
        arrays["unstim"] = density * 0.5
    sid = f"s{i:02d}"
    meta = {"format": cache.FORMAT, "dataset": "synth", "sample": f"synth/ws/{sid}.fcs",
            "sample_id": sid, "individual": f"human:{i}", "gate": GATE,
            "gate_id": cache.gate_id(GATE), "roles": {"als": role, "pretrain": None}}
    out = cache.gate_path({"dataset": "synth", "file": f"{sid}.fcs"}, GATE)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    out.with_suffix(".json").write_text(json.dumps(meta))


@pytest.fixture
def synth_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    for role, idx in ROLES.items():
        for i in idx:
            write_sample(i, role, rng, unstim=i % 2 == 0)
    return tmp_path


def test_select_by_evaluation_and_role(synth_cache):
    assert len(ds.select("als", ["train"])) == 24
    assert len(ds.select("als", ["val", "test"], [GATE])) == 6
    assert ds.select("pretrain", ["train"]) == []
    x, y = ds.GateDataset(ds.select("als", ["train"]), ds.SHARED_CHANNELS)[1]
    assert x.shape == (3, 101, 101) and y.shape == (1, 101, 101)
    assert float(x[0].max()) == 1.0 and float(x[2].abs().sum()) == 0.0   # no unstim -> zeros


def test_drift_moves_density_and_mask_together(synth_cache):
    items = ds.select("als", ["train"])
    data = ds.GateDataset(items, ds.SHARED_CHANNELS, ds.Augment(shift=8, scale=0.2), seed=1)
    for i in range(len(items)):
        x, y = data[i]
        clean_x, _ = ds.GateDataset(items, ds.SHARED_CHANNELS)[i]
        inside = x[0][y[0] > 0].sum() / x[0].sum()
        assert inside > 0.2                                  # the gated blob is still gated
        assert torch.equal(x[1], clean_x[1])                 # the template does not move


def test_unito_learns_two_blobs_and_writes_predictions(synth_cache):
    start = time.monotonic()
    cfg = train.Config(evaluation="als", model="unito", arch=unito_arch((16, 32, 64, 128), 256),
                       channels=ds.UNITO_CHANNELS, train_roles=("train",), val_roles=("val",),
                       gate_keys=[GATE], epochs=200, patience=200, max_seconds=30,
                       out_name=cache.gate_id(GATE), threads=2)
    out = train.train(cfg, log=lambda *_: None)
    assert time.monotonic() - start < 60
    recipe = json.loads((out / "recipe.json").read_text())
    assert recipe["best_val_dice"] > 0.9
    assert recipe["val_individuals"] == ["human:24", "human:25", "human:26"]
    assert recipe["normalisation"] == ds.NORMALISATION and recipe["split"] == "split_v1"

    written = infer.predict_roles("als", "unito", ["test"], per_gate_root=out.parent,
                                  log=lambda *_: None)
    assert len(written) == 3
    for path in written:
        sid = path.stem
        truth = np.load(cache.gate_path({"dataset": "synth", "file": f"{sid}.fcs"}, GATE))
        pred = np.load(path)
        assert pred["pred"].dtype == bool and pred["pred"].shape == truth["cell_label"].shape
        assert (pred["pred"] == truth["cell_label"]).mean() > 0.95
        assert np.array_equal(pred["pred"], pred["prob"] > 0.5)


def test_shared_pretrain_finetune_and_1ch(synth_cache):
    pytest.importorskip("segmentation_models_pytorch")
    base = {"evaluation": "als", "train_roles": ("train",), "val_roles": ("val",),
            "epochs": 1, "patience": 1, "threads": 2}
    pre = train.train(train.Config(model="shared", arch=shared_arch(3, None),
                                   channels=ds.SHARED_CHANNELS, augment=ds.Augment(),
                                   balance_datasets=True, **base), log=lambda *_: None)
    fine = train.train(train.Config(model="shared_finetuned", arch=shared_arch(3, None),
                                    channels=ds.SHARED_CHANNELS, init=str(pre), **base),
                       log=lambda *_: None)
    assert json.loads((fine / "recipe.json").read_text())["init"] == str(pre)
    assert len(infer.predict_roles("als", "shared_finetuned", ["test"], model_dir=fine,
                                   log=lambda *_: None)) == 3
    one = train.train(train.Config(model="shared_finetuned_1ch", arch=shared_arch(2, None),
                                   channels=ds.SHARED_1CH_CHANNELS, **base), log=lambda *_: None)
    _, recipe = infer.load(one)
    assert recipe["channels"] == ["density", "template"]
    with pytest.raises(ValueError):          # channels must match the checkpoint
        train.train(train.Config(model="x", arch=shared_arch(2, None),
                                 channels=ds.SHARED_1CH_CHANNELS, init=str(pre), **base),
                    log=lambda *_: None)
