"""Iterative fine-tuning on synthetic donors: labels, held-out split, retrain, promotion, rollback."""

import json

import flowio
import numpy as np
import pytest

from flow_er.config import settings
from flow_er.ml import finetune, registry
from flow_er.ml.gatetree import GateSpec, GateTree, inside
from flow_er.ml.unet.model import unito_arch

CHANNELS = ["FSC-A", "SSC-A"]
TINY = unito_arch((4, 8), 16)


def box(x0, x1, y0, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


@pytest.fixture(autouse=True)
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    registry._cache.clear()


def donor_fcs(path, shift):
    """One blob whose position moves with `shift` (display units ~ shift / 262144)."""
    rng = np.random.default_rng(int(shift))
    ev = np.column_stack([rng.normal(80_000 + shift, 6_000, 2000),
                          rng.normal(40_000, 4_000, 2000)]).astype(np.float32)
    with open(path, "wb") as fh:
        flowio.create_fcs(fh, ev.ravel().tolist(), CHANNELS)
    return path


TREE = GateTree([GateSpec("Cells", "FSC-A", "SSC-A", (False, False), box(0.2, 0.42, 0.05, 0.25))])


def donors(n_val=1, n_train=3):
    """Individual names giving n_val held-out and n_train training individuals."""
    val, train, i = [], [], 0
    while len(val) < n_val or len(train) < n_train:
        name = f"donor-{i}"
        (val if finetune.role_of(name) == "val" else train).append(name)
        i += 1
    return val[:n_val] + train[:n_train]


def test_role_of_is_stable_and_about_a_fifth():
    roles = [finetune.role_of(f"x{i}") for i in range(2000)]
    assert roles == [finetune.role_of(f"x{i}") for i in range(2000)]
    assert 0.15 < roles.count("val") / len(roles) < 0.25


def test_promotion_decision():
    assert finetune.promotion_decision({"a": 2.0, "b": 1.0}, {"a": 1.0, "b": 1.5})[0]
    ok, why = finetune.promotion_decision({"a": 2.0, "b": 1.0}, {"a": 0.1, "b": 2.5})
    assert not ok and "worse" in why                       # mean improves but b is 1.5 pp worse
    assert not finetune.promotion_decision({"a": 1.0}, {"a": 1.0})[0]
    assert not finetune.promotion_decision({}, {})[0]


def test_add_labels_writes_cache_examples(tmp_path):
    f = donor_fcs(tmp_path / "s.fcs", 30_000)
    accepted = {"Cells": np.array(box(0.3, 0.5, 0.05, 0.25))}
    [path] = finetune.add_labels("lab1", f, TREE, accepted, "donor-a", "s1")
    meta = json.loads(path.with_suffix(".json").read_text())
    z = np.load(path)
    assert meta["dataset"] == "lab_lab1" and meta["roles"] == {"lab_lab1":
                                                               finetune.role_of("donor-a")}
    assert z["mask"].sum() > 0 and z["template"].sum() > 0
    assert not np.array_equal(z["mask"], z["template"])     # the accepted gate, not the template
    disp = z["cell_bins"] / 100.0
    assert (z["cell_label"] == inside(accepted["Cells"], disp)).mean() > 0.97


def _label(tmp_path, names, start=0):
    for i, name in enumerate(names, start):
        shift = 25_000 + 2_000 * i                     # every donor drifted right of the template
        f = donor_fcs(tmp_path / f"{name}.fcs", shift)
        accepted = {"Cells": np.array(box(0.2 + shift / 262144, 0.42 + shift / 262144,
                                          0.05, 0.25))}
        finetune.add_labels("lab1", f, TREE, accepted, name, name)


def test_retrain_trains_per_gate_unito_only_and_rolls_back(tmp_path):
    people = donors(n_val=1, n_train=4)
    _label(tmp_path, people[:4])
    # a gate without a model needs `graduate` accepted samples before it gets one
    out = finetune.retrain("lab1", epochs=3, patience=2, unito_arch=TINY, graduate=99, log=lambda *a: None)
    assert out["gates"]["Cells"]["kept"] is False and "needs 99" in out["gates"]["Cells"]["why"]
    out = finetune.retrain("lab1", epochs=3, patience=2, unito_arch=TINY, graduate=3, log=lambda *a: None)
    v = {x["version"]: x for x in registry.list_versions("lab1")}[out["version"]]
    assert v["shared"] is None                             # no shared model is trained
    assert v["metrics"]["promoted"] == out["promoted"] and v["training_samples"]
    d = out["gates"]["Cells"]
    assert "template_copy_pp" in d and d["warm_start"] is False
    assert d["kept"] == (d["held_out_error_pp"] < d["template_copy_pp"])   # must beat the baseline
    assert v["current"] == out["promoted"]
    if out["promoted"] and d["kept"]:
        assert registry.route("lab1", "Cells").id.startswith(f"lab1/{out['version']}/unito/")
    # nothing new since the current version: nothing to retrain, no version written
    n = len(registry.list_versions("lab1"))
    if out["promoted"]:
        again = finetune.retrain("lab1", epochs=2, patience=1, unito_arch=TINY, graduate=3, log=lambda *a: None)
        assert again["version"] is None and len(registry.list_versions("lab1")) == n
    # a new donor's labels: only that gate (and those below it) is retrained, warm-started
    _label(tmp_path, people[4:], start=4)
    out3 = finetune.retrain("lab1", epochs=2, patience=1, unito_arch=TINY, graduate=3, log=lambda *a: None)
    assert set(out3["gates"]) == {"Cells"}
    if out["promoted"] and d["kept"]:
        assert out3["gates"]["Cells"]["warm_start"] is True
    registry.set_current("lab1", out["version"])           # rollback
    assert registry.current_version("lab1") == out["version"]


def test_gates_to_retrain_follows_new_samples_and_children():
    from types import SimpleNamespace as N
    it = lambda g, p, s: N(meta={"gate": g, "parent": p, "sample": s})
    items = [it("A", None, "s1"), it("A/B", "A", "s1"), it("A/B/C", "A/B", "s1"), it("D", None, "s1"),
             it("A/B", "A", "s2")]
    assert finetune.gates_to_retrain(items, {"s1"}) == {"A/B", "A/B/C"}        # s2 is new for A/B
    assert finetune.gates_to_retrain(items, {"s1", "s2"}) == set()
    assert finetune.gates_to_retrain(items, {"s1", "s2"}, gates=["D"]) == {"D"}


def test_small_lab_still_holds_out_one_individual():
    from flow_er.ml.finetune import held_out, role_of
    people = {f"donor-{i}" for i in range(40)}
    in_share = {p for p in people if role_of(p) == "val"}
    assert held_out(people) == in_share and in_share          # normal labs: the hash share
    few = sorted(people - in_share)[:3]
    assert len(held_out(set(few))) == 1                       # none in the share: fall back to one
    assert held_out({few[0]}) == set()                        # one person: nothing to hold out
