"""Runtime on a synthetic FCS file: cascade, template copy, a model, edits, GatingML, router."""

import json

import flowio
import numpy as np
import pytest

from flow_er.config import settings
from flow_er.cytometry.export import gatingml_ids, to_gatingml
from flow_er.cytometry.io import FcsFile
from flow_er.ml import images, registry
from flow_er.ml.gatetree import GateSpec, GateTree, inside
from flow_er.ml.runtime import apply_polygon, autogate

CHANNELS = ["FSC-A", "SSC-A", "FL1-A"]


def box(x0, x1, y0, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


@pytest.fixture
def fcs(tmp_path):
    """Two scatter blobs; in blob A a third of the cells are FL1-bright."""
    rng = np.random.default_rng(1)
    a = np.column_stack([rng.normal(80_000, 6_000, 3000), rng.normal(40_000, 4_000, 3000),
                         np.where(rng.random(3000) < 1 / 3, rng.normal(20_000, 2_000, 3000),
                                  rng.normal(100, 50, 3000))])
    b = np.column_stack([rng.normal(180_000, 8_000, 2000), rng.normal(150_000, 8_000, 2000),
                         rng.normal(100, 50, 2000)])
    events = np.vstack([a, b]).astype(np.float32)
    path = tmp_path / "synthetic.fcs"
    with open(path, "wb") as fh:
        flowio.create_fcs(fh, events.ravel().tolist(), CHANNELS)
    return path


@pytest.fixture
def tree():
    return GateTree([
        GateSpec("Cells", "FSC-A", "SSC-A", (False, False), box(0.2, 0.45, 0.05, 0.25)),
        GateSpec("Cells/FL1+", "FL1-A", "SSC-A", (False, False), box(0.55, 0.99, 0.0, 0.5),
                 is_cytokine=True),
    ])


@pytest.fixture(autouse=True)
def models(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    registry._cache.clear()


def expected(fcs_path, tree):
    f = FcsFile(fcs_path)
    ev = f.events(np.arange(f.n_events))
    out = {}
    parent = np.ones(f.n_events, bool)
    for g in tree.gates:
        disp = images.to_display(ev[:, [CHANNELS.index(g.x), CHANNELS.index(g.y)]], [g.x, g.y],
                                 [f.pnr[g.x], f.pnr[g.y]])
        p = out[g.parent] if g.parent else parent
        out[g.key] = p & inside(g.template_array(), disp)
    return out


def test_template_copy_cascade(fcs, tree):
    res = autogate(fcs, tree)
    want = expected(fcs, tree)
    assert res.gates["Cells"].model == registry.TEMPLATE_COPY
    for key, g in res.gates.items():
        assert np.array_equal(g.membership, want[key])
    cells, pos = res.gates["Cells"], res.gates["Cells/FL1+"]
    assert 2900 < cells.count < 3100 and pos.parent_count == cells.count
    assert 30 < pos.percent_of_parent < 37
    js = res.to_json(plot=True)
    assert js["gates"][1]["plot"]["bins"] == 101 and len(js["gates"][1]["polygon"]["raw"]) == 4
    json.dumps(js)                                   # serialisable


class FakeModel:
    """Predicts the template, slightly shrunk: exercises the model path without weights."""
    id = "fake/model"

    def predict(self, density, template, unstim=None):
        from scipy import ndimage
        self.unstim = unstim
        return ndimage.binary_erosion(template, iterations=1).astype(float) * 0.9


def test_model_path_and_confidence(fcs, tree, monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(registry, "route", lambda lab, key, qc=False: fake)
    res = autogate(fcs, tree, lab="demo", unstim=fcs)
    g = res.gates["Cells"]
    assert g.model == "fake/model" and g.prob is not None and 0 <= g.confidence <= 1
    assert fake.unstim is not None                   # the cytokine gate got the unstimulated well
    template = tree["Cells"].template_array()
    assert g.polygon_display.min(0)[0] > template.min(0)[0]     # the model's (shrunk) polygon
    assert np.array_equal(g.membership, inside(g.polygon_display, images.to_display(
        FcsFile(fcs).events(np.arange(5000))[:, :2], CHANNELS[:2],
        [FcsFile(fcs).pnr[c] for c in CHANNELS[:2]])))


def test_edit_recomputes_children(fcs, tree):
    res = autogate(fcs, tree)
    before = res.gates["Cells/FL1+"].parent_count
    res = apply_polygon(res, tree, "Cells", np.array(box(0.2, 0.33, 0.05, 0.25)))
    assert res.gates["Cells"].model == "user"
    assert res.gates["Cells/FL1+"].parent_count == res.gates["Cells"].count < before


def test_gatingml_round_trip(fcs, tree, tmp_path):
    import flowkit as fk
    res = autogate(fcs, tree)
    path = to_gatingml(res, tree, tmp_path / "gates.xml")
    gs = fk.parse_gating_xml(str(path))
    r = gs.gate_sample(fk.Sample(str(fcs)))
    ids = gatingml_ids(tree)
    for g in tree.gates:
        gate_id = next(i for i in gs.get_gate_ids() if i[0] == ids[g.key])
        assert np.array_equal(r.get_gate_membership(*gate_id), res.gates[g.key].membership)


def tiny_model(path, channels):
    import torch

    from flow_er.ml.unet.model import build, unito_arch
    arch = {**unito_arch((4, 8), 16), "in_channels": len(channels)}
    path.mkdir(parents=True)
    torch.save(build(arch).state_dict(), path / "model.pt")
    (path / "recipe.json").write_text(json.dumps({
        "arch": arch, "channels": channels, "recipe_id": images.recipe_id()}))
    return path


def test_router_order(tmp_path):
    root = settings.models_dir
    assert registry.route("lab1", "Cells") is None                       # template copy
    tiny_model(root / "base" / "shared", ["density", "template", "unstim"])
    assert registry.route("lab1", "Cells").id == "base/shared"
    shared = tiny_model(root / "elsewhere" / "ft", ["density", "template", "unstim"])
    gate = tiny_model(root / "elsewhere" / "cells", ["density"])
    v1 = registry.register_version("lab1", shared=shared, make_current=True)
    assert registry.route("lab1", "Cells").id == f"lab1/{v1}/shared"
    v2 = registry.register_version("lab1", gates={"Cells": gate}, shared=shared,
                                   make_current=True)
    assert registry.route("lab1", "Cells").id.startswith(f"lab1/{v2}/unito/")
    assert registry.route("lab1", "Cells/FL1+").id == f"lab1/{v2}/shared"
    assert registry.route("lab1", "Cells", qc=True) is None              # QC gates: template
    registry.set_current("lab1", v1)                                     # rollback
    assert registry.route("lab1", "Cells").id == f"lab1/{v1}/shared"
    assert [v["current"] for v in registry.list_versions("lab1")] == [True, False]
    prob = registry.route("lab1", "Cells").predict(np.ones((101, 101)), np.zeros((101, 101)))
    assert prob.shape == (101, 101) and ((0 <= prob) & (prob <= 1)).all()


def test_gate_drawn_to_the_axis_keeps_the_cells_beyond_it():
    from flow_er.ml.gatetree import inside
    from flow_er.ml.unet.polygon import EDGE_OUT, prob_to_polygon
    prob = np.zeros((101, 101))
    prob[40:80, :30] = 1                                   # a gate touching the left edge
    poly = prob_to_polygon(prob)
    assert poly[:, 0].min() == EDGE_OUT[0]
    cells = np.array([[-0.02, 0.6], [0.0, 0.6], [0.1, 0.6], [0.5, 0.6]])  # below / on the axis
    assert inside(poly, cells).tolist() == [True, True, True, False]
