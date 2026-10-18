"""The gating models in the app: reading the ALS screen (two FlowJo workspaces whose files repeat
internal names), who each file came from, and gating one unseen file with lab "als" against the
expert's FlowJo gates. Skipped when the ALS folder or the trained models are not on this machine
(they stay out of git)."""

import pytest

from flow_er.config import settings
from flow_er.workspace import samplemeta, store

ALS = settings.data_dir.expanduser() / "als_ics_screen4"
MODELS = settings.models_dir.expanduser() / "labs" / "als" / "template.json"
needs_als = pytest.mark.skipif(not ALS.exists(), reason=f"no ALS screen at {ALS}")
needs_models = pytest.mark.skipif(not MODELS.exists(), reason="no lab als models on this machine")


@needs_als
def test_every_workspace_is_read_and_files_are_known_by_name():
    """24 files, each gated by its own group's workspace (ALS with ALS, HC with HC), though both
    workspaces call their files "Specimen_001_C1_C01.fcs" and so on."""
    from flow_er.workspace.data import Dataset
    ds = Dataset(ALS)
    assert len(ds.sample_ids) == 24
    for sid in ds.sample_ids:
        assert sid.split()[0] in ds.workspace_for(sid).name, sid
    hc = "HC Specimen_001_C1_C01 Negative.fcs"
    assert len(ds.flowjo_gates(hc)) == 16 and ds._sample(hc).event_count != ds._sample("ALS Specimen_001_C1_C01 Negative.fcs").event_count



@needs_als
def test_one_display_for_every_file_and_flowjo_shapes_still_hold():
    """The HC and ALS workspaces scale FITC differently, and FlowJo's biex runs to about 4096, so
    the lab draws every file in the models' 0-1 display. FlowJo's own gates, moved into it, still
    hold the cells FlowJo says they hold."""
    import numpy as np
    from flow_er.workspace.data import Dataset, in_polygon
    ds = Dataset(ALS)
    assert ds.uniform
    for sid in ("HC Specimen_001_C5_C05 Peptide.fcs", "ALS Specimen_001_C8_C08 Peptide.fcs"):
        cols, ev = list(ds._sample(sid).pnn_labels), ds.events(sid)
        assert 0.1 < np.median(ev[:, cols.index("FITC-A")]) < 0.5
        for g in ds.flowjo_gates(sid):
            if len(g.dims) != 2 or not g.vertices:
                continue
            parent = ds.flowjo_membership(sid, g.parent) if g.parent else np.ones(len(ev), bool)
            mine = parent & in_polygon(ev[:, cols.index(g.dims[0])], ev[:, cols.index(g.dims[1])], g.vertices)
            truth = ds.flowjo_membership(sid, g.key)
            assert (mine ^ truth).sum() <= max(3, 0.02 * truth.sum()), (sid, g.key, mine.sum(), truth.sum())

def test_who_each_file_came_from_is_read_from_its_name():
    names = [f"{g} Specimen_001_C{n}_C{n:02d} {c}.fcs" for g in ("ALS", "HC") for n, c in
             zip(range(1, 7), ["Negative", "Peptide", "Positive"] * 2)]
    rows = {r["sample"]: r for r in samplemeta.infer(names)}
    hc5 = rows["HC Specimen_001_C5_C05 Peptide.fcs"]
    assert hc5["individual"] == "HC-2" and hc5["condition"] == "Peptide" and hc5["unstim"] == "HC Specimen_001_C4_C04 Negative.fcs"
    assert rows["ALS Specimen_001_C1_C01 Negative.fcs"]["individual"] == "ALS-1"
    plain = samplemeta.infer(["101_DEN084Y5_15_E01_008_clean.fcs", "101_DEN084Y5_15_E03_009_clean.fcs"])
    assert {r["individual"] for r in plain} == {"101_DEN084Y5_15"}


@needs_als
@needs_models
def test_lab_als_gates_an_unseen_file_like_the_expert(tmp_path, monkeypatch):
    """HC donor 2's peptide well, with its unstimulated well: 16 gates, the upstream ones (lymphocytes
    to CD4 and CD8) by the lab's U-Nets with F1 of at least 0.9 against the expert's FlowJo gates."""
    from flow_er.workspace import models
    monkeypatch.setattr(settings, "workspace_dir", tmp_path)
    exp = store.create(ALS, "ALS")["id"]
    assert models.lab_for(exp) == "als"
    body = models.autogate(exp, "HC Specimen_001_C5_C05 Peptide.fcs")
    assert len(body["gates"]) == 16 and body["unstim"] == "HC Specimen_001_C4_C04 Negative.fcs"
    upstream = [g for g in body["gates"] if g["gate"].split("/")[-1] in
                ("Lymphocytes", "Single Cells 1", "Single Cells 2", "Comp-LIVE_DEAD Aqua-A, SSC-A subset", "CD3",
                 "CD4 Single Positive", "CD8 Single Positive")]
    assert len(upstream) == 7
    for g in upstream:
        assert "/unito/" in g["model"] and g["f1"] >= 0.9, (g["gate"], g["model"], g.get("f1"))
    review = models.review(exp, "HC Specimen_001_C5_C05 Peptide.fcs")
    assert len(review["gates"]) == 15 and all(k.endswith("(model)") for k in review["gates"])


@needs_als
@needs_models
def test_raw_files_with_no_workspace_are_gated_by_the_lab_their_panel_matches(tmp_path, monkeypatch):
    """Files nobody has gated (no FlowJo workspace): channels are named as the models name them
    ("LIVE_DEAD Aqua-A"), the file's own spillover compensates them as FlowJo does, the lab is found
    by its panel, and the review nests each model gate under the model's own parent."""
    import numpy as np
    from flow_er.workspace import models
    from flow_er.workspace.data import Dataset
    raw = tmp_path / "raw"
    raw.mkdir()
    for n in (4, 5):
        f = next(ALS.glob(f"HC Specimen_001_C{n}_C0{n} *.fcs"))
        (raw / f.name).symlink_to(f)
    monkeypatch.setattr(settings, "workspace_dir", tmp_path / "ws")
    exp = store.create(raw, "raw")["id"]
    sid = "HC Specimen_001_C5_C05 Peptide.fcs"
    ds, flowjo = Dataset(raw), Dataset(ALS)
    cols = list(ds._sample(sid).pnn_labels)
    assert "LIVE_DEAD Aqua-A" in cols and ds.uniform
    assert np.allclose(ds.compensated(sid)[:, cols.index("FITC-A")],
                       flowjo.compensated(sid)[:, list(flowjo._sample(sid).pnn_labels).index("FITC-A")], atol=1e-3)
    assert models.lab_for(exp) == "als"
    review = models.review(exp, sid)
    assert len(review["gates"]) == 15
    assert "Lymphocytes (model)/Single Cells 1 (model)" in review["gates"]
