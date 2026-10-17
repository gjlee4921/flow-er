"""Metrics of flow_er.ml.evaluate on hand-made arrays, and scripts/report.py end to end on a
synthetic cache in tmp_path."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flow_er.config import settings
from flow_er.ml import cache, evaluate


def labels(n_true: int, n: int = 100) -> np.ndarray:
    return np.arange(n) < n_true


def test_cell_metrics_counts_precision_recall_f1():
    true = labels(25)
    pred = np.zeros(100, bool)
    pred[5:30] = True                                   # 20 true positives, 5 false positives
    m = evaluate.cell_metrics(pred, true)
    assert (m["n_cells"], m["n_true"], m["n_pred"]) == (100, 25, 25)
    assert (m["tp"], m["fp"], m["fn"]) == (20, 5, 5)
    assert m["precision"] == pytest.approx(0.8)
    assert m["recall"] == pytest.approx(0.8)
    assert m["f1"] == pytest.approx(0.8)


def test_f1_needs_twenty_expert_positives():
    below = evaluate.cell_metrics(labels(19), labels(19))
    assert math.isnan(below["f1"])
    assert below["n_true"] == 19 and below["tp"] == 19 and below["precision"] == 1.0
    assert evaluate.cell_metrics(labels(20), labels(20))["f1"] == 1.0


def test_empty_prediction_and_shape_mismatch():
    m = evaluate.cell_metrics(np.zeros(100, bool), labels(30))
    assert math.isnan(m["precision"]) and m["recall"] == 0.0 and m["f1"] == 0.0
    with pytest.raises(ValueError):
        evaluate.cell_metrics(np.zeros(5, bool), labels(3, 6))


def test_frequency_and_signed_absolute_error():
    assert evaluate.frequency(labels(1, 4)) == 25.0
    assert evaluate.frequency(labels(2, 4)) == 50.0
    assert evaluate.frequency_error(labels(2, 4), labels(1, 4)) == (25.0, 25.0)
    assert evaluate.frequency_error(labels(1, 4), labels(2, 4)) == (-25.0, 25.0)
    assert math.isnan(evaluate.frequency(np.zeros(0, bool)))


def test_noise_floor_from_counts():
    counts = {"flowkit": 10, "flowkit_parent": 1000, "flowjo": 12, "flowjo_parent": 1000}
    assert evaluate.count_frequencies(counts) == pytest.approx((1.0, 1.2))
    assert evaluate.noise_floor(counts) == pytest.approx(0.2)
    m = evaluate.well_metrics(labels(2, 4), labels(1, 4), counts)
    assert m["freq_pred"] == 50.0 and m["freq_true"] == 25.0 and m["freq_error"] == 25.0
    assert m["noise_floor"] == pytest.approx(0.2)
    assert math.isnan(evaluate.well_metrics(labels(1, 4), labels(1, 4))["noise_floor"])


def well(method, sample, condition, pred, true, fk, fj, gate="g", cyto=True, unstim="u1",
         individual="d1", err=None, f1=np.nan):
    e = pred - true if err is None else err
    return {"method": method, "gate": gate, "sample": sample, "condition": condition,
            "is_cytokine": cyto, "unstim_source": None if condition == "unstimulated" else unstim,
            "individual": individual, "freq_pred": pred, "freq_true": true, "freq_flowkit": fk,
            "freq_flowjo": fj, "freq_error": e, "abs_freq_error": abs(e), "noise_floor": abs(fj - fk),
            "f1": f1}


def test_background_subtracted_response():
    df = pd.DataFrame([
        well("a", "u1", "unstimulated", 1.0, 0.5, 0.5, 0.6),
        well("a", "p1", "peptide", 3.0, 2.5, 2.5, 2.5),
        well("a", "c1", "positive control", 10.0, 12.0, 12.0, 12.0),
        well("b", "p1", "peptide", 3.0, 2.5, 2.5, 2.5),                # no unstimulated well
        well("a", "p1", "peptide", 3.0, 2.5, 2.5, 2.5, gate="h", cyto=False),
    ])
    r = evaluate.add_responses(df)
    pep, pc, no_unstim, not_cyto = r.iloc[1], r.iloc[2], r.iloc[3], r.iloc[4]
    assert (pep["stimulus"], pc["stimulus"]) == ("peptide", "positive_control")
    assert pep["response_pred"] == pytest.approx(2.0) and pep["response_true"] == 2.0
    assert pep["response_error"] == pytest.approx(0.0)
    assert pep["response_noise_floor"] == pytest.approx(0.1)        # |1.9 - 2.0|
    assert pc["response_pred"] == 9.0 and pc["response_true"] == 11.5
    assert pc["response_error"] == -2.5 and pc["abs_response_error"] == 2.5
    assert math.isnan(no_unstim["response_error"]) and no_unstim["stimulus"] == "peptide"
    assert math.isnan(not_cyto["response_error"]) and pd.isna(not_cyto["stimulus"])
    assert r["response_error"].notna().sum() == 2


def test_donor_means_and_summary():
    df = pd.DataFrame([
        well("a", "u1", "unstimulated", 1.0, 0.0, 0.0, 0.2, f1=0.9),
        well("a", "p1", "peptide", 3.0, 0.0, 0.0, 0.0, f1=0.7),
        well("a", "c1", "positive control", 14.0, 10.0, 10.0, 10.0),
        well("a", "u2", "unstimulated", 0.0, 4.0, 4.0, 4.0, individual="d2", f1=0.5),
    ])
    r = evaluate.add_responses(df)
    donors = evaluate.donor_means(r).set_index("individual")["abs_freq_error"]
    assert donors["d1"] == pytest.approx((1 + 3 + 4) / 3) and donors["d2"] == 4.0
    s = evaluate.summarize(r).iloc[0]
    assert (s["n_wells"], s["n_wells_f1"], s["n_donors"]) == (4, 3, 2)
    assert s["median_f1"] == pytest.approx(0.7)
    assert s["mean_abs_freq_error"] == pytest.approx(12 / 4)
    assert s["mean_freq_error"] == pytest.approx((1 + 3 + 4 - 4) / 4)
    assert s["donor_mean_abs_freq_error"] == pytest.approx((8 / 3 + 4) / 2)
    assert s["mean_noise_floor"] == pytest.approx(0.05)
    assert s["mean_abs_response_error_peptide"] == pytest.approx(2.0)   # (3-1) - (0-0)
    assert s["mean_abs_response_error_positive_control"] == pytest.approx(3.0)  # 13 - 10
    assert s["n_responses_peptide"] == 1
    assert s["mean_response_noise_floor_peptide"] == pytest.approx(0.2)


# --- report.py end to end ---------------------------------------------------------------------


def load_report():
    path = Path(__file__).resolve().parents[2] / "scripts" / "report.py"
    spec = importlib.util.spec_from_file_location("report_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATES = {"A": False, "A/IFNy +": True, "B": False, "B/IFNy +": True}
WELLS = {"unstimulated": 4, "peptide": 30, "positive control": 60}   # expert positives / 200


def write_cache(sample, condition, unstim_key):
    for gk, cyto in GATES.items():
        n_true = WELLS[condition] if cyto else 100
        truth = labels(n_true, 200)
        path = cache.gate_path(sample, gk)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, cell_label=truth)
        meta = {"sample": f"ds/w.wsp/{sample['sample']}", "individual": "human:1",
                "condition": condition, "cohort": "HC", "gate": gk, "is_cytokine": cyto,
                "unstim_source": unstim_key if cyto else None,
                "counts": {"flowkit": n_true, "flowkit_parent": 200, "flowjo": n_true + 2,
                           "flowjo_parent": 200}}
        path.with_suffix(".json").write_text(json.dumps(meta))
        yield gk, truth


def test_report_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    report = load_report()
    samples = {c: {"dataset": "ds", "workspace": "w.wsp", "sample": f"{i}.fcs",
                   "file": f"ds/{i}.fcs"} for i, c in enumerate(WELLS)}
    unstim_key = "ds/w.wsp/0.fcs"
    for condition, sample in samples.items():
        for gk, truth in write_cache(sample, condition, unstim_key):
            for method in ("perfect", "shifted"):
                if method == "shifted" and gk == "B":
                    continue                                 # a missing prediction
                pred = truth if method == "perfect" else np.roll(truth, 10)
                p = cache.results_path("als", method, sample, gk)
                p.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(p, pred=pred)

    assert report.available_methods("als") == ["perfect", "shifted"]
    pairs = [(s, g) for s in samples.values() for g in report.cached_gates(s)]
    assert len(pairs) == 12
    df, missing = report.score("als", ["perfect", "shifted", "unito"], pairs)
    assert len(df) == 12 + 9 and missing == {"shifted": 3, "unito": 12}
    perfect = df[df["method"] == "perfect"]
    assert (perfect["abs_freq_error"] == 0).all()
    assert perfect["f1"].isna().sum() == 2                  # unstimulated cytokine wells: 4 cells
    assert (perfect["f1"].dropna() == 1).all()
    assert perfect["noise_floor"].tolist() == pytest.approx([1.0] * 12)
    assert perfect["response_error"].dropna().tolist() == [0.0] * 4

    out = tmp_path / "out"
    paths = report.write_report(df, ["perfect", "shifted", "unito"], out, "test", "als / test",
                                missing)
    assert all(p.is_file() for p in paths)
    rows = pd.read_csv(out / "results_test.csv")
    assert len(rows) == 21 and "IFNy + [A]" in set(rows["gate_name"])
    md = (out / "report_test.md").read_text(encoding="utf-8")
    assert "IFNy + [B] *" in md and "| A |" in md and "shifted 3, unito 12" in md
    assert "| Gate | perfect | shifted |" in md
    assert report.gate_labels(list(GATES))["B/IFNy +"] == "IFNy + [B]"
    assert len(report.select_gates(pairs, ["IFNy +"])) == 6
