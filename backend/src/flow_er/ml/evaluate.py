"""Scores of predicted gates against FlowKit's per-cell truth (docs/unito-spec.md §3).

Per well: per-cell precision, recall and F1 (F1 only with >= MIN_POSITIVES expert-positive
cells), population frequency in % of the parent and its error in percentage points (pp), and the
FlowKit-vs-FlowJo noise floor. Per donor: background-subtracted responses of cytokine gates
(stimulated well minus the donor's unstimulated well). Per gate: summaries of a long table.
No file I/O here; scripts/report.py loads the cache and predictions.
"""

import numpy as np
import pandas as pd

MIN_POSITIVES = 20
UNSTIMULATED = "unstimulated"
STIMULI = {"peptide": "peptide", "positive control": "positive_control"}


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def cell_metrics(pred: np.ndarray, true: np.ndarray, min_positives: int = MIN_POSITIVES) -> dict:
    """Counts, precision, recall and F1 of pred vs true; F1 is NaN below min_positives."""
    pred, true = np.asarray(pred, bool), np.asarray(true, bool)
    if pred.shape != true.shape:
        raise ValueError(f"pred {pred.shape} and truth {true.shape} differ")
    tp = int(np.sum(pred & true))
    n_pred, n_true = int(pred.sum()), int(true.sum())
    precision, recall = _ratio(tp, n_pred), _ratio(tp, n_true)
    f1 = _ratio(2 * tp, n_pred + n_true) if n_true >= min_positives else float("nan")
    return {"n_cells": int(true.size), "n_true": n_true, "n_pred": n_pred, "tp": tp,
            "fp": n_pred - tp, "fn": n_true - tp,
            "precision": precision, "recall": recall, "f1": f1}


def frequency(member: np.ndarray) -> float:
    """Population frequency in % of the parent cells."""
    member = np.asarray(member, bool)
    return _ratio(100.0 * member.sum(), member.size)


def frequency_error(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    """(signed, absolute) frequency error in pp: predicted % minus expert %."""
    err = frequency(pred) - frequency(true)
    return err, abs(err)


def count_frequencies(counts: dict) -> tuple[float, float]:
    """(FlowKit %, FlowJo %) of the gate in its parent from the meta counts."""
    return (100 * _ratio(counts["flowkit"], counts["flowkit_parent"]),
            100 * _ratio(counts["flowjo"], counts["flowjo_parent"]))


def noise_floor(counts: dict) -> float:
    """|FlowJo % - FlowKit %| in pp."""
    fk, fj = count_frequencies(counts)
    return abs(fj - fk)


def well_metrics(pred: np.ndarray, true: np.ndarray, counts: dict | None = None) -> dict:
    """Every per-well number: cell metrics, frequencies, errors and the noise floor."""
    signed, absolute = frequency_error(pred, true)
    fk, fj = count_frequencies(counts) if counts else (float("nan"), float("nan"))
    return {**cell_metrics(pred, true), "freq_pred": frequency(pred), "freq_true": frequency(true),
            "freq_error": signed, "abs_freq_error": absolute,
            "freq_flowkit": fk, "freq_flowjo": fj, "noise_floor": abs(fj - fk)}


def add_responses(wells: pd.DataFrame) -> pd.DataFrame:
    """Add background-subtracted responses to stimulated wells of cytokine gates.

    Needs columns method, gate, sample, condition, is_cytokine, unstim_source (the sample key of
    the donor's unstimulated well) and freq_pred, freq_true, freq_flowkit, freq_flowjo. A response
    is computed only when the unstimulated well of the same method and gate is in the table.
    Adds stimulus, response_pred, response_true, response_error, abs_response_error,
    response_noise_floor (|FlowJo response - FlowKit response| in pp).
    """
    df = wells.copy()
    cols = ["freq_pred", "freq_true", "freq_flowkit", "freq_flowjo"]
    unstim = df[df["condition"] == UNSTIMULATED].set_index(["method", "gate", "sample"])[cols]
    unstim = unstim[~unstim.index.duplicated()]
    stim = df["is_cytokine"].astype(bool) & df["condition"].isin(list(STIMULI))
    df["stimulus"] = df["condition"].map(STIMULI).where(stim)
    idx = pd.MultiIndex.from_arrays([df["method"], df["gate"], df["unstim_source"].fillna("")])
    base = unstim.reindex(idx).set_axis(df.index)
    base.loc[~stim] = np.nan
    df["response_pred"] = df["freq_pred"] - base["freq_pred"]
    df["response_true"] = df["freq_true"] - base["freq_true"]
    df["response_error"] = df["response_pred"] - df["response_true"]
    df["abs_response_error"] = df["response_error"].abs()
    df["response_noise_floor"] = ((df["freq_flowjo"] - base["freq_flowjo"])
                                  - (df["freq_flowkit"] - base["freq_flowkit"])).abs()
    return df


def donor_means(wells: pd.DataFrame) -> pd.DataFrame:
    """Per (method, gate, individual): mean |frequency error| over the donor's wells."""
    return (wells.groupby(["method", "gate", "individual"], as_index=False)["abs_freq_error"]
            .mean())


def summarize(wells: pd.DataFrame) -> pd.DataFrame:
    """One row per (method, gate) from a long per-well table (after add_responses)."""
    rows = []
    donors = donor_means(wells).groupby(["method", "gate"])["abs_freq_error"]
    donor_mean = donors.mean()
    for (method, gate), d in wells.groupby(["method", "gate"], sort=False):
        row = {"method": method, "gate": gate, "is_cytokine": bool(d["is_cytokine"].any()),
               "n_wells": len(d), "n_wells_f1": int(d["f1"].notna().sum()),
               "median_f1": d["f1"].dropna().median() if d["f1"].notna().any() else np.nan,
"mean_freq_error": d["freq_error"].mean(),
               "mean_abs_freq_error": d["abs_freq_error"].mean(),
               "donor_mean_abs_freq_error": donor_mean.get((method, gate), np.nan),
               "n_donors": int(d["individual"].nunique()),
               "mean_noise_floor": d["noise_floor"].mean()}
        for stimulus in STIMULI.values():
            s = d[d["stimulus"] == stimulus]
            row[f"n_responses_{stimulus}"] = int(s["abs_response_error"].notna().sum())
            row[f"mean_abs_response_error_{stimulus}"] = s["abs_response_error"].mean()
            row[f"mean_response_noise_floor_{stimulus}"] = s["response_noise_floor"].mean()
        rows.append(row)
    return pd.DataFrame(rows)
