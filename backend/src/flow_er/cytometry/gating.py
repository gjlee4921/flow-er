"""Apply a gate tree and return per-cell membership for each gate.

workspace_labels gates events with a FlowJo workspace's own gating strategy (FlowKit), CHUNK
events at a time: gates are per event, so the labels equal a whole-file FlowKit analysis at a
fraction of its memory. compensate applies a spillover matrix like FlowKit does.
"""

from pathlib import Path

import flowkit as fk
import numpy as np
from flowutils import compensate as fu_compensate

from flow_er.cytometry.io import FcsFile


def name_and_path(gate_key: str) -> tuple[str, tuple[str, ...]]:
    """'A/B/C' -> ('C', ('root', 'A', 'B')): FlowKit's gate id."""
    parts = gate_key.split("/")
    return parts[-1], ("root", *parts[:-1])


def compensate(events: np.ndarray, channels: list[str], spill: np.ndarray,
               detectors: list[str]) -> np.ndarray:
    """Events with the detector columns compensated (inverse of the spillover matrix)."""
    idx = [channels.index(d) for d in detectors]
    return fu_compensate.compensate(events, spill, idx)


def workspace_labels(fcs: FcsFile, wsp_path: Path, sample_id: str, gate_keys: list[str],
                     columns: list[tuple[str, bool]], rows: np.ndarray | None = None
                     ) -> tuple[np.ndarray, np.ndarray, fk.GatingStrategy]:
    """FlowKit membership of `gate_keys` for `rows` (all events by default), and the values of
    `columns` = [(channel, compensated?)] for the same rows. Also returns the gating strategy."""
    wsp = fk.Workspace(str(wsp_path), load_missing_file_data=True)
    gs = wsp.get_gating_strategy(sample_id)
    comp = wsp.get_comp_matrix(sample_id)
    rows = np.arange(fcs.n_events) if rows is None else rows
    labels = np.zeros((len(gate_keys), len(rows)), bool)
    values = np.zeros((len(rows), len(columns)), np.float32)
    start = 0
    for _, events in fcs.chunks(rows):
        chunk = fk.Sample(events, sample_id=sample_id, channel_labels=fcs.channels)
        result = gs.gate_sample(chunk)
        raw = chunk.get_events(source="raw")
        comped = comp.apply(chunk) if comp is not None else raw
        for g, key in enumerate(gate_keys):
            labels[g, start:start + len(events)] = result.get_gate_membership(*name_and_path(key))
        for c, (ch, compensated) in enumerate(columns):
            values[start:start + len(events), c] = (comped if compensated else
                                                    raw)[:, fcs.channels.index(ch)]
        start += len(events)
    return labels, values, gs
