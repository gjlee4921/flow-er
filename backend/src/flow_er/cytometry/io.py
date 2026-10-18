"""Read FCS files without loading them whole: header, events by row (memory-mapped), spillover.

Values are preprocessed exactly as FlowKit/flowio do on a full load (Time x $TIMESTEP, log-scale
channels expanded, then division by a gain that is not 1), so results match FlowKit while only
the rows asked for are read. Channel names are FlowJo's ('/' written as '_').
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CHUNK = 50_000          # events read and processed at a time


def read_fcs_text(path: Path) -> dict[str, str]:
    """The FCS TEXT segment as a dict; reads a few KB, never the event data."""
    with open(path, "rb") as f:
        header = f.read(58)
        start, end = int(header[10:18]), int(header[18:26])
        f.seek(start)
        text = f.read(end - start + 1).decode("latin-1")
    delim = text[0]
    # A doubled delimiter is an escaped literal delimiter inside a value.
    parts = text[1:].replace(delim * 2, "\0").rstrip(delim).split(delim)
    parts = [p.replace("\0", delim) for p in parts]
    return {k.upper(): v.strip() for k, v in zip(parts[::2], parts[1::2])}


def preprocess(events: np.ndarray, header: dict) -> np.ndarray:
    """FlowKit's (flowio's) event preprocessing, so values match a FlowKit file load exactly:
    the Time channel times $TIMESTEP, log-scale channels expanded, then every channel divided by
    its gain when the gain is not 1 (or 0)."""
    names = [header[f"$P{i}N"] for i in range(1, events.shape[1] + 1)]
    time = next((i for i, n in enumerate(names) if n.lower() == "time"), None)
    if time is not None and header.get("$TIMESTEP", "").strip():
        events[:, time] *= float(header["$TIMESTEP"])
    for i in range(events.shape[1]):
        decades, log0 = (float(v) for v in header.get(f"$P{i + 1}E", "0,0").split(","))
        if decades > 0:
            events[:, i] = 10 ** (decades * events[:, i] / float(header[f"$P{i + 1}R"])) * log0
        gain = float(header.get(f"$P{i + 1}G", "1") or 1)
        if gain not in (1.0, 0.0):
            events[:, i] /= gain
    return events


@dataclass
class FcsFile:
    """An FCS file opened for row access. Float ($DATATYPE F or D) list-mode data only."""
    path: Path
    header: dict = field(init=False)
    channels: list[str] = field(init=False)      # FlowJo-style $PnN
    pnr: dict[str, float] = field(init=False)

    def __post_init__(self):
        self.path = Path(self.path)
        self.header = read_fcs_text(self.path)
        if self.header["$DATATYPE"] not in ("F", "D"):
            raise NotImplementedError(f"{self.path}: $DATATYPE {self.header['$DATATYPE']}")
        n = int(self.header["$PAR"])
        self.channels = [self.header[f"$P{i}N"].replace("/", "_") for i in range(1, n + 1)]
        self.pnr = {c: float(self.header[f"$P{i}R"]) for i, c in enumerate(self.channels, 1)}

    @property
    def n_events(self) -> int:
        return int(self.header["$TOT"])

    def _data(self) -> np.memmap:
        h = self.header
        dtype = {"F": "f4", "D": "f8"}[h["$DATATYPE"]]
        order = "<" if h["$BYTEORD"].startswith("1") else ">"
        begin = int(h.get("$BEGINDATA", 0) or 0)
        if not begin:
            with open(self.path, "rb") as f:
                begin = int(f.read(34)[26:34])
        return np.memmap(self.path, dtype=order + dtype, mode="r", offset=begin,
                         shape=(self.n_events, len(self.channels)))

    def events(self, rows: np.ndarray) -> np.ndarray:
        """(len(rows), channels) preprocessed float64 values of these event rows."""
        return preprocess(np.asarray(self._data()[rows], dtype=np.float64), self.header)

    def chunks(self, rows: np.ndarray | None = None,
               size: int = CHUNK) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """(row numbers, events) CHUNK rows at a time; all rows by default."""
        rows = np.arange(self.n_events) if rows is None else rows
        for start in range(0, len(rows), size):
            part = rows[start:start + size]
            yield part, self.events(part)

    def spillover(self) -> tuple[np.ndarray, list[str]] | None:
        """The acquisition spillover matrix ($SPILLOVER / SPILL / $SPILL) and its detectors."""
        text = next((self.header[k] for k in ("$SPILLOVER", "SPILL", "$SPILL")
                     if self.header.get(k)), None)
        if not text:
            return None
        v = [t.strip() for t in text.split(",")]
        n = int(v[0])
        detectors = [d.replace("/", "_") for d in v[1:n + 1]]
        return np.array(v[n + 1:n + 1 + n * n], float).reshape(n, n), detectors
