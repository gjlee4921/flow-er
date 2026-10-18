"""The data lab's operations: the gate tree, events to draw, and the population table.

Gates come from three places and are told apart by `source`: the FlowJo workspace, the
scientist (drawn in the lab, kept like any other change), and later the gating models in
`flow_er.ml`. Counts always use every event; plots use a fixed subset.
"""
from __future__ import annotations

import csv
from dataclasses import asdict

import numpy as np

from flow_er.workspace import store, toolkit
from flow_er.workspace.data import Dataset, Gate, in_polygon, open_dataset, short_label

MAX_POINTS = 6000  # sent to the browser per plot


def dataset(exp_id: str) -> Dataset:
    return open_dataset(store.meta(exp_id)["data_path"])


def _user_gates(exp_id: str, with_draft: bool = True) -> list[Gate]:
    gates = [Gate(**g) for g in store.state(exp_id, "gates")]
    d = store.draft(exp_id)
    if with_draft and d and d["kind"] == "gate":
        # one gate, or several drawn at once (a quadrant is four)
        drafts = d["payload"] if isinstance(d["payload"], list) else [d["payload"]]
        gates += [Gate(**g) for g in drafts]
    return gates


def gates(exp_id: str, sid: str) -> list[Gate]:
    return dataset(exp_id).flowjo_gates(sid) + _user_gates(exp_id)


def _channel_index(ds: Dataset, sid: str, pnn: str) -> int:
    return list(ds._sample(sid).pnn_labels).index(pnn)


def membership(exp_id: str, sid: str, key: str | None, by_key: dict[str, Gate] | None = None) -> np.ndarray:
    """Which events of a sample fall in a gate (and all its parents)."""
    ds = dataset(exp_id)
    n = ds._sample(sid).event_count
    if not key:
        return np.ones(n, dtype=bool)
    by_key = by_key or {g.key: g for g in gates(exp_id, sid)}
    g = by_key[key]
    if g.source == "flowjo":
        return ds.flowjo_membership(sid, key)
    parent = membership(exp_id, sid, g.parent, by_key)
    ev = ds.events(sid, "processed")
    if g.kind == "range":
        # one parameter, from a low level to a high one (drawn on a histogram)
        x = ev[:, _channel_index(ds, sid, g.dims[0])]
        lo, hi = sorted((g.vertices[0][0], g.vertices[1][0]))
        return parent & (x >= lo) & (x <= hi)
    xi, yi = (_channel_index(ds, sid, d) for d in g.dims)
    return parent & in_polygon(ev[:, xi], ev[:, yi], g.vertices)


def tree(exp_id: str, sid: str) -> list[dict]:
    """The gate tree for one sample, with counts and percentages."""
    ds = dataset(exp_id)
    gs = gates(exp_id, sid)
    by_key = {g.key: g for g in gs}
    total = ds._sample(sid).event_count
    d = store.draft(exp_id)
    drafted = d["payload"] if d and d["kind"] == "gate" else []
    draft_keys = {g["key"] for g in (drafted if isinstance(drafted, list) else [drafted])}
    counts: dict[str | None, int] = {None: total}
    out = []
    for g in gs:
        m = membership(exp_id, sid, g.key, by_key)
        c = int(m.sum())
        counts[g.key] = c
        parent_n = counts.get(g.parent, total)
        out.append({**asdict(g), "count": c, "pct_parent": round(100 * c / parent_n, 3) if parent_n else 0.0,
                    "pct_total": round(100 * c / total, 3), "draft": g.key in draft_keys})
    return out


def plot(exp_id: str, sid: str, x: str, y: str, parent: str | None, stage: str = "processed") -> dict:
    """Events of one population on two channels, with which child gate each falls in."""
    ds = dataset(exp_id)
    gs = gates(exp_id, sid)
    by_key = {g.key: g for g in gs}
    sub = ds.subset(sid)
    ev = ds.events(sid, "raw" if stage == "raw" else "processed")[sub]
    pmask = membership(exp_id, sid, parent, by_key)[sub]
    xi, yi = _channel_index(ds, sid, x), _channel_index(ds, sid, y)
    pts = ev[pmask][:, [xi, yi]]
    children = [g for g in gs if g.parent == (parent or None) or (not parent and g.parent is None)]
    label = np.full(len(pts), -1, dtype=int)
    for i, g in enumerate(children):
        cm = membership(exp_id, sid, g.key, by_key)[sub][pmask]
        label[(label < 0) & cm] = i
    keep = slice(0, MAX_POINTS)
    return {
        "x": np.round(pts[keep, 0], 4).tolist(),
        "y": np.round(pts[keep, 1], 4).tolist(),
        "label": label[keep].tolist(),
        "children": [{"key": g.key, "name": g.name, "source": g.source, "dims": g.dims, "vertices": g.vertices}
                     for g in children],
        "parent": parent,
        "n_parent": int(membership(exp_id, sid, parent, by_key).sum()),
        "n_shown": int(min(len(pts), MAX_POINTS)),
        "stage": stage,
    }


def stats(exp_id: str, gate_keys: list[str] | None, channels: list[str] | None = None,
          samples: list[str] | None = None, columns: list[dict] | None = None) -> list[dict]:
    """One row per gate per file: count, share of the parent and of everything, the events of the
    parent outside the gate, and any toolkit statistic as an extra column.

    `columns` are {"stat", "parameter"?, "arg"?} from the toolkit (arg is the percentile, or the
    ancestor for "% of an ancestor"). `channels` is the older shorthand: median and CV of each.
    Values are on compensated events before display scaling, which is what FlowJo reports. No gates
    means every gate in the tree.
    """
    ds = dataset(exp_id)
    cols = list(columns or []) + [{"stat": s, "parameter": c} for c in (channels or []) for s in ("median", "cv")]
    for c in cols:
        if c["stat"] not in toolkit.STAT_IDS:
            raise ValueError(f"no statistic '{c['stat']}'; the toolkit has {sorted(toolkit.STAT_IDS)}")
    names = {c["id"]: short_label(c["id"], "") for c in ds.channels()}
    legacy = {("median", c): f"median {names.get(c, c)}" for c in (channels or [])} | {("cv", c): f"CV {names.get(c, c)}" for c in (channels or [])}
    rows = []
    for sid in samples or ds.sample_ids:
        by_key = {g.key: g for g in gates(exp_id, sid)}
        nodes = {t["key"]: t for t in tree(exp_id, sid)}
        total = ds._sample(sid).event_count
        keys = gate_keys if gate_keys else list(nodes)
        comp = ds.compensated(sid) if any(c.get("parameter") for c in cols) else None
        xf = ds._transforms(sid) if any(c["stat"] == "mode" for c in cols) else {}
        for key in keys:
            node = nodes.get(key)
            if key and not node:
                continue
            m = membership(exp_id, sid, key or None, by_key)
            count = int(m.sum())
            parent = nodes.get(node["parent"]) if node and node["parent"] else None
            grand = nodes.get(parent["parent"]) if parent and parent["parent"] else None
            parent_n = parent["count"] if parent else total
            row = {"sample": sid, "gate": node["name"] if node else "All events", "gate_path": key or "",
                   "source": node["source"] if node else "", "count": count,
                   "pct_parent": node["pct_parent"] if node else 100.0, "pct_total": node["pct_total"] if node else 100.0,
                   "outside_count": int(parent_n - count) if node else 0,
                   "pct_outside": round(100.0 - node["pct_parent"], 3) if node else 0.0}
            for c in cols:
                stat, param, arg = c["stat"], c.get("parameter"), c.get("arg")
                label = legacy.get((stat, param)) or toolkit.column_label(stat, param, arg, names)
                if stat == "count":
                    val = count
                elif stat in ("pct_parent", "pct_total"):
                    val = row[stat]
                elif stat == "pct_grandparent":
                    val = round(100 * count / (grand["count"] if grand else total), 3) if node else 100.0
                elif stat == "pct_of":
                    anc = nodes.get(arg) if arg else None
                    ok = not arg or (anc and (key + "/").startswith(anc["key"] + "/"))
                    val = round(100 * count / (anc["count"] if anc else total), 3) if ok and (anc["count"] if anc else total) else None
                else:
                    j = _channel_index(ds, sid, param)
                    t = xf.get(param)
                    mode_of = None
                    if t is not None:
                        def mode_of(v, t=t):
                            d = t.apply(v)
                            h, edges = np.histogram(d, bins=256, range=(0, 1))
                            peak = (edges[h.argmax()] + edges[h.argmax() + 1]) / 2
                            return float(t.inverse(np.array([peak]))[0])
                    val = toolkit.value(stat, comp[m, j], arg, mode_of)
                    val = round(val, 2) if val is not None else None
                row[label] = val
            rows.append(row)
    return rows


def population_table(exp_id: str, samples: list[str] | None = None) -> list[dict]:
    """Every gate in every sample: the table the agent quotes and the views are drawn from. With
    `samples`, only those files are counted and their rows replace theirs in the saved table."""
    ds = dataset(exp_id)
    rows = [r for r in read_table(exp_id, "population_stats") if r["sample"] not in samples] if samples else []
    for sid in samples or ds.sample_ids:
        for t in tree(exp_id, sid):
            if t["draft"]:
                continue
            rows.append({"sample": sid, "gate": t["name"], "gate_path": t["key"], "parent": t["parent"] or "",
                         "source": t["source"], "count": t["count"], "pct_parent": t["pct_parent"],
                         "pct_total": t["pct_total"]})
    save_table(exp_id, "population_stats", rows)
    return rows


def after_change(exp_id: str, kind: str | None) -> None:
    """A kept or dropped gate changes the counts: recount, so no table is ever stale."""
    if kind == "gate" and store.path(exp_id, "tables", "population_stats.csv").exists():
        population_table(exp_id)


# ---- saved tables ------------------------------------------------------------------------

def save_table(exp_id: str, name: str, rows: list[dict]) -> None:
    p = store.path(exp_id, "tables", f"{name}.csv")
    if not rows:
        p.write_text("")
        return
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def tables(exp_id: str) -> list[dict]:
    out = []
    for p in sorted(store.path(exp_id, "tables").glob("*.csv")):
        rows = read_table(exp_id, p.stem)
        out.append({"name": p.stem, "rows": len(rows), "columns": list(rows[0].keys()) if rows else []})
    return out


def read_table(exp_id: str, name: str) -> list[dict]:
    p = store.path(exp_id, "tables", f"{name}.csv")
    if not p.exists():
        return []
    with open(p) as f:
        rows = list(csv.DictReader(f))
    for r in rows:  # numbers back to numbers
        for k, v in r.items():
            try:
                r[k] = int(v) if v.lstrip("-").isdigit() else float(v)
            except (ValueError, AttributeError):
                pass
    return rows
