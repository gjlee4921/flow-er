"""How a lab's models got better, round by round, for the demo: each version's % against the expert's on
samples no version was taught, gate by gate, from a fine-tuning run done earlier (GJ's loop comparison:
per_item.csv and steps.json). Read-only numbers, kept outside git in MODELS_DIR/rounds/<lab>.json.

    uv run python -m flow_er.workspace.rounds ~/Downloads/loop_comparison/mouse_lab_mo280 mo280
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from flow_er.config import settings

TISSUES = ["Liver", "Aorta", "Blood", "Spleen"]


def _dir() -> Path:
    return Path(settings.models_dir).expanduser() / "rounds"


def available() -> list[str]:
    return sorted(p.stem for p in _dir().glob("*.json")) if _dir().exists() else []


def get(lab: str) -> dict | None:
    p = _dir() / f"{lab}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _key(column: str) -> str:
    m = re.match(r"(v\d{3})", column)
    return m.group(1) if m else "template"


def _tree(paths: list[str]) -> list[dict]:
    """Gates parents first, siblings in the order the run listed them."""
    kids: dict[str, list[str]] = {}
    for p in paths:
        parts = p.split("/")
        for i in range(1, len(parts) + 1):
            node, up = "/".join(parts[:i]), "/".join(parts[:i - 1])
            kids.setdefault(up, [])
            if node not in kids[up]:
                kids[up].append(node)
    keep, out = set(paths), []

    def walk(node: str) -> None:
        for k in kids.get(node, []):
            if k in keep:
                parts = k.split("/")
                out.append({"path": k, "name": parts[-1], "parent": parts[-2] if len(parts) > 1 else "", "depth": len(parts) - 1})
            walk(k)
    walk("")
    return out


def build(src: Path, lab: str) -> dict:
    rows = list(csv.DictReader(open(src / "per_item.csv")))
    steps = json.loads((src / "steps.json").read_text()) if (src / "steps.json").exists() else []
    cols = list(dict.fromkeys(r["column"] for r in rows))
    rounds = []
    for c in cols:
        k = _key(c)
        if k == "template":
            continue
        step = next((s for s in steps if s.get("version") == k), {})
        rounds.append({"key": k, "label": k, "column": c,
                       "taught": step.get("taught"), "promoted": step.get("promoted"),
                       "reason": step.get("reason"), "routes": step.get("routes")})
    items = sorted({(r["tissue"], r["mouse"]) for r in rows},
                   key=lambda t: (TISSUES.index(t[0]) if t[0] in TISSUES else 9, int(t[1]) if t[1].isdigit() else 0))
    values: dict[str, dict] = {}
    err: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        item, k = f"{r['tissue']} {r['mouse']}", _key(r["column"])
        f1 = float(r["f1"]) if r.get("f1") not in (None, "") else None
        values.setdefault(item, {}).setdefault(r["gate"], {})[k] = [round(float(r["pct_pred"]), 3), round(float(r["pct_expert"]), 3),
                                                                  None if f1 is None else round(f1, 4)]
        for scope in ("all", r["tissue"]):
            err.setdefault(k, {}).setdefault(scope, []).append(float(r["abs_err_pp"]))
    means = {k: {s: round(sum(v) / len(v), 3) for s, v in by.items()} for k, by in err.items()}
    taught = [s.get("taught") for s in steps if s.get("taught")]
    return {"lab": lab, "source": str(src), "rounds": rounds, "reference": {"key": "template", "label": "Template copy"},
            "gates": _tree(list(dict.fromkeys(r["gate"] for r in rows))),
            "items": [{"id": f"{t} {m}", "tissue": t, "mouse": m, "sample": f"{t}_{m}.fcs"} for t, m in items],
            "values": values, "means": means, "taught": taught}


def save(src: Path, lab: str) -> Path:
    data = build(src, lab)
    _dir().mkdir(parents=True, exist_ok=True)
    out = _dir() / f"{lab}.json"
    out.write_text(json.dumps(data))
    return out


if __name__ == "__main__":
    out = save(Path(sys.argv[1]).expanduser(), sys.argv[2])
    d = json.loads(out.read_text())
    print(out, f"{len(d['rounds'])} rounds, {len(d['gates'])} gates, {len(d['items'])} samples")
    for k, m in d["means"].items():
        print(f"  {k:9s} " + "  ".join(f"{s} {v:.2f}" for s, v in m.items()))
