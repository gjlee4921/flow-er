"""The seam between the agent and the gating models in `flow_er.ml`.

`flow_er.ml` (images, baseline, splits, evaluate, registry, unet, cytobert) is built and owned
separately; this module only reads it. Until it exposes training and inference, the agent says
so plainly instead of pretending, and the FlowJo workspace's own gates stand in as labels.

When `flow_er.ml` lands, these are the calls to wire, in this order:
  - registry: list model versions per gate, and which one is promoted
  - unet.train: train one gate from labelled samples (runs in the worker, not in a request)
  - unet.infer: place a gate on a sample, returning a polygon and a confidence
The promotion rule stays in `ml.registry`: code, not the agent, decides if a model is better.
"""
from __future__ import annotations

import importlib

from flow_er.workspace import store


def _module(name: str):
    try:
        return importlib.import_module(f"flow_er.ml.{name}")
    except Exception:  # noqa: BLE001  not built yet, or built but missing a dependency: either way, not usable
        return None


def status() -> dict:
    """What the gating models can do today."""
    reg, train_mod, infer_mod = _module("registry"), _module("unet.train"), _module("unet.infer")
    has = lambda m: bool(m) and any(not k.startswith("_") and callable(getattr(m, k)) for k in dir(m))
    available = has(reg) and has(train_mod) and has(infer_mod)
    return {
        "available": available,
        "registry": has(reg), "train": has(train_mod), "infer": has(infer_mod),
        "message": ("Gating models are available." if available else
                    "The per-gate models in flow_er.ml are still being built, so nothing can be trained yet. "
                    "The FlowJo workspace's gates stand in as labels until then."),
    }


def train(exp_id: str, gate: str | None) -> dict:
    s = status()
    if not s["available"]:
        store.journal(exp_id, "train_blocked", {"gate": gate})
        return {"queued": False, "message": s["message"]}
    # Training runs in the worker; queueing it is the worker's job once flow_er.ml is wired.
    store.journal(exp_id, "train_requested", {"gate": gate})
    return {"queued": False, "message": "flow_er.ml is present but the training queue is not wired yet."}
