"""Gating models: classical baseline, per-gate U-Net, shared U-Net, and the runtime around them.

Stable API for the app (shapes and endpoints: docs/runtime-api.md). Imports are lazy so that
`import flow_er.ml` stays light; torch is only loaded when a model is used.
"""

from pathlib import Path as _Path


def autogate(fcs_path, tree, lab=None, comp=None, unstim=None, plot=False):
    """Gate an FCS file with a lab's template -> GatingResult (.to_json(plot) for the API)."""
    from flow_er.ml.runtime import autogate as run
    return run(fcs_path, tree, lab, comp=comp, unstim=unstim, plot=plot)


def apply_edit(result, tree, gate_key, polygon_display, fcs_path=None, comp=None):
    """A user's edit of one gate: recompute it and every gate below it -> GatingResult."""
    from flow_er.ml.runtime import apply_polygon
    return apply_polygon(result, tree, gate_key, polygon_display, fcs_path, comp)


def template_from_workspace(wsp_path, sample_id, fcs_path=None, gate_keys=None):
    """A lab template (GateTree) from one sample of a FlowJo workspace."""
    from flow_er.cytometry.io import FcsFile
    from flow_er.ml.gatetree import from_workspace
    return from_workspace(_Path(wsp_path), sample_id, FcsFile(fcs_path) if fcs_path else None,
                          gate_keys)


def load_template(data):
    """A lab template from its JSON (dict or file path), e.g. a previous experiment's gates."""
    from flow_er.ml.gatetree import GateTree
    return GateTree.from_json(data)


def export_gatingml(result, tree, path):
    """Write a GatingResult as GatingML 2.0 (polygon gates in compensated raw units)."""
    from flow_er.cytometry.export import to_gatingml
    return to_gatingml(result, tree, path)


def add_labels(lab, fcs_path, tree, accepted, individual, sample_id=None, condition="",
               unstim=None, comp=None):
    """Store a sample's accepted gates as training examples for the lab -> written files."""
    from flow_er.ml.finetune import add_labels as add
    return add(lab, fcs_path, tree, accepted, individual, sample_id, condition, unstim, comp)


def retrain(lab, **kwargs):
    """Train a new lab version on all its examples; promoted only if better -> summary dict."""
    from flow_er.ml.finetune import retrain as run
    return run(lab, **kwargs)


def list_versions(lab):
    """Every model version of a lab, with metrics and which one is current."""
    from flow_er.ml.registry import list_versions as run
    return run(lab)


def set_current(lab, version):
    """Make an existing version the lab's current one (rollback or roll forward)."""
    from flow_er.ml.registry import set_current as run
    return run(lab, version)
