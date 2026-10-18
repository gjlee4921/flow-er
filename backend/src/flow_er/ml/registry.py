"""Model versions per lab and the router that picks a model for each gate.

Layout under settings.models_dir:
  base/shared/                    the base shared model (zero-shot): recipe.json + model.pt
  labs/<lab>/<version>/version.json
      {"version", "created", "parent", "gates": {gate_id: model dir}, "shared": model dir | null,
       "metrics": {...}, "training_samples": [...], "notes": ...}
      Model dirs may be inside the version folder or anywhere under models_dir (stored relative
      to it), so already-trained models can be registered without copying their weights.
  labs/<lab>/current              the active version's name (rollback = point it elsewhere)

Router order for a gate: the lab's per-gate UNITO -> the lab's shared fine-tuned model -> the
base shared model -> template copy. QC gates (Time) always use the template. A new version is
only made current when it is promoted (ml/finetune.py); see docs/runtime-api.md.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from flow_er.config import settings
from flow_er.ml import cache

TEMPLATE_COPY = "template_copy"


def models_root() -> Path:
    return Path(settings.models_dir)


def lab_dir(lab: str) -> Path:
    return models_root() / "labs" / lab


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(models_root().resolve()).as_posix()
    except ValueError:
        return str(path)


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else models_root() / p


# --- versions ---------------------------------------------------------------------------------


def list_versions(lab: str) -> list[dict]:
    """Every version of a lab (oldest first), each with "current": bool."""
    root, current = lab_dir(lab), current_version(lab)
    out = []
    for p in sorted(root.glob("*/version.json")):
        v = json.loads(p.read_text())
        out.append({**v, "current": v["version"] == current})
    return out


def current_version(lab: str) -> str | None:
    p = lab_dir(lab) / "current"
    return p.read_text().strip() if p.exists() else None


def set_current(lab: str, version: str) -> None:
    if not (lab_dir(lab) / version / "version.json").exists():
        raise ValueError(f"lab {lab!r} has no version {version!r}")
    (lab_dir(lab) / "current").write_text(version)
    _cache.clear()


def next_version_name(lab: str) -> str:
    n = len(list(lab_dir(lab).glob("*/version.json")))
    return f"v{n + 1:03d}"


def register_version(lab: str, gates: dict[str, Path] | None = None, shared: Path | None = None,
                     metrics: dict | None = None, training_samples: list[str] | None = None,
                     parent: str | None = None, notes: str = "", version: str | None = None,
                     make_current: bool = False) -> str:
    """Write a version.json. `gates` maps gate keys (or gate ids) to per-gate model dirs."""
    version = version or next_version_name(lab)
    d = lab_dir(lab) / version
    d.mkdir(parents=True, exist_ok=True)
    gate_dirs = {(k if "__" in k else cache.gate_id(k)): _rel(v) for k, v in (gates or {}).items()}
    data = {"version": version, "created": datetime.now().isoformat(timespec="seconds"),
            "parent": parent, "gates": gate_dirs, "shared": _rel(shared) if shared else None,
            "metrics": metrics or {}, "training_samples": training_samples or [], "notes": notes}
    (d / "version.json").write_text(json.dumps(data, indent=1))
    if make_current:
        set_current(lab, version)
    return version


def version_info(lab: str, version: str | None = None) -> dict | None:
    version = version or current_version(lab)
    if version is None:
        return None
    return json.loads((lab_dir(lab) / version / "version.json").read_text())


# --- predictors --------------------------------------------------------------------------------


@dataclass
class Predictor:
    """A trained U-Net that turns input planes into a 101 x 101 probability map."""
    id: str
    model_dir: Path

    def __post_init__(self):
        from flow_er.ml.unet import infer

        self.model, self.recipe = infer.load(self.model_dir)

    @property
    def channels(self) -> list[str]:
        return list(self.recipe["channels"])

    def predict(self, density: np.ndarray, template: np.ndarray,
                unstim: np.ndarray | None = None) -> np.ndarray:
        import torch

        from flow_er.ml.unet.dataset import normalise

        planes = {"density": normalise(density), "template": template.astype(np.float32),
                  "unstim": normalise(unstim) if unstim is not None else
                  np.zeros_like(density, dtype=np.float32)}
        x = torch.from_numpy(np.stack([planes[c] for c in self.channels])[None])
        with torch.no_grad():
            return torch.sigmoid(self.model(x))[0, 0].numpy()


_cache: dict[Path, Predictor] = {}


def _predictor(model_id: str, model_dir: Path) -> Predictor | None:
    model_dir = Path(model_dir)
    if not (model_dir / "model.pt").exists():
        return None
    if model_dir not in _cache:
        _cache[model_dir] = Predictor(model_id, model_dir)
    return _cache[model_dir]


def route(lab: str | None, gate_key: str, qc: bool = False) -> Predictor | None:
    """The model that answers this gate: the lab's current per-gate UNITO, or None for template
    copy. Shared and base models stay on disk but are not routed (final decision, 18 Oct)."""
    if qc:
        return None
    gid = cache.gate_id(gate_key)
    info = version_info(lab) if lab else None
    if info and gid in info["gates"]:
        return _predictor(f"{lab}/{info['version']}/unito/{gid}", _abs(info["gates"][gid]))
    return None
