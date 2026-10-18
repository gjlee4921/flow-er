# Architecture

Full product description: the flow-er PRD (Confluence, "Reinvent the wheel" page). App plan: [spec-plan.md](spec-plan.md). Models and runtime: [models.md](models.md).

- **backend/**: FastAPI app (`src/flow_er/main.py`) plus a separate worker process for training and inference.
  - `cytometry/`: FCS and FlowJo workspaces.
    - `io.py` reads FCS files in chunks through a memory map, with FlowKit's own preprocessing.
    - `gating.py` gates with a workspace's FlowKit gating strategy and applies compensation.
    - `export.py` writes GatingML 2.0.
  - `ml/`: the gating models and their runtime.
    - The public API is in `ml/__init__.py` (`autogate`, `apply_edit`, `add_labels`, `retrain`, `list_versions`, `set_current`, ...).
    - `images.py` is the single recipe for turning a 2D plot into a model input; training and inference both use it.
    - `gatetree.py` defines a lab's template.
    - `runtime.py` gates an FCS file in cascade.
    - `registry.py` keeps every model version per lab and routes each gate to the best model available.
    - `finetune.py` turns accepted gates into training data and retrains.
    - A new version only becomes current when it beats the current one (template copy, for a new lab) on held-out individuals, and no gate gets worse by more than 1 pp.
    - `unet/` holds the U-Nets: model, dataset, training, inference, polygons.
    - `baseline.py`, `evaluate.py`, `splits.py` and `cache.py` serve the research evaluation.
  - `scripts/`: research and ops command lines.
    - `inspect_wsp.py`, `build_registry.py`: inspect the data and build the dataset registry.
    - `build_dataset.py`: build the cache.
    - `run_baselines.py`, `train_gate.py`, `report.py`: baselines, training and reports.
    - `autogate.py`: the runtime from the command line.
- **frontend/**: React + TypeScript (Vite). The API client's types are generated from the backend's OpenAPI schema. The app and agent are on `feat/web-agent` (see spec-plan.md).
- **Outside the repo**:
  - **Data** (`DATA_DIR`, default `../data`): the datasets (FCS + FlowJo workspaces), `_splits/` (dataset registry and frozen split; they name donors, so never in git), `_cache/v1/` (model inputs, [cache-format.md](cache-format.md)) and `_results/`.
  - **Models** (`MODELS_DIR`, default `../models`): `base/shared/`, `labs/<lab>/<version>/`, and the research runs.
