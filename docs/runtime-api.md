# Runtime API (ml/ ↔ api/, worker/, frontend/)

The gating runtime is a Python library in `backend/src/flow_er/ml`. The API and worker call the
functions below; everything they return is JSON-serialisable through `.to_json()` or plain dicts.
Heavy calls (`autogate`, `retrain`) belong in worker jobs. For what the models are, how
fine-tuning is enabled and where training runs (CPU or GPU), see [models.md](models.md).

## Python functions (`from flow_er import ml`)

| Function | Returns | Notes |
| --- | --- | --- |
| `ml.template_from_workspace(wsp_path, sample_id, fcs_path=None, gate_keys=None)` | `GateTree` | A lab template from one sample of a FlowJo workspace (any gate type: rectangles, polygons, biex edges). `fcs_path` gives `$PnR`; 262144 is assumed without it. |
| `ml.load_template(json_or_path)` | `GateTree` | A saved template, or a previous experiment's accepted gates. Save with `tree.to_json()`. |
| `ml.autogate(fcs_path, tree, lab=None, comp=None, unstim=None, plot=False)` | `GatingResult` | Gates every event. `unstim`: the donor's unstimulated-well FCS (or its `GatingResult`), used by cytokine gates. `comp`: `(spillover ndarray, detector names)`; default = the file's `$SPILLOVER`. |
| `ml.apply_edit(result, tree, gate_key, polygon_display)` | `GatingResult` | A user's edit: the gate and every gate below it are recomputed; the edited gate gets `model: "user"`. |
| `ml.export_gatingml(result, tree, path)` | path | GatingML 2.0, polygon gates in compensated raw units with the spillover matrix included. Gate ids are XML-safe names: `[^A-Za-z0-9._-]` → `_`, duplicates get `__<parent id>`. |
| `ml.add_labels(lab, fcs_path, tree, accepted, individual, sample_id=None, condition="", unstim=None)` | list of files | Stores the accepted gates (`{gate key: display polygon}`) of one sample as training examples. `individual` = the person or animal the sample came from; it decides the held-out share. |
| `ml.retrain(lab, gates=None, max_minutes=None, log=print, on="cpu", wait_minutes=None)` | dict | A new version of the lab's per-gate UNITO models, for the gates with new accepted samples (or `gates`) and their child gates; made current only if better (see below). `max_minutes` caps each gate's training; `log` receives one progress line per epoch. `on="cpu"` trains in this process (on its CUDA GPU if it has one); `"colab"` sends the job to the Colab notebook through `COLAB_JOBS_DIR` and waits up to `wait_minutes`; `"auto"` uses Colab when the notebook's heartbeat is under 3 min old, else the CPU. |
| `ml.confidence.threshold(lab)` | float | The low-confidence threshold autogate uses: the current version's calibrated one (`scripts/calibrate_confidence.py`, on validation donors), else 0.6. |
| `ml.list_versions(lab)` | list of dicts | Every version with `metrics`, `training_samples`, `parent`, `current`. |
| `ml.lab_status(lab)` | dict | `state`: `live`, `outdated` (the lab has kept >= 3 files its current version was not trained on: retrain) or `offline` (a model of the current version has no weights here or another image recipe); `why`, `version`, `new_files`, `new_individuals`. |
| `ml.set_current(lab, version)` | — | Rollback or roll forward. |

## Coordinates

- **Display units**: every channel goes through one shared transform (linear `x / $PnR` for
  scatter, logicle `T = $PnR, W = 0.5, M = 4.5` otherwise) to about 0–1. Plots, polygons the
  user edits, templates and model inputs are all in display units. Plot axes: 0–1.
- **Raw units**: compensated channel values, the units of the FCS file after compensation. Every
  polygon is also given in raw units (vertices converted; for export, edges are densified first).
- **Membership**: a cell is in a gate when it is in the parent gate and its display coordinates
  are inside the gate's polygon. Counts, the drawn polygon, edits and the GatingML export
  therefore always agree.

## `GateTree` JSON (a lab template)

```json
{
  "format": "gate-tree-v1", "recipe_id": "1bd7b6e6",
  "source": {"kind": "workspace", "path": "...wsp", "sample": "Specimen_001_C1_C01.fcs"},
  "gates": [
    {"key": "Lymphocytes", "x": "FSC-A", "y": "SSC-A", "compensated": [false, false],
     "template": [[0.33, 0.04], [0.24, 0.05], "..."], "is_cytokine": false, "qc": false},
    {"key": "Lymphocytes/.../CD4 Single Positive/IFNy FITC +", "x": "FITC-A", "y": "SSC-A",
     "compensated": [true, false], "template": [[0.95, 0.535], "..."], "is_cytokine": true,
     "qc": false}
  ]
}
```

- `key`: the gate's path below root joined by `/` (unique; names can repeat under different
  parents). Gates are listed parents first. `parent` = the key without its last part.
- `template`: polygon vertices in display units (x, y), not closed (first ≠ last).
- `is_cytokine`: gets the donor's unstimulated well as an extra model input. `qc`: Time/QC gates,
  always the template, never a model.

## `GatingResult` JSON (`result.to_json(plot)`)

```json
{
  "format": "gating-result-v1", "fcs": "...fcs", "n_events": 164335,
  "lab": "als", "model_version": "v001", "recipe_id": "1bd7b6e6",
  "gates": [
    {"gate": "Lymphocytes/.../CD4 Single Positive/IFNy FITC +",
     "parent": "Lymphocytes/.../CD4 Single Positive",
     "x": "FITC-A", "y": "SSC-A",
     "model": "template_copy",
     "polygon": {"display": [[0.95, 0.535], [0.93, 0.535], "..."],
                 "raw": [[156192.48, 140247.04], [126977.90, 140247.04], "..."]},
     "count": 2408, "parent_count": 76732, "percent_of_parent": 3.1382,
     "confidence": null, "low_confidence": false,
     "plot": {"density": [[0, 0, "..."], "..."], "cells": [[0.31, 0.12], "..."],
              "bins": 101, "axes": "display units 0..1; density[y][x]"}}
  ]
}
```

- `model`: which model answered: `"<lab>/<version>/unito/<gate_id>"` (the lab's per-gate model),
  `"template_copy"`, or `"user"` after an edit. Shared and base models stay on disk but are not
  routed (final decision, 18 Oct).
- `polygon`: `null` when the model found no gate (count 0).
- `confidence`: mean `|2p - 1|` of the parent cells within 2 pixels of the gate's boundary
  (1 = the model is sure, 0 = coin flip); `null` for template copy and user gates.
  `low_confidence: true` when it is below the version's `confidence_threshold` (calibrated on
  validation donors), or 0.6 when the version has none → show the gate for review.
- `plot` (only with `plot=true`): the parent population's 101 × 101 event counts in display units
  (`density[y][x]`, pixel `i` covers `i/100 ± 0.005`), and up to 20,000 sampled parent cells in
  display units. Draw the polygon's `display` vertices on top.

## Model versions

```
models/labs/<lab>/current           name of the current version
models/labs/<lab>/<version>/version.json
models/labs/<lab>/<version>/gates/<gate_id>/   per-gate UNITO models trained in that version
models/base/shared/, .../shared/    older shared models: kept on disk, not routed
```

`version.json`:

```json
{"version": "v004", "created": "2026-10-18T03:30:00", "parent": "v003",
 "gates": {"CD3__7f4ff286": "labs/als/v004/gates/CD3__7f4ff286",
           "CD4_Single_Positive__22c4f057": "als/unito_gpu/CD4_Single_Positive__22c4f057"},
 "shared": null, "confidence_threshold": 0.71,
 "metrics": {"held_out_error_pp": {"current": {"<gate>": 1.32}, "candidate": {"<gate>": 0.84}},
             "promoted": true, "reason": "held-out mean error 1.02 -> 0.71 pp",
             "gates": {"<gate>": {"kept": true, "held_out_error_pp": 0.84, "template_copy_pp": 1.9,
                                  "current_pp": 1.32, "warm_start": true}},
             "accepted_samples": ["lab_als/<sample id>", "..."],
             "confidence": {"threshold": 0.71, "wrong": 9, "wrong_flagged": 7, "calibrated_on": ["human:4191"]},
             "lr": 0.0001, "patience": 40, "held_out_individuals": ["donor-A"]},
 "training_samples": ["lab_als/<sample id>", "..."], "notes": "per-gate UNITO retrain of 2 gate(s) from v003"}
```

Routing per gate: the current version's per-gate UNITO → template copy. Retraining (`retrain`):
per-gate UNITO only, for the gates with accepted samples the current version has not seen (or the
`gates` given) plus their child gates. A gate with a model warm-starts from it at lr 1e-4 with early
stopping (patience 40) on the held-out individuals (~20% by hash of the individual id, or the
newest untrained individual in a small lab; fixed once chosen, `models/labs/<lab>/held_out.json`); a
gate without one needs ≥ 15 accepted samples and starts fresh. Each gate is then answered by the
best of its current model, the new model and template copy on its held-out examples
(`finetune.choose_route`; per-gate `route` in the summary): a gate on a model falls back to template
copy only if template copy wins by ≥ 0.5 pp on ≥ 2 examples, and that model is kept as `parked` for
the next retrain to start from. **Promote** only if the held-out mean frequency error improves and no
gate is worse by more than 1 pp; the candidate is stored either way, with its metrics. With nothing
new since the current version, `retrain` writes no version.

GPU jobs (`on="colab"`): the worker writes `<COLAB_JOBS_DIR>/<job>/` with `params.json`, the lab's
examples (`data/_cache/v1/lab_<lab>/`) and the current version's gate models in fp16 (`models/`),
and sets `job.json` to `queued`. The notebook loop touches `<COLAB_JOBS_DIR>/heartbeat`, runs
`ml.remote.run_job(job)` (the same `retrain` on the T4) and writes the new version and
`summary.json` back. The worker copies the version into `models/labs/<lab>/` (weights back to fp32)
and promotes by the rule above, unless the current version changed while the job ran.

## Endpoints this supports

| Endpoint | Calls |
| --- | --- |
| `POST /samples` (FCS upload) | store the file; record its individual (person/animal) and condition |
| `POST /samples/{id}/autogate` (job) | `ml.autogate(fcs, template, lab, unstim=<the donor's unstimulated sample, if uploaded>, plot=True)`; keep the `GatingResult` |
| `GET /samples/{id}/gates` | `result.to_json(plot=True)` (or without plot data for lists) |
| `PUT /samples/{id}/gates/{gate}` (user edit) | body `{"polygon": [[x, y], ...]}` in display units → `ml.apply_edit(...)` → the updated result |
| `POST /samples/{id}/accept` | `ml.add_labels(lab, fcs, template, {gate: polygon display}, individual, sample_id)` |
| `POST /labs/{id}/retrain` (job) | `ml.retrain(lab, on="auto")` → `{"version", "promoted", "reason", "graduated", "gates", "metrics"}` |
| `GET /labs/{id}/models` | `ml.list_versions(lab)` and `ml.lab_status(lab)` (as `status`); `POST /labs/{id}/models/{version}/current` → `ml.set_current` |
| `GET /samples/{id}/gatingml` | `ml.export_gatingml(result, template, path)` |

The template for a lab is a `GateTree` JSON stored by the API: built once from an uploaded `.wsp`
(`template_from_workspace`), saved, and optionally replaced by a later experiment's accepted gates
(`tree.with_polygons({...})`).

## CLI

`backend/scripts/autogate.py` runs the same path from the command line (see its docstring); with
`--expert-wsp` it also prints the expert's percentages, F1 per gate, and an overlay PNG.
