# Hand-over: running the gating models in the app

For whoever wires `flow_er.ml` into the API, worker and frontend. What to put next to the repo,
how to check it works, what to call, and what is still being improved. Function signatures and
JSON shapes: [runtime-api.md](runtime-api.md). Models, routing and fine-tuning: [models.md](models.md).

## 1. What lives outside git

Data and models are never committed. They sit next to the repo (defaults in
`backend/src/flow_er/config.py`; override with `DATA_DIR` / `MODELS_DIR`):

```
reinvent-the-wheel/
  flow-er/                         this repo
  models/                          MODELS_DIR
    als/unito_gpu/<gate_id>/       11 per-gate UNITO models: model.pt (124 MB) + recipe.json
    labs/als/current               one line: the active version (v003)
    labs/als/v003/version.json     gate -> model folder, metrics, training samples
    labs/als/template.json         the ALS lab template (GateTree JSON, 16 gates)
    base/shared/                   optional: the shared U-Net (55 MB), see §4
  data/                            DATA_DIR
    als_ics_screen4/               demo input: 24 FCS files (8 donors x 3 wells: unstimulated, peptide,
                                   positive control; 6 of the donors were never used in training)
                                   and the expert's two FlowJo workspaces (for comparison overlays)
```

What each part is for:

| Folder | Needed for | Without it |
| --- | --- | --- |
| `models/als/unito_gpu/` | The trained gate models (from the Colab GPU run) | No model can gate |
| `models/labs/als/` | Tells the router which weights belong to lab `als`, which version is current, and the lab's template. A few KB | Every gate silently falls back to template copy |
| `models/base/shared/` | Fallback for gates with no per-gate model (being removed, §4) | Those gates use template copy, which tested as good or better |
| `data/als_ics_screen4/` | Unseen FCS files to demo on, plus the expert's gates to compare against | Use any FCS files from the same panel |

Not needed for the app: `data/_cache`, `data/_splits`, `data/_results` and the other datasets are
research and evaluation files. Training data for a lab is created by the app itself, from the
gates users accept (`ml.add_labels`).

Data licences: the ALS data is public on Zenodo (records 16754591 and 16754609). Share it
privately with the team; never commit it.

## 2. Set up and check

```bash
cd backend
uv sync                                       # app + gating models (PyTorch CPU)
uv run python scripts/check_models.py verify  # every model folder has weights that load
```

End to end on one unseen donor, with the expert's own gates for comparison (about 15 s on a laptop CPU):

```bash
D=../../data
uv run python scripts/autogate.py "$D/als_ics_screen4/HC Specimen_001_C5_C05 Peptide.fcs" \
    --template-json ../../models/labs/als/template.json \
    --unstim "$D/als_ics_screen4/HC Specimen_001_C4_C04 Negative.fcs" --lab als \
    --expert-wsp "$D/als_ics_screen4/Screen #4 HC ICS.wsp" --expert-sample Specimen_001_C5_C05.fcs \
    --out result.json --gatingml gates.xml --overlay overlay.png
```

To build a template from a FlowJo workspace instead, use `--template-wsp <file.wsp> --template-sample
<sample id>` (add `--save-template tree.json` to keep it as JSON).

Expected output on that sample: upstream gates F1 >= 0.99, CD8+ 0.93, cytokine gates 0.70–0.86;
the model column reads `als/v003/unito/<gate>` for the 11 trained gates.

## 3. What the app calls

```python
from flow_er import ml

tree = ml.load_template("../models/labs/als/template.json")      # the lab's template
result = ml.autogate(fcs_path, tree, lab="als", unstim=unstim_fcs_path, plot=True)  # worker job
payload = result.to_json(plot=True)        # per gate: polygon, count, %, model, confidence, plot data

result = ml.apply_edit(result, tree, gate_key, polygon_display)   # user drags a gate
ml.add_labels("als", fcs_path, tree, accepted_polygons, individual="donor-id")      # user accepts
ml.retrain("als", max_minutes=60, log=progress)                   # worker job
ml.list_versions("als"); ml.set_current("als", "v003")            # versions, rollback
ml.export_gatingml(result, tree, "gates.xml")                     # export
```

Endpoint mapping: [runtime-api.md](runtime-api.md#endpoints-this-supports). Polygons come in display
units (0–1, what the user sees and drags) and raw units (export). `low_confidence: true` means
"show this gate for review". `model` says which model answered: `<lab>/<version>/unito/<gate_id>`,
`template_copy`, or `user` after an edit.

In the data lab, model gates show beside FlowJo's as "<gate> (model)". When a project's files come
from several FlowJo workspaces, or a workspace uses FlowJo's biex scale, the lab draws every file in
the models' display (the same transform as above), so a gate means the same cells in every file and a
model polygon needs no conversion; FlowJo's own gates are moved into it. A reviewed gate remembers
its file (`reviewed_on`), and only that file's corrections are taught from it.

Raw files with no FlowJo workspace work too: the project finds its lab by panel (every channel the
lab's gates are drawn on is in the files), and the review nests each model gate under the model's own
parent, since there are no FlowJo gates to hang them on.

Per-sample metadata the app must collect: `individual` (the person or animal the sample came from)
and, for cytokine panels, which file is that individual's unstimulated well.

## 4. What the tests found, and what changes next

Locked ALS test (6 donors never seen in training, from a different batch); mean error in population
frequency over the 7 gates every method covers, percentage points (lower is better):

| Method | All 7 gates | 4 cytokine gates | Decision |
| --- | --- | --- | --- |
| **Per-gate UNITO** | 0.93 | 0.27 | **Used**: most consistent on validation and test; beat template copy on the cytokine gates for all 6 donors |
| Template copy | 1.08 | 0.65 | **Used** as the fallback for gates without a model |
| Shared U-Net, fine-tuned | 0.89 | 0.31 | Not used: within noise of UNITO on test, twice its error on validation; no better than template copy on gates without UNITO |
| UNITO + CytoBERT | 0.99 | 0.26 | Not used: -0.014 pp vs UNITO (95% CI -0.05 to +0.03); a plain MLP gives the same; removed 4,032 true CD8+ cells on one well |
| Classical baseline | 3.43 | 0.96 | Evaluation only |

Unseen tissue (MO280: trained on mouse blood/spleen, tested on liver/aorta): no method transfers
(template copy 7.3 pp, shared zero-shot 7.6, UNITO 11.4, classical 17.0). A new tissue needs the
lab's corrections, then retraining: the review-and-retrain loop is the product, not an extra.

**Final routing:** per-gate UNITO where the lab has a trained model for that gate, otherwise
template copy, flagged for review. Done in `ml/` on 18 Oct (same function names and JSON shapes,
so app code does not change):
1. The router and `retrain()` no longer use the shared model (bf28e0e); `retrain` trains per-gate
   UNITO only, for the gates with new accepted samples and their child gates, warm-started at
   lr 1e-4; a new gate needs >= 15 accepted samples (7897ff0).
2. A gate's new model is kept only if it beats both its current model and template copy on that
   gate's held-out samples (7897ff0).
3. The confidence threshold is calibrated on validation donors and stored in the version
   (fac4ad6, db2f9e7). Lab `als` v003 now uses **0.94**, from the screen 3 validation donors
   4191 and 7207: it flags all 5 gates that were off by more than 1 pp (0.6 flagged none), at the
   cost of sending 27 of 61 right gates to review. Rerun with
   `scripts/calibrate_confidence.py --lab als --wells <val wells json> --write`.
4. Retraining a per-gate UNITO on a laptop CPU takes about 6–8 min per gate; it belongs in the
   worker, never in a request. `ml.retrain(lab, on="colab")` sends it to the Colab notebook through
   `COLAB_JOBS_DIR`, and `on="auto"` falls back to the CPU when the notebook is quiet (8a74dcf).
