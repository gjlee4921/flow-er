# Gating models, the runtime and fine-tuning

What the app can call today in `flow_er.ml`, what each call takes and returns, and where
training runs. Exact JSON shapes and the endpoint mapping are in
[runtime-api.md](runtime-api.md); the model specification is [unito-spec.md](unito-spec.md);
design choices are logged in [decisions.md](decisions.md).

Status: 18 October 2026, branch `phase1-splits`.

## 1. Models

| Model | What it does | Inputs (per gate) | Weights | Status |
| --- | --- | --- | --- | --- |
| **Template copy** | The lab's template gate, unchanged | template polygon | none | Always available; the fallback for every gate |
| **Classical baseline** | Template moved to the data: density-peak shift for scatter/lineage gates; cytokine threshold set from the donor's unstimulated well | template + parent events | none | Evaluation only (`ml/baseline.py`, `scripts/run_baselines.py`); not in the runtime |
| **Per-gate UNITO** (spec §1) | One U-Net per gate: parent density image → gate mask | parent density | `models/als/unito/` (4 ALS gates, trained on CPU) · `models/als/unito_gpu/` (11 ALS gates, trained on Colab) | CPU models usable now; GPU weights still on Colab (recipes only here) |
| **Shared U-Net, base** (spec §2) | One model for every gate, pretrained on the pool (spleen, macrophages, spectral): moves the template to fit this sample | parent density, template mask, unstimulated-well density (cytokine gates; zeros otherwise) | `models/pretrain/shared/shared` | Weights on Colab; becomes `models/base/shared` |
| **Shared U-Net, ALS fine-tuned** | The base model fine-tuned on ALS training donors | same | `models/als/shared_finetuned/shared` | Weights on Colab |
| Single-input variants (`_1ch`) and MO280 models | Comparison without the unstimulated well; the unseen-tissue test | | | Running on Colab |

All models share one image recipe (`ml/images.py`):
- **Transform:** a fixed display transform: linear for scatter, logicle for fluorescence.
- **Grid:** 101 × 101 bins.
- **Recipe id:** `1bd7b6e6`, stored with every model. A model trained on another recipe is refused.

**How they compare** (locked ALS test, 6 unseen donors from a different screen; full tables in
`data/_results/als/report_test.md`). Per-gate UNITO is the most consistent; the shared
fine-tuned model is best on CD4 IFNγ and CD4+; template copy is still best on the upstream
scatter gates, which the expert rarely redrew. Mean frequency error, percentage points:

| Gate | Template copy | Classical | UNITO | Shared zero-shot | Shared fine-tuned |
| --- | --- | --- | --- | --- | --- |
| Live/Dead | 0.48 | 0.64 | **0.08** | 1.16 | 1.02 |
| CD8+ | 1.29 | 1.44 | **0.52** | 1.33 | 0.97 |
| CD4+ | 2.31 | 16.18 | 3.13 | 3.35 | **2.25** |
| IFNγ [CD4] | 0.88 | 0.64 | 0.34 | 1.36 | **0.17** |
| IFNγ [CD8] | 1.00 | 2.09 | **0.15** | 0.90 | 0.27 |

## 2. The runtime: an FCS file in, gates out

```python
from flow_er import ml

template = ml.template_from_workspace("lab.wsp", "Specimen_001_C1_C01.fcs", fcs_path="ref.fcs")
result = ml.autogate("new_sample.fcs", template, lab="als",
                     unstim="same_donor_unstimulated.fcs", plot=True)
result.to_json(plot=True)                         # -> the API's GET /samples/{id}/gates
ml.export_gatingml(result, template, "gates.xml") # GatingML 2.0 for FlowJo and others
```

**Input**
- **An FCS file:** float data, any size; it's read in chunks, so memory stays small.
- **A template (`GateTree`):** the lab's gates, parents first. It can come from:
  - an uploaded FlowJo workspace, any gate type (`template_from_workspace`);
  - a saved template (`load_template`);
  - a previous experiment's accepted gates (`tree.with_polygons`).
- **Optional:**
  - `lab`: picks that lab's models.
  - `unstim`: the same donor's unstimulated well, used by cytokine gates.
  - `comp`: a spillover matrix; the default is the file's own `$SPILLOVER`.

**What it does, gate by gate down the tree**
1. Takes the parent's cells.
2. Builds a density image from them (at most 200k cells).
3. Asks the routed model for a probability map, or uses the template.
4. Traces the 0.5 contour into an editable polygon.
5. Puts every parent cell inside the polygon into the gate; those cells become the next gate's parent.

**Which model answers a gate.** The router takes the first of these that exists:
1. the lab's current per-gate UNITO for that gate;
2. the lab's current shared fine-tuned model;
3. the base shared model;
4. template copy.

Time/QC gates always use the template.

**Output** (`GatingResult`), for each gate:
- the polygon in display units (0–1, what the user draws on) and in raw compensated units;
- the event count, the parent count and % of parent;
- which model answered;
- a confidence (how decisive the model is near the boundary), with a `low_confidence` flag below 0.6 so the gate gets reviewed;
- optional plot data: the parent's 101 × 101 density and up to 20,000 sampled cells.

Every event's membership is kept in memory for edits and exports.

**Other runtime calls**
- **`apply_edit(result, template, gate, polygon)`:** a user's correction. It recomputes that gate and everything below it.
- **`export_gatingml(...)`:** writes polygon gates in raw units with the spillover matrix included. FlowKit gating with this file reproduces our membership exactly.

**Speed:** about 6–11 s per ALS sample (160k events, 15 gates) on a laptop CPU, including the
unstimulated well. Inference always runs on the CPU (32 ms per gate for the shared model,
143 ms for a per-gate UNITO), so serving needs no GPU.

**Command line:** `backend/scripts/autogate.py`. With the expert's workspace, it also prints the
expert's percentages and F1 per gate, and draws an overlay of model, template and expert
(demo: `data/_demo/`).

## 3. Fine-tuning per lab

Fine-tuning is **on for a lab as soon as its users accept gates**; there is no switch. Two calls
do it:

```python
# 1. The user accepts (or corrects) a sample's gates -> training examples for the lab
ml.add_labels("als", "sample.fcs", template,
              accepted={gate_key: polygon_display, ...},   # the gates as the user left them
              individual="donor-A", sample_id="sample-123",
              unstim=unstim_result)                        # optional, for cytokine gates

# 2. Later, as a background job: train a new version, adopt it only if it is better
summary = ml.retrain("als", max_minutes=60, log=print)
# {"version": "v002", "promoted": True, "reason": "held-out mean error 1.02 -> 0.71 pp",
#  "graduated": ["CD3__7f4ff286"], "metrics": {"current": {...}, "candidate": {...}}}

ml.list_versions("als")        # every version with metrics; which one is current
ml.set_current("als", "v001")  # rollback
```

**`add_labels`**
- **Inputs:** the FCS file, the template, the accepted polygons in display units, and who the sample came from (`individual`).
- **Output:** the training files it wrote. Each accepted gate becomes one example: the parent density, the accepted gate as the target, the lab template, and the unstimulated well for cytokine gates. They're stored in the cache format under `data/_cache/v1/lab_<lab>/`.
- **Held-out individuals:** about 20% of individuals, chosen by a hash of the individual id, are never trained on. They stay held out as labels accumulate, so successive versions are compared fairly.

**`retrain`**
- **Starting point:** the lab's current shared model, else the base model, else the ImageNet encoder.
- **Training:** on all the lab's accepted examples, new and old (replay), at lr 1e-4. Early stopping with patience 40 on the held-out individuals' Dice. Drift augmentation is on.
- **Graduation:** a gate with ≥ 15 accepted samples also gets its own per-gate UNITO. It's kept only if it beats the shared model on that gate.
- **Promotion:** the new version becomes current only if the held-out mean frequency error is lower than with the current models (or template copy, for a new lab), and no gate is more than 1 pp worse. Otherwise it's stored as a candidate and the old version stays current. The code decides; nothing else does.
- **Output:** a version folder `models/labs/<lab>/<version>/` holding the shared model, any graduated per-gate models, `version.json` (metrics, the training sample ids, the parent version, the decision and its reason), and the `current` pointer.

**To enable it for a lab you need:**
1. The ML packages installed: `uv sync --group ml`.
2. A template for the lab, stored by the app.
3. An individual id for every uploaded sample (person or animal). Without it, held-out samples could come from people who are also in training.
4. At least a few individuals: with fewer than about five, the 20% rule may hold none out, and `retrain` stops with a clear error.
5. Optionally, the base model in `models/base/shared`, so a new lab starts from it instead of from ImageNet weights.

## 4. Where training runs: CPU or GPU, and Colab

**Training runs wherever the process that calls it runs.** `train.py` uses a CUDA GPU when
PyTorch sees one and the CPU otherwise. Nothing in the code connects to Colab.

- **On this laptop** (CPU only, Intel Core Ultra 5, no CUDA GPU), measured per batch of 8:

  | Model | One training step | Inference, per gate image |
  | --- | --- | --- |
  | Shared model (ResNet-18 U-Net) | 0.35 s | 32 ms |
  | Per-gate UNITO | 1.24 s | 143 ms |

  **Example estimate:** a lab with 20 accepted samples × 15 gates is 300 examples. That's about 15 s per epoch for the shared model, or roughly 15–25 min for a typical retrain (75 min at most with the default 300 epochs). Each graduated gate adds up to about 12 min. That's fine as an overnight or background job for a small lab; use `max_minutes` to cap it.
- **Colab was only used for the research runs** (steps 4–7: the 31 M-parameter UNITO grids and the pool pretraining). Those were run by hand: upload the cache, run `scripts/train_gate.py`, bring back the predictions and `recipe.json`. **The app does not need Colab running in the background.** Inference is CPU-only, and fine-tuning works on CPU. Colab also isn't suitable as a background service: sessions disconnect when idle and are capped in length.
- **If retrains get too slow on the CPU** (large labs, many graduated gates), run the worker on a machine with a CUDA GPU, such as a cloud VM. The same code uses the GPU automatically; no code change is needed.

## 5. Mapping to the calls proposed in spec-plan §2

| Proposed | Use | Notes |
| --- | --- | --- |
| `registry.versions(gate_def)` | `ml.list_versions(lab)` | Versions are per lab, and a version holds the shared model plus any per-gate models, so one version covers every gate |
| `registry.current(gate_def)` | `ml.list_versions(lab)` → the one with `current: true` (or `registry.current_version(lab)`) | |
| `unet.train.train_gate(pool, *, seed, budget, on_progress)` | `ml.retrain(lab, max_minutes=..., log=callback)` | The pool is the lab's accepted examples (`add_labels`). `log` gets one line per epoch (loss, held-out Dice). The summary includes held-out error for the current and candidate models, per gate |
| `unet.infer.place_gate(version, parent_events_2d)` | `ml.autogate(fcs, template, lab)` | Gates the whole tree in one pass (each gate needs its parent), returning polygon + confidence + model id per gate. A single-gate helper can be added if a workflow node needs one |
| `baseline.place_gate(parent_events_2d, template)` | template copy inside `autogate` | The classical peak-shift baseline is evaluation code; on unseen tissue and screens it was worse than template copy, so the runtime falls back to template copy |
| "Must beat the baseline and the current version" | the promotion rule above | For a new lab the current model is template copy (the baseline), so a first version must beat it |

The acceptance test in spec-plan §2 (Figshare spleen, young → aged) no longer applies. The
spleen gates are one template for every mouse, so spleen is pretraining data only. The
benchmarks are the ALS per-donor test and the MO280 unseen-tissue test (`phase1-brief.md`).

## 6. Getting the trained weights into the router

The GPU-trained weights are on Colab. Once they're copied into `models/`, register them (no
copying of weights; versions point at the folders):

```python
from flow_er.config import settings
from flow_er.ml import registry
import shutil

m = settings.models_dir
shutil.copytree(m / "pretrain/shared/shared", m / "base/shared")          # the base model
gates = {p.name: p for p in (m / "als/unito_gpu").iterdir() if (p / "model.pt").exists()}
registry.register_version("als", gates=gates, shared=m / "als/shared_finetuned/shared",
                          notes="GPU UNITO (11 gates) + shared fine-tuned", make_current=True)
```

Lab `als` version `v001` (the 4 CPU-trained gates) stays available for rollback.
