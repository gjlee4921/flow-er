# Cache and prediction formats (phase 1)

Everything here lives under `settings.data_dir` (outside git). Written by
`backend/scripts/build_dataset.py`; read through `flow_er.ml.cache`. Models and reports never
read FCS files.

## Layout

```
data/_cache/v1/
  recipe.json                                   image recipe (flow_er.ml.images.RECIPE)
  _samples/<dataset>/<sample_id>.npz            per-sample source layer (below)
  <dataset>/<gate_id>/<sample_id>.npz           per-(sample, gate) model files (below)
  <dataset>/<gate_id>/<sample_id>.json          meta for that file
data/_results/<evaluation>/<method>/<dataset>/<gate_id>/<sample_id>.npz   predictions (below)
```

- `sample_id` = the FCS file name without `.fcs` (unique within a dataset).
- `gate_id` = `<slug of the gate name>__<first 8 hex of sha256(gate key)>`, e.g.
  `IFNy_FITC__3f2a9c1e`. The gate key is the full path below root joined by `/`
  (e.g. `Lymphocytes/Single Cells 1/.../CD4 Single Positive/IFNy FITC +`), as in the registry.

## Per-(sample, gate) model file: `<dataset>/<gate_id>/<sample_id>.npz`

| Array | dtype, shape | Meaning |
| --- | --- | --- |
| `density` | float32 [101, 101] | raw event counts of the gate's **parent** cells; `density[y, x]` |
| `mask` | uint8 [101, 101] | expert gate of this sample: pixel centre inside the gate |
| `template` | uint8 [101, 101] | template gate: the same gate of the dataset's reference sample (meta `template_source`) |
| `unstim` | float32 [101, 101] | cytokine gates only: `density` of the same gate in the donor's unstimulated well; absent for other gates |
| `cell_bins` | uint8 [N, 2] | pixel (x, y) of each parent cell, each in 0..100 |
| `cell_label` | bool [N] | FlowKit membership of each parent cell (the truth) |
| `cell_index` | uint32 [N] | the cell's event index in the FCS file |

Events: files with more than 500,000 events keep a fixed random subsample of 500,000 events
(seed 0) before anything else (`meta.events`; the source layer's `event_index` maps rows to the
file's event numbers, and `cell_index` indexes those kept rows). Gating is per event, so the kept
events' labels are exactly those of the whole file. ALS files are all below that size.

Pixels: display value `v` (about 0..1 in the shared transform) → pixel `round(100 v)`, clipped to
0..100. Row = y channel, column = x channel. Parent cells are all cells in the **expert** parent
gate. If a parent has more than 200,000 cells, a fixed random subsample of 200,000 is kept
(seed 0; `meta.cells.subsampled`); ALS parents are all below that, so ALS keeps every cell.
A model's per-cell prediction is `pred[i] = predicted_mask[cell_bins[i, 1], cell_bins[i, 0]]`.

## Meta: `<dataset>/<gate_id>/<sample_id>.json`

```json
{
  "format": "cache-v1", "recipe_id": "1bd7b6e6",
  "dataset": "als_ics_screen3", "sample": "<registry key dataset/workspace/sample>",
  "sample_id": "...", "individual": "human:2008", "condition": "unstimulated", "cohort": "HC",
  "gate": "<gate key>", "gate_id": "...", "parent": "<parent gate key or empty for root>",
  "is_cytokine": true,
  "x": {"channel": "FITC-A", "transform": {...}, "lo": 0.0, "hi": 1.0},
  "y": {"channel": "SSC-A",  "transform": {...}, "lo": 0.0, "hi": 1.0},
  "roles": {"als": "train", "als_cv[0]": "test", "pretrain": null},
  "template_source": "<registry key of the reference sample>",
  "unstim_source": "<registry key of the donor's unstimulated well, or null>",
  "cells": {"n_parent": 12345, "n_kept": 12345, "subsampled": false, "seed": 0},
  "counts": {"flowkit": 321, "flowkit_parent": 12345, "flowjo": 318, "flowjo_parent": 12301}
}
```

- `roles`: this sample's role in every evaluation of `split_v1.json` (`train`, `val`,
  `finetune`, `test`; `null` = not used there). `pretrain` is `train`/`val` for the pool.
- `counts`: FlowKit (truth) and FlowJo (saved in the workspace) event counts for the gate and its
  parent, for the FlowKit-vs-FlowJo noise floor: `|flowjo/flowjo_parent - flowkit/flowkit_parent|`
  in percentage points.
- Reference sample (template), chosen per gate: ALS = donor 2008's unstimulated well (first
  training donor); any other dataset = the first training sample (sorted by key; evaluation
  `train` stages and the pool's `train`) of the same dataset **and workspace** that has the gate,
  since workspaces have different trees. No such sample → `template` is all zeros and
  `template_source` is null (e.g. MO280 gates that exist only in the test tissues).
- The pool (`--evaluation pretrain`) only gets model files for gates whose FlowKit counts agree
  with FlowJo (registry `agrees_with_flowjo`), never Time gates.

## Per-sample source layer: `_samples/<dataset>/<sample_id>.npz`

Used by the baselines (they move templates through raw space, so they need values, not pixels)
and to derive the model files.

| Array | Meaning |
| --- | --- |
| `raw` float32 [n events, k] | compensated value of each gated channel (`columns`; `|raw` suffix = gate uses it uncompensated) |
| `columns`, `pnr` | channel names and their `$PnR` |
| `labels` bool [gates, n events] | FlowKit membership of every gate |
| `parent` int16 [gates] | parent gate index, -1 = all events |
| `dims` int16 [gates, 2] | columns of each gate's x and y channel |
| `images`, `masks` | [gates, 101, 101] parent density and expert mask (same as the model files) |
| `gate_keys` | gate keys, in tree order |
| `meta` | JSON string: registry row, recipe, each gate's FlowKit geometry and workspace transforms |

## Predictions: `data/_results/<evaluation>/<method>/<dataset>/<gate_id>/<sample_id>.npz`

| Array | Meaning |
| --- | --- |
| `pred` bool [N] | predicted membership of each parent cell, **same order as `cell_label`** in the model file |
| `prob` float32 [N] | optional: probability per cell |

`method` names: `template_copy`, `classical`, `unito`, `shared_zero_shot`, `shared_finetuned`
(plus suffixes for variants, e.g. `shared_finetuned_1ch`). Every method is scored on the expert
parent cells of each gate (gate by gate, not cascaded).
