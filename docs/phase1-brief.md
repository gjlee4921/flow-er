# Phase 1 brief: per-gate U-Net vs classical baseline

Hand-off for the sessions that build phase 1. Product context: the flow-er PRD on Confluence
(https://darling-so.atlassian.net/wiki/spaces/darlingso/pages/40599642/flow-er+PRD).

## Goal

Gate unseen people and unseen tissues better than the classical baseline, scored against gates an
expert set per sample:

1. **ALS benchmark (headline):** human PBMC intracellular cytokine staining, where the expert set
   each donor's cytokine thresholds. Test on 6 donors never seen in training.
2. **MO280 unseen tissue:** mouse blood/spleen for training, liver/aorta for testing, where the
   expert drew different upstream gates per tissue.

The Figshare spleen set is **pretraining only**: its gates are one template copied to every mouse,
so testing on it would only measure template copying. Phase 1 is the U-Net only; CytoBERT
(phase 2) comes after.

## Where things are

| What | Path |
| --- | --- |
| Repo | `~/code/reinvent-the-wheel/flow-er` |
| Data (never commit) | `~/code/reinvent-the-wheel/data/` (`settings.data_dir`, override with `DATA_DIR`) |
| Dataset registry | `data/_splits/datasets_v1.json` — every workspace sample: file, MD5, species, tissue, person/mouse, condition, control flag, usable gates; per-gate FlowKit-vs-FlowJo agreement |
| Frozen split | `data/_splits/split_v1.json` — read it with `flow_er.ml.splits.load_split()` |
| Inspection outputs | `data/_inspect/<dataset>/` (CSV/JSON per workspace) and `<dataset>.txt` |
| Scripts | `backend/scripts/inspect_wsp.py` → `build_registry.py` → `python -m flow_er.ml.splits freeze/check` |
| Prototype (reference only) | `docs/prototype/flow-er-prototype.html` (`<script id="engine">`: the JS classical baseline) |

The registry and split name donors, so they stay outside git (`docs/decisions.md`); the tests that
need them skip when `data/_splits/` is missing.

## Datasets and their roles

| Dataset | What | Role |
| --- | --- | --- |
| `als_ics_screen3/4` (Zenodo 16754591/609) | 16 donors (8 HC, 8 ALS) × 3 wells (unstimulated, peptide, positive control); per-donor cytokine gates | ALS benchmark |
| `aorta_olfr2`, MO280 experiment (Fortessa, `27-Feb-2023.wsp`) | 16 mice × aorta/blood/liver/spleen; mouse = tube N (inferred from the lab's Analysis.xlsx) | MO280 unseen tissue |
| `figshare_spleen` (Figshare 210475) | 11 mice, templated gates | pretraining |
| `macrophages_peritoneal` (Figshare 12546686) | 81 samples, 17 specimen groups (mice unknown) | pretraining |
| `als_spectral` #1–#3 | 22 donors not in the benchmark, mostly templated | pretraining |
| `als_ics_screen1/2` | donors named only "Donor N" (may be benchmark donors) | **not used** |
| `als_spectral` #4/#5 | the benchmark donors; FlowKit is 10–36% below FlowJo from `cd45+` down | **not used** |
| `aorta_olfr2`, MO CVR experiment (Aurora) | 6 samples; 2 of 28 gates agree with FlowJo | **not used** |

Controls (FMOs, beads, single stains, unstained, compensation controls) are flagged in the
registry and never used.

## Split v1 (frozen)

The unit is the individual (person or mouse): every sample of an individual has one role per
evaluation, and nobody is on the training side (pretraining pool, train, fine-tune) and the
held-out side (validation, test) of any evaluation. `flow_er.ml.splits.validate()` enforces this
plus: no controls, no unusable or excluded samples, no unknown identities in val/test, a donor's
three wells together, and no identical file content on both sides. `check` also verifies every
used file's MD5.

- **ALS:** train 2008, 2238, 3551, 7182, 7183, 7184 · val 4191, 7207 · fine-tune 3417, 6936 ·
  **test (locked) 3740, 4595, 6934, 7181, 8710, 8830**. Score every gate except the Time QC gate.
- **MO280 round 1:** train blood + spleen of mice 1, 3, 5, 9, 11, 13 (val 7, 15) · test liver +
  aorta of mice 2, 4, 6, 8, 10, 12, 14, 16. Score the 35 gates shared by the training and test
  tissues: **headline the 14 whose expert gate changes between tissues**; report the other 21
  separately as "template copy is perfect here". 4 aorta-only gates (Neutrophiles and children,
  `Ly6C+/CCR2+`) cannot be scored.
- **Shared pretraining pool:** spleen (all 11 mice) + macrophages + spectral #1–#3 except 6936;
  early-stopping hold-out = spleen young 5–6, macrophage groups SM_PM_2:C and SM_PM_4:A, spectral
  donors 3553 and 8075. Only gates that agree with FlowJo (median ≤ 1%, worst ≤ 5% on wells with
  ≥ 100 events), no Time gates; each dataset's gates are separate tasks; datasets weighted
  equally. Never scored.
- **Stretch (defined in the split, run later):** ALS 4-fold CV by donor (only after the locked
  test result is recorded) → MO280 round 2 (halves swapped).

## Decisions already made

- **Labels:** FlowKit's per-cell memberships are the truth; FlowJo's saved counts are reported
  beside them for every gate and well as the noise floor. FlowKit does not reproduce FlowJo
  exactly (ALS IL-4 up to +12%, IL-17 about −15% in screen 4 HC; scatter gates 0.2–1% low); a
  time-boxed biex investigation did not fix it, so we stopped.
- **Model:** UNITO pipeline, one U-Net per gate, built on `segmentation_models_pytorch`
  (`smp.Unet("resnet18", encoder_weights="imagenet", ...)`); no UNITO package or weights exist.
  Input: the gate's parent cells on its two channels as a 101×101 density image; target: the gate
  mask; output: probability map → per-cell membership and an editable 0.5-contour polygon.
  BCEWithLogits, Adam. Hyperparameter search (lr {1e-3, 3e-4} × batch {4, 8}) on ALS only;
  MO280 uses lr 1e-3, batch 8.
- **Cytokine gates:** the main model takes the donor's unstimulated well as a second input
  channel; the single-input U-Net is the comparison.
- **Image space:** one shared display transform for all wells and datasets (workspaces use
  different biex settings); masks come from mapping pixel centres back into each workspace's
  display space. Same recipe for training and inference (`ml/images.py`), stored with each model.
- **Time gate:** dropped from modelling (per-well acquisition QC).
- **Baselines:** (0) template copied unchanged; (a) classical: peak-shift template transfer for
  scatter/live/lineage gates; for cytokine gates, the threshold placed from the donor's own
  unstimulated well (template shifted until the unstimulated well shows a background rate fitted
  on training donors).
- **Shared U-Net:** one template-refining U-Net pretrained on the pool, evaluated zero-shot and
  fine-tuned. No mouse-only vs mixed comparison.
- **Metrics:** per-cell F1 (only for wells with ≥ 20 expert-positive cells) and population
  frequency error in percentage points per well and per donor; for ICS also the error in the
  background-subtracted response (peptide − unstimulated).
- **Memory:** load one FCS at a time; cache only the samples a run uses (skip screens 1–2,
  Aurora, spectral #4/#5, controls; MO280 only what round 1 uses). Never read FCS inside a
  training loop.
- Package manager: `uv` (backend), `pnpm` (frontend). Python 3.12. Package name `flow_er`.

## Build order (time limits; stop and check in if a step runs > 50% over)

1. Freeze split v1, `ml/splits.py` + pytest checks (20 min): **done**.
2. ALS cache (20 min): **done**. 48 wells; `docs/cache-format.md`.
3. ALS baselines: template copy and classical (30 min): **done**.
4. Per-gate UNITO on ALS (1–1.5 h): **done on GPU (Colab)** as `unito_gpu`, 11 gates. The locked
   test is recorded in `data/_results/als/report_test.md`. The CPU run was stopped after 4
   gates (`models/als/unito`).
5. Pool cache (45 min): **done**. 114 samples, chunked reader.
6. Shared template-refining U-Net on ALS, zero-shot and fine-tuned (1.5–2 h): **3-channel done
   on GPU**, in the ALS report. The `_1ch` variant and a lower-lr fine-tune (`--lr 1e-4 3e-5`)
   are queued.
7. MO280 round 1 (1 h): **cache and baselines done** (`data/_results/mo280_round1/`); UNITO and
   the shared models are queued on GPU.
8. Stretch, in order: ALS 4-fold CV → MO280 round 2 → ALS learning curve (2/4/6 training donors)
   → spleen drift test.

Added 18 Oct: the inference runtime and per-lab fine-tuning for the app (`ml/runtime.py`,
`ml/registry.py`, `ml/finetune.py`); see `docs/models.md` and `docs/runtime-api.md`.

**Notes on the results:**
- The locked ALS test is cross-batch: every train/val donor is from screen 3, every
  finetune/test donor from screen 4.
- The classical baseline is weak on both benchmarks. Its peak shift jumps between populations,
  and its cytokine threshold breaks for donors whose unstimulated well has a large IFNγ+
  population. Template copy is the fair bar.

After each step, report the numbers and the time taken.
