# Gating model spec: UNITO reimplementation + flow-er extensions

Implementation reference for phase 1. Framework: **PyTorch** (UNITO itself is PyTorch;
`segmentation_models_pytorch` is PyTorch).

## Sources

- Paper: Chen, Ionita et al., "Automated cytometric gating with human-level performance using
  bivariate segmentation", *Nature Communications* 16 (2025), doi:10.1038/s41467-025-56622-2
  (open access: https://pmc.ncbi.nlm.nih.gov/articles/PMC11821879/ ; preprint:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11100732).
- Authors' code: https://github.com/KyleeCJ/UNITO — **custom "no redistribution" licence: no
  distribution of the code or any modification, no use inside a commercial product.**
  flow-er must ship under a permissive licence, so **do not copy, vendor or adapt code from that
  repository.** Implement from this spec and the paper. The architecture is a standard U-Net
  (Ronneberger et al. 2015); an independent implementation is fine.
- No trained UNITO weights are published.
- UNITO's own data (for reference / optional later use):
  - mass cytometry with 5 independent manual annotations — Pennsieve, CC BY:
    https://discover.pennsieve.io/datasets/408 (v2 17.6 GB; v1 2.3 GB);
  - flow cytometry (97 subjects, 4 annotators) — FlowRepository FR-FCM-ZZGS / FR-FCM-Z8P9 (login required).

## 1. UNITO as published (implement this first; it is the "per-gate U-Net" row)

**Unit of work:** one binary segmentation model per gate. Gates are applied down the gate tree:
cells outside a gate are removed and the survivors form the next gate's input.

**Input image (per sample, per gate)**
- The gate's two channels, taken from the gate's parent population.
- Per channel, min–max normalise to [0, 100] and round to integers → a 101 × 101 count image.
  (Paper: mass cytometry is arcsinh(x/5) first; it gives no flow transform. flow-er decision:
  one shared transform for all datasets, then the fixed-range binning in `ml/images.py`.)
- One input channel.

**Target mask**
- 101 × 101 binary: pixel = 1 where the expert's in-gate cells fall (flow-er: pixel centre inside
  the expert gate geometry, mapped through the workspace transform).
- Optional smoothing: convex hull of the mask (SciPy `ConvexHull`), switchable; without it, fill holes.

**Network (standard U-Net)**
- Encoder: 4 stages with 64, 128, 256, 512 channels. Each stage = two (3×3 conv, stride 1,
  padding 1, no bias → BatchNorm → ReLU), then 2×2 max-pool.
- Bottleneck: same double-conv block with 1024 channels.
- Decoder: 4 stages mirroring the encoder. Each = 2×2 transposed conv (stride 2), concatenate
  the encoder skip, double-conv block. Because 101 is odd, resize the upsampled map to the
  skip's spatial size before concatenating.
- Output: 1×1 conv → 1 channel of logits (101 × 101).

**Training**
- Loss: `BCEWithLogitsLoss` per pixel. Optimiser: Adam.
- Hyperparameters tuned per gate: learning rate and batch size only (defaults 1e-3 and 8),
  chosen on a held-out 1/8 of the *training* individuals by Dice score.
- No augmentation in the paper (flow-er adds some; see §2).
- Monitor Dice / pixel accuracy on validation; stop when converged (flow-er: early stopping
  on the validation individuals).

**Inference → cells**
- sigmoid(logits) > 0.5 → predicted mask; each cell takes the value of the pixel it falls in.
- Cascade: in-gate cells become the parent for child gates.
- flow-er addition: keep the probability per cell, and trace the 0.5 contour into an editable polygon.

**Evaluation (paper):** per-cell accuracy, precision, recall, F1 per gate; Pearson correlation
of predicted vs expert population proportions.

## 2. flow-er extensions (separate rows; keep UNITO-as-published unchanged for comparison)

| Extension | What changes | Why |
| --- | --- | --- |
| **Pretrained encoder** | `smp.Unet("resnet18", encoder_weights="imagenet", in_channels=C, classes=1)`; pad 101→128 | Starts from downloaded weights |
| **Shared template-refining U-Net** (main flow-er model) | One model for all gates. Inputs: [density image, template-gate mask, (cytokine gates) donor's unstimulated-well image]. Output: the gate adjusted to this sample. | Learns from every gate in the whole pool; can be tested zero-shot on new gates; matches the product (user's template adapted per sample) |
| **Unstimulated-well channel** | For cytokine gates, add the donor's negative well as an extra input channel | The information the expert and the classical baseline use |
| **Augmentation** | Random shifts/scaling of each axis in display space (drift), event re-subsampling per epoch | Robustness to staining/instrument drift |
| **Confidence** | Fold models from donor-level cross-validation form an ensemble; disagreement near the boundary → flag | Review flags in the app |

## 3. Evaluation rows (same test individuals for every row)

1. Template copied unchanged (moved through raw space)
2. Classical baseline (peak-shift; for cytokine gates, threshold from the donor's unstimulated well)
3. UNITO as published, per gate, trained on the benchmark training individuals only
4. Shared template-refining U-Net, **zero-shot** (pretrained on the pool, no benchmark training)
5. Shared template-refining U-Net, **fine-tuned** on the benchmark training individuals
6. Rows 4–5 with a mouse-only pretraining pool vs the mixed (human + mouse) pool

Metrics: per-cell F1 (wells with ≥ 20 expert-positive cells), population-frequency error in
percentage points (per well and per donor mean), background-subtracted response error for
cytokine gates, FlowKit-vs-FlowJo noise floor beside every number. Report the locked test plus
4-fold cross-validation by individual.

## 4. Rules

- Labels: FlowKit's per-cell memberships are the truth; FlowJo counts are the noise floor.
- Split by individual (person/mouse) across all datasets; no individual on both sides; controls
  (FMO, single stains, beads, unstained) never in training.
- The same image recipe in training and inference; store it (and the transform) with each model.
