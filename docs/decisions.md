# Decisions

| Decision | Reason |
| --- | --- |
| Per-gate U-Net (UNITO-style) as the main model | Reproduces 2D expert gates as editable polygons; trains on CPU from ~10 gated samples |
| Splits by mouse/donor (and date), never by cell | Cells from the same animal in train and test inflate scores |
| Data and trained models kept outside git | Size, and donor data must never be published |
| Package name `flow_er` | `flower` is an existing PyPI package (Celery's monitor) |
| Response baseline = the well named in meta `unstim_source`, same method and gate; no response when that well's prediction is missing | Exact donor pairing from the cache; never mixes methods |
| Frequencies and errors are computed on the cached parent cells (the 200k subsample when a parent is larger); precision/recall are NaN when their denominator is 0 | Same cells the models are scored on; ALS keeps every cell |
| U-Net input normalisation: log1p(count) / max of the image, per channel (density and unstim alike); template stays 0/1; no ImageNet mean/std for the pretrained encoder | Spec only says "count image"; scale-free across event counts; one rule for every model, stored in each recipe |
| Shared-model channels: [density, template, unstim] with unstim = zeros for non-cytokine gates; the `_1ch` variant drops unstim ([density, template]) | One fixed input layout for every gate; the variant isolates what the unstimulated well adds |
| Augmentation (shared model, training only): per-axis shift ±5 px and scaling ±10% about the centre, nearest-pixel, same map for density, unstim and mask (density/unstim zero-filled, mask edge-replicated); event re-subsampling = binomial thinning of the counts with keep p ~ U(0.5, 1) | Equivalent in distribution to subsampling cells, without reading cell_bins every step; mask keeps gates that run to the plot edge |
| Pretraining weights datasets equally per epoch (sampling with replacement, 1/files-of-dataset); early stopping and grid selection use the mean per-image Dice of sigmoid > 0.5 (1 when both masks are empty) | split_v1 pretrain note; Dice per image as in UNITO's selection |
| Polygon = largest closed 0.5 contour of the probability map padded with zeros, simplified with a 0.5-pixel tolerance | Edge-touching gates still close; fewer vertices to edit |
| Labels = FlowKit per-cell memberships; FlowJo's saved counts are the noise floor beside every number | A time-boxed biex investigation could not make FlowKit reproduce FlowJo (ALS IL-4/IL-17, spectral #4/#5) |
| Shared display transform for every dataset: linear x / $PnR for scatter, logicle (T = $PnR, W = 0.5, M = 4.5, A = 0) for fluorescence; pixel = round(100 v) | Workspaces use different biex settings; one transform makes images comparable; masks map pixel centres back into each workspace's display space |
| Every method is scored gate by gate on the expert parent cells (no cascade) | Isolates each gate; cascading is a later product step |
| ALS template = donor 2008's unstimulated well (first training donor); other datasets = their first training sample | Simplest fixed reference, always on the training side |
| Classical peak-shift uses the cached 101x101 parent density, two 3x3 box-smoothing passes, window = template bounding box + 50% per side; < 120 parent cells keep the template | The prototype's peak2d on the shared image grid |
| Classical cytokine threshold: shift the template along x until the donor's unstimulated well has the background rate = median expert positive fraction of the training (train + finetune) donors' unstimulated wells; bisection within ±0.4 display units; same shift for the donor's three wells | How experts set ICS thresholds; one rate per gate fitted on training donors only |
| UNITO per gate and the fine-tuned shared model train on the ALS train + finetune donors (8), early stopping on the 2 val donors | Spec rows 3/5 say "benchmark training individuals"; the finetune role adds screen-4 donors to training |
| Step-4 UNITO budget: grid points capped at 1.5 min (max 40 epochs, patience 8) on 12 CPU threads | The spec's 31 M-parameter U-Net takes ~1.5 s per batch of 8 on CPU; uncapped the grid would exceed the 1-1.5 h limit |
| Shared-model pretraining uses only gates whose FlowKit counts agree with FlowJo (registry `agrees_with_flowjo`) and no Time gates | split_v1 pretrain rule; disagreeing gates would teach the wrong boundary |
