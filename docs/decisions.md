# Decisions

| Decision | Reason |
| --- | --- |
| Per-gate U-Net (UNITO-style) as the main model | Reproduces 2D expert gates as editable polygons; trains on CPU from ~10 gated samples |
| Splits by mouse/donor (and date), never by cell | Cells from the same animal in train and test inflate scores |
| Data and trained models kept outside git | Size, and donor data must never be published |
| Package name `flow_er` | `flower` is an existing PyPI package (Celery's monitor) |
