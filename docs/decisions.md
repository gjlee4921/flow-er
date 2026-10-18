# Decisions

| Decision | Reason |
| --- | --- |
| Per-gate U-Net (UNITO-style) as the main model | Reproduces 2D expert gates as editable polygons; trains on CPU from ~10 gated samples |
| Splits by mouse/donor (and date), never by cell | Cells from the same animal in train and test inflate scores |
| Data and trained models kept outside git | Size, and donor data must never be published |
| Package name `flow_er` | `flower` is an existing PyPI package (Celery's monitor) |
| Every change the agent makes is a draft until kept | Nothing in an analysis changes without the scientist seeing it; undo is always one click |
| The agent never places gates or produces numbers | Gates come from FlowJo, the scientist or the models; numbers come from saved tables, so every figure traces to a file |
| Figures are specs over saved tables, drawn by the browser | A new way of showing a result can be asked for in a sentence without running generated code, and stays interactive |
| The web app's agent runs on OpenAI's Responses API with `store=False` | Function tools with reasoning models need it; nothing is kept on OpenAI's side. The adapter is one function, so Claude can replace it |
| The data lab has its own assistant, apart from the main chat | Looking and laying out is a different job from planning: a separate thread keeps both short, and lab-only tools mean it can never change the analysis |
