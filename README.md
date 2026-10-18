# flow-er

Flow cytometry gating that learns each lab's own gates from its data, with a review screen for the expert and an agent that plans the analysis with you.

- **Gates** are placed by one small U-Net per gate, reimplemented from the [UNITO](https://pmc.ncbi.nlm.nih.gov/articles/PMC11821879/) paper (Chen, Ionita et al., *Nature Communications* 2025). A gate without a model falls back to the lab's template, flagged for review. Every gate carries a confidence score; gates below the lab's calibrated threshold are flagged.
- **Each lab's models learn from the gates its scientists accept.** Retraining keeps, per gate, the best of the current model, the new model and template copy. A new version is used only if it does better on people it was never taught.
- **The agent** plans and runs the analysis, writes figures and scripts, and writes up. It never places a gate or produces a number.

## Results

All errors are mean |model % of parent − expert %|, in percentage points (pp); lower is better.

| Test | Template copy | flow-er |
| --- | --- | --- |
| 6 unseen human donors (ALS ICS, different screen): 7 gates every method covers | 1.08 pp | **0.93 pp** |
| The same donors, 4 cytokine gates | 0.65 pp | **0.27 pp** |
| New mouse tissue: the starting models, then three rounds of the app's own teach-and-retrain loop, scored on never-taught mice | 3.00 pp | 25.1 → 5.3 → 4.0 → **1.35 pp** |

The full numbers, the models we tried and dropped (a shared U-Net, CytoBERT) and what the results do not show are on the results page below.

## Docs

In this repo:

- [architecture.md](docs/architecture.md): where things are
- [spec-plan.md](docs/spec-plan.md): functional spec and build plan for the app
- [handover.md](docs/handover.md): wiring the gating models into the app (files outside git, set-up, calls)
- [models.md](docs/models.md): the gating models, the runtime (FCS in, gates out) and per-lab fine-tuning
- [unito-spec.md](docs/unito-spec.md): the model specification, written from the paper
- [runtime-api.md](docs/runtime-api.md): exact JSON shapes for the gating runtime and the endpoints they back
- [decisions.md](docs/decisions.md), [cache-format.md](docs/cache-format.md)

## Layout

```
backend/    FastAPI app + training/inference worker (Python, uv)
frontend/   React + TypeScript (Vite, pnpm)
docs/       architecture, decisions, specs
```

Data, trained models and experiment workspaces live outside the repo, next to it in `../data`, `../models` and `../workspaces` (override with `DATA_DIR` / `MODELS_DIR` / `WORKSPACE_DIR`, see `backend/.env.example`). The app's start page lists only `../data/demo` (`DATASETS_DIR`): files no model was trained on. Donor data, trained weights and the splits never go in git.

## Run (development)

```bash
# backend: copy backend/.env.example to backend/.env and set OPENAI_API_KEY for the agent
cd backend && uv sync && uv run uvicorn flow_er.main:app --reload --app-dir src

# frontend
cd frontend && pnpm install && pnpm dev
```

`uv sync` installs the app and the gating models (PyTorch, CPU build). For a light install without the models: `uv sync --no-group ml`.

If 8000 or 5173 are taken on your machine, move both and point the frontend at the backend:

```bash
cd backend && uv run uvicorn flow_er.main:app --reload --app-dir src --port 8040
cd frontend && FLOW_ER_API=http://localhost:8040 FLOW_ER_PORT=5180 pnpm dev
```

Put a folder of FCS files (with its FlowJo workspace, if there is one) under `DATASETS_DIR`; it appears on the start screen.

## The agent

The chat plans the analysis, runs it, writes figures from saved tables and points the data lab at what you ask about. Everything it changes is a draft you keep or undo, every number it quotes comes from a saved table, and it never places gates: those come from the FlowJo workspace, from you drawing them in the data lab, or from the gating models.

It runs on OpenAI's Responses API (`OPENAI_MODEL`, default `gpt-5.5`) behind a one-function adapter, so Claude can replace it. Script steps are written by a subagent from reference code with a permissive licence and run in a sandbox (macOS `sandbox-exec` or Linux `bubblewrap`; they do not run on Windows).

The same tools are served over MCP for Claude Code or any MCP client:

```bash
cd backend && uv run python -m flow_er.mcp.server
```

## Gating models

`flow_er.ml` gates an FCS file with a lab's template (`ml.autogate`), takes the expert's edits, and fine-tunes per lab from accepted gates (`ml.add_labels`, `ml.retrain`, on the CPU or a Colab GPU); see [models.md](docs/models.md). Command line: `cd backend && uv run python scripts/autogate.py --help`.

The weights live in `../models` (see [handover.md](docs/handover.md)). Lab `als` (human ALS ICS panel) uses 11 per-gate models trained on a Colab GPU. Lab `mo280` (mouse) starts each of its 33 gates from the human model with the same kind of plot, and is adapted by retraining. The start page shows each lab as live, outdated (3 or more taught files the current version hasn't learned from) or offline.

UNITO's own code has a no-redistribution licence, so flow-er implements the model from the paper alone; nothing is copied or adapted from that repository.

## Tests

```bash
cd backend && uv run pytest -q                            # everything
cd backend && uv run pytest tests/test_workspace.py -q    # the app's workspace only
```

The workspace tests run on real FCS files without calling the model; they skip if the test dataset is not under `DATA_DIR`. The split and cache checks skip without `DATA_DIR/_splits`; the model tests use synthetic data.

## Licence

MIT. The datasets are CC BY 4.0 or CC0 (ALS ICS: Zenodo 16754591 and 16754609; MO280 and mouse spleen: Figshare); credit them when shown.
