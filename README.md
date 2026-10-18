# flow-er

Flow cytometry gating that learns each lab's own gates from its data, with a review screen for the expert and an agent that plans the analysis with you.

## Layout

```
backend/    FastAPI app + training/inference worker (Python, uv)
frontend/   React + TypeScript (Vite, pnpm)
docs/       architecture, decisions, specs
```

Data, trained models and experiment workspaces live outside the repo, next to it in `../data`, `../models` and `../workspaces` (override with `DATA_DIR` / `MODELS_DIR` / `WORKSPACE_DIR`, see `backend/.env.example`).

## Docs

- [architecture.md](docs/architecture.md): where things are
- [spec-plan.md](docs/spec-plan.md): functional spec and build plan for the app
- [handover.md](docs/handover.md): wiring the gating models into the app (files outside git, set-up, calls)
- [models.md](docs/models.md): the gating models, the runtime (FCS in, gates out) and per-lab fine-tuning
- [runtime-api.md](docs/runtime-api.md): exact JSON shapes for the gating runtime and the endpoints they back
- [decisions.md](docs/decisions.md), [cache-format.md](docs/cache-format.md)

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

Put a folder of FCS files (with its FlowJo workspace, if there is one) under `DATA_DIR`; it appears on the start screen.

## The agent

The chat plans the analysis, runs it, writes figures from saved tables and points the data lab at what you ask about. Everything it changes is a draft you keep or undo, every number it quotes comes from a saved table, and it never places gates: those come from the FlowJo workspace, from you drawing them in the data lab, or from the gating models.

The same tools are served over MCP for Claude Code or any MCP client:

```bash
cd backend && uv run python -m flow_er.mcp.server
```

## Gating models

`flow_er.ml` gates an FCS file with a lab's template (`ml.autogate`), takes the expert's edits, and fine-tunes per lab from accepted gates (`ml.add_labels`, `ml.retrain`); see [models.md](docs/models.md). Command line: `cd backend && uv run python scripts/autogate.py --help`.

## Tests

```bash
cd backend && uv run pytest -q                            # everything
cd backend && uv run pytest tests/test_workspace.py -q    # the app's workspace only
```

The workspace tests run on real FCS files without calling the model; they skip if the test dataset is not under `DATA_DIR`. The split and cache checks skip without `DATA_DIR/_splits`; the model tests use synthetic data.
