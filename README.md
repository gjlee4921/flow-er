# flow-er

Flow cytometry gating that learns each lab's own gates from its data, with a review screen for the expert and a Claude-powered assistant.

## Layout

```
backend/    FastAPI app + training/inference worker (Python, uv)
frontend/   React + TypeScript (Vite, pnpm)
docs/       architecture, decisions, specs
```

Data and trained models live outside the repo, next to it in `../data` and `../models` (override with `DATA_DIR` / `MODELS_DIR`, see `.env.example`).

## Docs

- [architecture.md](docs/architecture.md): where things are
- [models.md](docs/models.md): the gating models, the runtime (FCS in, gates out) and per-lab fine-tuning
- [runtime-api.md](docs/runtime-api.md): exact JSON shapes and the endpoints they back
- [spec-plan.md](docs/spec-plan.md): functional spec and build plan for the app
- [decisions.md](docs/decisions.md), [cache-format.md](docs/cache-format.md)

## Run (development)

```bash
# backend: core packages, plus the ML stack (torch) and the web app
cd backend && uv sync --group ml --group api
uv run uvicorn flow_er.main:app --reload --app-dir src

# tests
uv run pytest

# frontend
cd frontend && pnpm install && pnpm dev
```

`uv sync` alone installs only the core packages (FlowKit, NumPy, pandas, scikit-image). Add `--group ml` for the models and `--group api` for the web app.
