# flow-er

Flow cytometry gating that learns each lab's own gates from its data, with a review screen for the expert and a Claude-powered assistant.

## Layout

```
backend/    FastAPI app + training/inference worker (Python, uv)
frontend/   React + TypeScript (Vite, pnpm)
docs/       architecture and decisions
```

Data and trained models live outside the repo (see `.env.example`).

## Run (development)

```bash
# backend
cd backend && uv sync && uv run uvicorn flow_er.main:app --reload --app-dir src

# frontend
cd frontend && pnpm install && pnpm dev
```
