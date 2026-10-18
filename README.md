# flow-er

Flow cytometry gating that learns each lab's own gates from its data, with a review screen for the expert and an agent that plans the analysis with you.

## Layout

```
backend/    FastAPI app + training/inference worker (Python, uv)
frontend/   React + TypeScript (Vite, pnpm)
docs/       architecture and decisions
```

Data, trained models and experiment workspaces live outside the repo (see `.env.example`).

## Run (development)

```bash
# backend: copy .env.example to backend/.env and set OPENAI_API_KEY for the agent
cd backend && uv sync && uv run uvicorn flow_er.main:app --reload --app-dir src

# frontend
cd frontend && pnpm install && pnpm dev
```

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

## Tests

```bash
cd backend && uv run pytest tests/test_workspace.py -q
```

These run on real FCS files without calling the model; they skip if the test dataset is not under `DATA_DIR`.
