# Architecture

Full product description: the flow-er PRD (Confluence, "Reinvent the wheel" page). The agent's design: the flow-er agent tech spec (a child page of the PRD).

- **backend/** — FastAPI app (`src/flow_er/main.py`) plus a separate worker process for training and inference.
  - `cytometry/` reads FCS files and FlowJo workspaces with FlowKit.
  - `ml/images.py` is the single recipe for turning a 2D plot into a model input; training and inference both use it.
  - `ml/registry.py` keeps every model version and only promotes one that beats the classical baseline and the current version.
  - `workspace/` is the tool layer: one folder per experiment (`WORKSPACE_DIR`) holding its plan, gates, views, macros, saved tables, journal and chat. Every change is a draft until kept.
    - `data.py` reads a dataset (FlowKit for now; it switches to `cytometry/` once that has its loaders).
    - `lab.py` gives the gate tree with counts, events to draw, and the population table.
    - `plan.py` validates plans and runs their steps; `views.py` validates figure specs and resolves their rows from a saved table.
    - `models.py` is the seam to `ml/`: it reports honestly what the gating models can do and is where training and inference get wired.
  - `llm/` is the agent: `client.py` (provider adapter; OpenAI Responses API), `tools.py` (the tool registry), `prompts.py`, `agent.py` (the turn loop and subagents).
    - The data lab has its own assistant on the same loop: its own thread (`lab_chat.jsonl`), its own prompt, and only the tools that move the lab and read numbers (`LAB_TOOLS`), so it cannot change the analysis.
  - `mcp/server.py` serves the same tools over MCP.
- **frontend/** — React + TypeScript (Vite). The chat is in the middle, the plan in the sidebar, and the workflow panel on the right shows the pipeline, results, measurements and models; the data lab takes the whole screen, with a plot, a stats sheet, saved diagrams and its own assistant. The API client is hand-typed for now; `pnpm gen:api` can generate it from the backend's OpenAPI schema.
- **Data** lives outside the repo (`DATA_DIR`), currently `../data/figshare_spleen` (11 FCS files + `Spleen_Y_A_wsp_file.wsp`).
