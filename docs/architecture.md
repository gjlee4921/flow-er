# Architecture

Full product description: the flow-er PRD (Confluence, "Reinvent the wheel" page).

- **backend/** — FastAPI app (`src/flow_er/main.py`) plus a separate worker process for training and inference.
  - `cytometry/` reads FCS files and FlowJo workspaces with FlowKit.
  - `ml/images.py` is the single recipe for turning a 2D plot into a model input; training and inference both use it.
  - `ml/registry.py` keeps every model version and only promotes one that beats the classical baseline and the current version.
- **frontend/** — React + TypeScript (Vite). The API client's types are generated from the backend's OpenAPI schema.
- **Data** lives outside the repo (`DATA_DIR`), currently `../data/figshare_spleen` (11 FCS files + `Spleen_Y_A_wsp_file.wsp`).
