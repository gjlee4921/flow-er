# flow-er: functional spec and build plan

Draft for review · 17 October 2026 · Goes with the PRD and the agent tech spec (Confluence).
Owners: app and agent (Eun), gating models (Gijeong), domain review (Will).

This plan covers what comes next:

- workflows you can run (n8n-style);
- gating models in those workflows;
- planning, scripts and web search;
- a results page that reads like a short paper, with custom figures;
- a chat that learns about the user;
- switching projects;
- the start page and its upload bug.

Each section gives the outcome, the design, what is in v1 and what comes later, and how we will know it works.

## Where we are

The web app and agent live on `feat/web-agent`, which is not merged yet.

| Area | Today | What this plan adds |
| --- | --- | --- |
| Chat agent | Loop on the OpenAI Responses API with 16 tools. Every change is a draft with keep/undo, and numbers come from saved tables. | Web search, memory of the user, script tools |
| Pipeline | A plan of steps (6 ops) on a hand-made canvas; steps run one at a time | Steps don't pass data to each other. There are no triggers, no run history and no per-node output, and written macros never run. |
| Gating models | `workspace/models.py` reports "not ready" | Waiting on `flow_er.ml` (Gijeong) |
| Data lab | Real events, FlowJo gates, drawing gates, walk-through | Feeds the gating-strategy figure |
| Results | 7 hand-written chart types over saved tables | No paper structure, no custom visuals, not publication quality |
| Start page | Lists the folders already in `DATA_DIR` | No upload (the bug); thin |
| Projects | You can only switch by going back to the start | The project bar does nothing |
| Tests | 6 backend tests; local end-to-end 8/8 | New cases in every section below |

## Rules that stay

- Every change is a draft until kept, and undo is one click.
- The LLM never places gates or produces numbers. Numbers come from tables that code writes.
- Steps are deterministic given their seed, so a run can be replayed and checked.
- Data, models, workspaces and the scientist's files stay out of git.
- Reused code is MIT, BSD, Apache-2.0 or ISC. Reused drawings are CC BY or CC0, and they are credited.

## Build order

| # | Work | Size | Needs | Before the 18 Oct demo? |
| --- | --- | --- | --- | --- |
| 1 | Start page and upload fix (§7) | S–M | — | Yes |
| 2 | Project switcher (§6) | S | — | Yes |
| 3 | User memory (§5) | M | — | Basic version |
| 4 | Web search, scripts, sandbox (§3) | M | — | Web search only |
| 5 | Workflows: graph, executor, canvas (§1) | L | 4 for script nodes | Canvas and run status, if time allows |
| 6 | Results: paper standard and figures (§4) | L | 5 for per-run tables (works on today's tables first) | One golden figure, if time allows |
| 7 | Gating model nodes (§2) | M | Gijeong's `ml/` API | No |

Sizes: S is half a day or less, M is 1–2 days, L is 3–5 days. The demo column says what could land before Sunday without putting the demo at risk.

---

## 1. Workflows: the pipeline as n8n-style graphs

**Outcome.** The plan is a graph you can run:

- Each node is a real step with typed inputs and outputs, and running it writes files the next node reads.
- You can run the whole graph, run up to a node, or run one node.
- New files can trigger a run.
- Every run records what each node produced.
- The chat and the canvas edit the same graph.

### Model

- **`state/graph.json`** holds nodes as `{id, type, title, params, inputs: {port: "node.port"}, position, note}`. Edges follow from `inputs`, and today's `plan.json` migrates one step to one node.
- **Node types** live in `workspace/nodes/`, one file per type. Each declares:
  - typed ports;
  - a JSON Schema for its parameters;
  - a `version`;
  - `run(ctx, inputs, params) -> outputs`.

  This one definition does three jobs. The canvas builds its parameter form from it, the agent reads it as tool documentation, and validation uses it.
- **Port types:** `samples` (a set of FCS files), `events` (compensated, transformed, per sample), `gates`, `labels` (per-event membership), `table`, `figure`, `model` and `file`. Only matching types connect. Validation also checks for cycles and missing inputs.

v1 node catalogue:

| Node | In → out | Notes |
| --- | --- | --- |
| FCS files | — → samples | The dataset's files; the "new files" trigger lives here |
| FlowJo workspace | — → gates | |
| Uploads | — → file | Attachments from the chat |
| Compensate and transform | samples → events | FlowKit; spillover from the workspace or the file |
| Script | any → any | Written by the agent or the user; runs in the sandbox (§3) |
| Apply gates | events + gates → labels, table | Counts per gate per sample |
| Train / apply gate model, classical baseline | see §2 | Shown as "waiting on models" until `ml/` lands |
| Statistics | table → table | The test the study sheet implies (t-test, Mann–Whitney, ANOVA with correction), run by code |
| Figure | table → figure | A figure spec (§4) |
| Export | table or gates → file | Prism CSV, GatingML 2.0 (FlowKit), FCS subset |

### Running

- **Starting a run.** `POST /experiments/{id}/executions {mode: all | to | only | from, node?, trigger}` queues an execution, and a worker process runs the nodes in dependency order.
- **What each node records.** Its status (queued, running, done, cached, failed, blocked), timing, a one-line summary ("3 files · 395,172 events"), output artifacts, a log, and any error.
- **Live progress.** Server-sent events from `GET /executions/{id}/events`. The canvas and the chat both listen.
- **Artifacts.** Stored in `artifacts/<hash>/` with a manifest giving the type, node, execution, inputs and size.
  - Events: Parquet per sample (float32).
  - Tables: CSV and Parquet.
  - Gates: JSON, plus GatingML on export.
- **Cache.** The key is a hash of node type, version, parameters, seed and input hashes.
  - An unchanged node is reused and shown as "cached".
  - Editing one node re-runs it and everything downstream, nothing else.
  - This is also what makes replay checkable.
- **Triggers.**
  - Manual: Run workflow, Run to here, Run this node.
  - Agent: the `run_workflow` tool, which calls the same API.
  - New files: watch the dataset folder, and run the graph on new FCS files only. This is the PRD's "data comes off the cytometer". A workflow with this trigger on is *active*, n8n's term.
  - Schedules come later.
- **Worker.** A separate process (`worker/`) with a job table in SQLite (`db/`). It runs one execution per project at a time and can cancel. If the backend dies mid-run, the execution is marked failed, and nothing in `state/` has changed.

### Canvas

- **Libraries.** React Flow (`@xyflow/react` 12.12, MIT), with dagre (`@dagrejs/dagre`, MIT) laying the graph out left to right in stage columns: Data → Clean → Gate → Measure → Results.
- **Node card.** Icon, title, type and status chip, plus the summary line after a run. Handles are coloured by port type, a flower shows while a node runs, and the edge into a failed node turns red.
- **Agent drafts.** Added nodes are dashed, removed nodes ghosted and changed parameters marked. One Keep or Undo covers the whole edit.
- **Node panel.** Three panes as in n8n: Input | Parameters | Output.
  - Output previews: a table grid, a density thumbnail, the figure, or the log.
  - *Pin output*, which lets you iterate downstream without re-running upstream, comes later.
- **Toolbar.** Run workflow; Add node (a searchable menu, plus "Ask the agent for a step…"); Fit; zoom; minimap.
- **Executions tab.** Past runs, each with its trigger, time and per-node status, and which parameters changed since the last run.
- **Fix for today's canvas.** In the side panel, Fit shrinks nodes until they can't be read. The fix:
  - clamp the minimum zoom at 0.6;
  - open the canvas full screen when the graph is wide.

### With the chat

- **Agent tools on the graph:**
  - `plan_pipeline` puts the whole graph up as a draft.
  - `edit_pipeline` applies a patch: add, remove, connect, set parameters.
  - `run_workflow` takes a mode and a node.
  - `read_execution` returns statuses, summaries, errors and output tables.
  - `explain_node`.
- **Runs in the chat.** When the agent runs something, the chat shows a compact run card that opens the canvas, with node chips filling in as they finish. The canvas highlights the nodes the agent is touching.
- **From the canvas to the chat:**
  - a node's menu has *Ask about this node*;
  - a failed node offers *Fix with the agent*, which sends the error and log;
  - selected nodes become context chips in the composer.

### Acceptance

- Plan → keep → Run workflow on the 8-colour dataset: every node finishes, and the counts table matches FlowJo's.
- Changing one parameter re-runs that node and what is downstream; the rest show "cached".
- A new FCS file dropped into a watched folder runs the workflow for that file only.
- The agent can build, edit and run a workflow from the chat alone, and every edit is a draft.

---

## 2. Gating models in the workflow (needs Gijeong)

The model code is Gijeong's. We only call it: `ml/` and `cytometry/` are his to change, and `workspace/models.py` is the one file on our side that changes when they land.

**Calls we need, proposed so we can agree them:**

```python
registry.versions(gate_def) -> list[Version]        # every version, with metrics
registry.current(gate_def) -> Version | None        # the promoted one
unet.train.train_gate(pool, *, seed, budget, on_progress) -> Version   # metrics include baseline F1 on held-out mice
unet.infer.place_gate(version, parent_events_2d) -> (polygon, confidence)
baseline.place_gate(parent_events_2d, template) -> (polygon, confidence)
```

**Nodes:**

- **Train gate model.** A long job that streams progress (epoch, loss, F1 against the baseline) and outputs a model version.
- **Apply gate model.** A polygon and confidence per sample. Gates below the confidence threshold are flagged and appear first in the data lab.
- **Classical baseline.** Always available once `ml/baseline.py` exists.

**Rules:**

- Code decides whether a version is adopted: it must beat the baseline and the current version. The LLM explains the decision but does not make it.
- `train_model` starts a subagent with a brief (goal, label source, budget, acceptance). What it produces comes back as a draft model node with its metrics.

**Until then:** the nodes stay in the catalogue, marked "waiting on models".

**Acceptance (Figshare spleen):** train on young mice, apply to aged mice. Per-gate F1 against the baseline shows on the model node and in the results.

---

## 3. Planning, scripts and web search

**Outcome.** The planning agent turns the experiment and its files into a graph. The arithmetic steps are scripts it found or wrote, each checked, sandboxed and labelled with its source and licence. Web search serves that, and the scientist's questions too.

### Planning quality

Most of the macro problem is solved here.

- **Plans are graph patches (§1)**, checked by code before the scientist sees them.
- **Plan eval set: 12 briefs.**
  - The main ones: Will's Treg panel, Figshare spleen young vs aged, and 8-colour ICS.
  - Edge cases: no workspace, missing FMO controls, mixed panels.
  - Each brief lists the expected node types and the must-have steps.
  - Scores: valid on the first try, coverage, no invented node types, and asking when it should.
  - Run it before every prompt change.
- **Rule per step** (from the tech spec): arithmetic becomes a built-in node or a script; a judgement call with labels becomes a model; anything else, the agent asks.

### Web search

- **The tool.** OpenAI's hosted `web_search` tool in the Responses API.
- **Domains.** `filters.allowed_domains` limits code and methods lookups to: github.com, pypi.org, readthedocs.io, scikit-image.org, bioconductor.org, ncbi.nlm.nih.gov, biorxiv.org and isac-net.org.
- **Sources in the chat.** They come back as `url_citation` annotations, which the chat shows as numbered sources under the answer.
- **First spike.** Confirm that hosted web search works next to our function tools in one call, with `store=False` and our model. If it doesn't, fall back to a `search_web` function tool backed by a search API.
- **Web content is data.** It cannot keep anything or call tools on its own. Code from the web enters only as a script draft.

### Scripts as nodes

- **One subagent per script**, with four tools:
  - `web_search`;
  - `fetch_source`, which fetches a file from GitHub or a PyPI sdist on the server, capped in size, with the licence read from the repository or package metadata;
  - `write_script`;
  - `test_script`.
- **The contract.** One Python file with:
  - a provenance header: what it does, source URL and version, SPDX licence, which lines were adapted;
  - its ports declared at the top;
  - `def run(inputs, params) -> outputs`.

  Imports are limited to the pinned environment: numpy, pandas, scipy, scikit-image, flowkit, flowio and flowutils.
- **Before it can be kept:**
  - it parses;
  - its imports are allowed;
  - its licence is on the allowlist (MIT, BSD-2/3, Apache-2.0, ISC, PSF), or the scientist approves the exception;
  - it gives the known answer on a fixture;
  - it gives the same output hash on two runs.
- **Sandbox, v1 (laptop).**
  - A subprocess of the pinned environment's Python (`-I`), in a throwaway folder.
  - Inputs are read-only and outputs go to one folder.
  - No API keys in its environment.
  - A timeout and limits on CPU, memory and file size.
  - No network: a `sandbox-exec` profile on macOS, bubblewrap (`--unshare-net`, read-only binds) on Linux.
- **Sandbox, v2 (hosted).** nsjail, a container with `--network none`, or a microVM. Not RestrictedPython, which is not a security boundary.
- **The scientist's own scripts** use the same node. The editor shows the code, its provenance and its test result.

### Acceptance

- Ask for a QC step that drops events in Time spikes. The agent finds reference code and writes a script node with provenance; the test passes in the sandbox, keep adds it, and the run uses it.
- A script that opens a socket or reads `~/.ssh` fails in the sandbox.
- A GPL source is refused, with the reason.
- The plan eval scores at least 90% valid on the first try.

---

## 4. Results: a short paper with custom figures

**Outcome.** The results page reads like a short research paper about this experiment:

- a title that says what was found;
- a summary whose numbers you can click through to their source;
- numbered figures with proper legends;
- methods written from the workflow;
- a gating-strategy figure.

It is interactive on screen and exports to PDF and Prism. A new kind of figure is something you ask for in a sentence, and custom visuals such as a mouse anatomy map are part of the kit.

### 4.1 The report standard

| Part | What it holds | Where it comes from |
| --- | --- | --- |
| Title | One line on what was found, not what was done | Agent, from kept results |
| Summary | 3–5 sentences; every result number bound to a table cell | Agent text plus bound numbers |
| Key numbers | A strip of 3–5 (for example CD4+ as % of CD3+, and its range across files) | Bound numbers |
| Figures | 3–6, numbered. The title states the takeaway ("Figure 2 · IFNγ+ CD8+ T cells rise after stimulation"). The legend says what is shown, n, what the error bars are and which test. | Figure specs over saved tables; the legend's facts come from the spec |
| Methods | Samples, panel and reagents, instrument, compensation, gating strategy, software and versions, statistics (the MIFlowCyt items) | Generated from the graph, the file headers and the study sheet |
| Gating strategy | One panel per gate down the tree, each with its % | The data lab's plots, rendered to SVG |
| Data and code | Files with checksums, the tables, the workflow version, licences and credits for reused code and drawings | Generated |
| References | FlowKit, the models, the methods used | citation-js with CSL styles |

**Bound numbers.** Report text refers to table cells, for example `{{population_stats | gate=CD4+ | sample=E01_008 | pct_parent}}`. The page shows the value, and hovering shows the table, row and step that produced it. A lint flags any percentage or decimal in the text that is not bound. This carries "numbers come from saved tables" into the prose.

### 4.2 The figure standard (enforced by lint)

- **Size.** On Nature's grid: one column is 89 mm, two columns 183 mm, with 7–9 pt text at print size.
- **Fonts.** Figures in Inter, text in Source Serif 4 (both SIL OFL).
- **Colour.**
  - Okabe–Ito for groups, so it is safe for colour-blind readers.
  - Each group keeps one colour across every figure, fixed in the study sheet.
  - Densities use a perceptual sequential scale.
- **Show the data.** When n < 20, show the points; no bar-only "dynamite" plots. Use mean ± SD or median with IQR, and say which in the legend.
- **Axes.** Use a log axis when values span more than two decades. A frequency says "% of parent" and names the parent.
- **Labels.** Label series directly instead of a legend when there are four or fewer. Multi-panel figures get panel letters a, b, c.
- **Statistics.** Brackets and p-values come from a Statistics node's table and are never typed in.
- **3-D.** Only where the third dimension is anatomy, and only on screen. The PDF uses the 2-D map.

### 4.3 How figures are made: three tiers

| Tier | What | Who writes it | Covers |
| --- | --- | --- | --- |
| A · Grammar | Vega-Lite 6 JSON specs over a saved table, styled by one house theme | The agent writes the spec; code checks it against the Vega-Lite JSON Schema and our lint | Bars, dot and strip plots, box and violin, lines, areas, heatmaps, dumbbells, forest plots, small multiples, scatter |
| B · Figure kit | React components with typed props: anatomy map, gating strategy, group comparison with brackets, 3-D organ view, study-design schematic, plate map | We build the components; the agent only fills in props (which table, columns, organ ids) | The custom visuals that made the mockup |
| C · Custom code (later) | A figure the agent writes as code, run in a sandboxed iframe with no network and data passed in from a saved table | The agent; the scientist keeps it like a step | Anything A and B cannot draw |

**Why Vega-Lite for tier A:**

- It is a JSON grammar with a published schema. An LLM can write it, and code can reject a bad one before anyone sees it.
- Its selections and parameters give the interaction we want without code: tooltips, hover rules, legend toggles, brushing and linked highlights.
- It is BSD-3.
- `vl-convert-python` renders the same spec to SVG or PNG on the server, for the PDF and for the agent to check its own figure.

**The alternatives we compared:**

- ECharts is more interactive, but its options have no formal schema and a much larger surface.
- Observable Plot looks good, but a chart is JavaScript, not data, and it has no built-in interaction.
- Plotly is heavier and harder to bring to a house style.

The mockup's best figures, the anatomy map and the gating strategy, are not grammar charts at all, which is why tier B exists.

**Interaction on every figure:**

- a tooltip shows the table row;
- clicking a legend entry hides that series;
- lines and areas get a hover guide;
- clicking a mark selects that sample, gate or organ and highlights it in every figure on the page (one shared selection, bridged from Vega's signals to our store);
- *Open in data lab* from any mark that is a sample and gate;
- a Data tab under each figure shows the rows used.

### 4.4 Open resources for custom visuals

All licences below were checked.

| Visual | Resource | Licence | How we use it |
| --- | --- | --- | --- |
| Mouse anatomy map | EBI Expression Atlas anatomograms (`@ebi-gene-expression-group/anatomogram` 2.4.0): mouse male, female and brain SVGs, every tissue tagged with an UBERON id. The body views have 51 tissues, including spleen, thymus, lymph node, bone marrow, lung, liver, gut, kidney, heart and brain. | Code Apache-2.0; drawings CC BY 4.0 | Tier B map: organs coloured or sized by a table column, with hover and click per organ. Our tissue names map to UBERON ids. Blood and skin are not drawn, so they get markers beside the body. A credit line is added automatically. |
| Study-design schematic | Servier Medical Art (mouse, tubes, lab items) and SciDraw drawings | CC BY 4.0 (SciDraw: per drawing, check each) | A "how this experiment was done" panel built from the study sheet |
| 3-D organ view | three.js and our own organ placement, as in the mockup | MIT | On screen only, for exploring |
| Server-side figure rendering | `vl-convert-python` 1.9 | BSD-3 | PNG and SVG for the PDF and for figure checks |
| Interactive charts | `vega` 6.4, `vega-lite` 6.5, `vega-embed` 7.3 | BSD-3 | Tier A, loaded only on the Results page |
| Print and PDF | Paged.js 0.4 | MIT | PDF from the same HTML, with page layout |
| Journal formats (later) | MyST (`mystmd`) or Typst | MIT / Apache-2.0 | Export to journal templates, Word and JATS |
| Maths, references | KaTeX, citation-js with CSL styles | MIT | Units and formulas; the reference list |

BioRender is excluded: it is proprietary, and its icons cannot ship in an open-source app.

### 4.5 The agent's figure loop

1. Choose from what is kept: which table, and which comparison the study sheet implies.
2. Write a tier A spec or tier B props.
3. Code checks the schema, that the fields exist in the table, that the aggregations are valid, and the lint in §4.2.
4. Render to PNG on the server, and have the model check it against a short list:
   - labels are readable and don't overlap;
   - axes and units are right;
   - the legend matches;
   - the caption says what is on the page.

   It gets at most one revision.
5. The figure lands in the report as a draft with Keep/Undo, and the numbers in its caption are bound.

### 4.6 Page and export

- **Layout.** One reading column of about 720 px, with figures breaking out to full width. Legends go in the margin on wide screens, and a sticky outline lists Summary, Figures, Methods and Data.
- **Per figure:** expand; data; export SVG, PNG (300 dpi) or Prism CSV; *Ask about this figure*.
- **Whole report:** PDF, the tables as a zip, and one Prism file per figure.
- **Drafts** are dashed in place with Keep/Undo. A kept figure is a spec, not a picture, so it redraws from the new tables after the next run.

### 4.7 Acceptance: the golden report

- **The mockup's figures, rebuilt from real tables:**
  - a dumbbell on a log axis;
  - a distribution with a hover guide and legend toggles;
  - an anatomy map with linked selection;
  - per-sample bars;
  - a numbers table;
  - the key-number strip.

  First on the 8-colour ICS data, then on Figshare spleen (young vs aged) once it is wired in.
- **Every number traces to a table cell.** An automated check walks the page and resolves every bound number.
- **Lint passes**, and the golden figures have visual snapshots.
- **The PDF** of the golden report fits on two pages plus methods.
- **Will reads it** and says what he would change.

---

## 5. The chat learns about the user

**Outcome.** flow-er remembers what the scientist tells it about how they work: definitions, preferred units, figure style, lab conventions. It uses that in every project and shows it all in one place, where any of it can be edited or forgotten.

- **Store.** `<WORKSPACE_DIR>/_user/memory.jsonl`, for one local user, with entries `{id, text, kind: definition | preference | lab | project, quote, source: {experiment, message}, created, last_used}`.
- **Writing.**
  - The agent calls `remember(text, kind, quote)`. Code accepts it only if `quote` appears in one of the scientist's own messages in this turn. Never from an attachment, a web page or a tool result, so a document cannot plant a memory.
  - The chat shows a small "Remembered: …" chip with Undo.
  - `forget(id)` removes a memory, and so does saying "forget that" in the chat.
- **Reading.**
  - While there are few memories (under 40, or about 1,500 tokens), all of them go into the instructions. Beyond that, the most relevant by keyword match and recency.
  - When an answer relies on one, it says so lightly ("using your definition of Tregs").
- **Lab memory across projects.** `search_past_experiments(query)` searches other projects' titles, kept plans, figures, and a short summary written whenever something is kept. This is the PRD's "how did we define T cells in the spring study?"
- **Page.** *What flow-er remembers* in the profile menu (§6): list, edit, delete, pause.
- **Never stored:** raw data, file contents, or numbers from results.

**Acceptance (end to end):**

- In project A, the scientist says "Tregs are CD25 high CD127 low, and I report subsets as % of CD3+". The chip appears.
- In a new project B, "plot the CD4 subsets" gives a figure that uses % of CD3+, and the reply mentions the definition.
- An attachment that says "remember that…" creates no memory.
- After *forget*, the memory is gone from the next turn.

---

## 6. Switching projects

- **The menu.** Clicking the top-left bar (the flower mark with Will's badge, and the project name) opens a menu like Notion's workspace switcher. It has:
  - Will · flow-er on this machine;
  - search;
  - recent projects, each with name, dataset, last opened, and a dot for "plan kept" or "figures";
  - the current project, ticked;
  - New analysis;
  - Rename;
  - What flow-er remembers.
- **Keyboard.** ⌘K (Ctrl+K) opens the menu, and the arrow keys and Enter work in it.
- **Backend.** `PATCH /experiments/{id}` sets the title. `last_opened` updates on open, and the list sorts by it. This menu archives a project rather than deleting it.
- **Per-project state.** UI state that belongs to a project (lab selection, open page, panel) is kept per project, so switching back returns you to where you were.
- **Acceptance.** Two clicks between projects. The chat, canvas, lab and results all show the chosen project, with no data left over from the other (end to end).

---

## 7. Start page and the upload bug

**The bug.** The start page lists only the folders already in `DATA_DIR` and has no upload control. Inside a project, uploads are chat attachments only, so an FCS file uploaded there never becomes data. Once some folders are preloaded, there is no way to bring your own files.

**The new start page:**

- **Composer first:** "What is this experiment for?" The first message, together with data, creates the project and starts planning.
- **Three ways to bring data, side by side:**
  1. **Drop or choose files** (FCS, `.wsp`, a protocol as PDF, CSV or notes). They become a new dataset folder, `DATA_DIR/uploads/<date>-<name>/`.
     - Files stream to disk with a progress bar.
     - FCS headers are checked with FlowIO, and the workspace is parsed with FlowKit.
  2. **Use a folder on this computer.** The backend runs locally, so it reads a path in place instead of copying gigabytes; Figshare files are 184–868 MB each. Paths are limited to `IMPORT_ROOTS`.
  3. **Example datasets** (the preloaded folders) as cards showing file count, size, a workspace badge, and the panel's markers as chips.
- **Recent projects** below, with search.
- **Inside a project,** FCS files dropped into the chat become a draft: "Add 2 files to this experiment?" This is the PRD's "new FCS files join the run".
- **Backend:**
  - `POST /datasets`: multipart, streamed, with a configurable size limit;
  - `POST /datasets/link {path}`;
  - `POST /experiments/{id}/samples`, as a draft.

**Acceptance:**

- With preloaded datasets present, upload two FCS files and a workspace from the start page. A new card appears, and the project opens with its gates.
- A 1 GB file uploads without the server holding it in memory.
- A non-FCS file named `.fcs` is rejected with a plain message.

---

## 8. Testing

- **Backend unit tests** go in `backend/tests/`, not `tests/ml/`, which is Gijeong's. They cover:
  - graph validation and cache keys;
  - executor order and failure handling;
  - sandbox denials;
  - the licence check;
  - the memory quote check;
  - bound-number resolution;
  - figure lint;
  - upload validation.
- **Agent evals.** Real model calls, run locally before prompt changes:
  - plan eval (§3);
  - figure eval: on 10 asks, the spec is valid, the lint is clean and the visual check passes;
  - memory eval (§5).

  Track cost and seconds per turn.
- **End to end.** The local Playwright harness (kept out of git, as now) gains one case per section:
  - build and run a workflow;
  - a script found with web search;
  - the golden report;
  - memory across projects;
  - switching projects;
  - uploading from the start page.
- **Visual snapshots** of the golden figures.

## 9. Decisions to record once agreed

| Decision | Reason |
| --- | --- |
| The pipeline is a typed graph of node types, each defined once: ports, parameter schema, run | The canvas, the agent and validation read the same definition |
| Node outputs are content-addressed artifacts, and unchanged nodes are reused | Partial re-runs, replay checks and speed |
| Figures are Vega-Lite specs or figure-kit props; agent-written code runs only in a sandboxed iframe | Checked before shown, interactive, renderable on the server |
| Numbers in report text are bound to table cells | Carries "numbers come from saved tables" into the prose |
| Memories are written only from the scientist's own words | Documents and web pages cannot plant instructions |
| Scripts run with no network and no keys | Code an LLM wrote is untrusted by default |

## Open questions

- **Gijeong:** are the `ml/` calls in §2 right? When can `train_gate` and `place_gate` be called?
- **Will:**
  - Which figures does he rebuild by hand today?
  - What does a good results page look like to him?
  - Is the short-paper format right for a lab meeting, or does he also want a slide view?
- **Model:** keep OpenAI for the demo and add Claude after, or switch now?
- **Hosting:** which sandbox (nsjail, container or microVM) if flow-er runs anywhere other than a laptop?
- **Organ names:** do UBERON ids come from our own table or from the study sheet?

## Sources

- OpenAI web search tool (Responses API): https://developers.openai.com/api/docs/guides/tools-web-search
- React Flow: https://reactflow.dev/ · v12 migration notes: https://reactflow.dev/learn/troubleshooting/migrate-to-v12
- VegaChat, LLM-generated Vega-Lite and its validation: https://arxiv.org/html/2601.15385v1
- Vega-Lite as a model-authored chart format: https://github.com/rummand/nexus/issues/219
- vl-convert-python: https://pypi.org/project/vl-convert-python
- EBI anatomogram (code Apache-2.0, drawings CC BY 4.0): https://www.npmjs.com/package/anatomogram · https://www.ebi.ac.uk/gxa/licence.html
- Servier Medical Art (CC BY 4.0): https://smart.servier.com/category/general-items/animals/rodents
- SciDraw drawings on Zenodo (CC BY 4.0 per record), for example https://zenodo.org/records/3925915
- MyST Markdown: https://mystmd.org/guide · Typst: https://en.wikipedia.org/wiki/Typst
- MIFlowCyt (ISAC reporting standard): https://pubmed.ncbi.nlm.nih.gov/18752282/
- Sandboxing LLM-written Python (nsjail, bubblewrap, containers): https://www.morphllm.com/nsjail-sandbox · https://johnstowers.co.nz/2026/04/15/sandbubble/
