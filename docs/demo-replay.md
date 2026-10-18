# Demo: retraining replayed from GJ's trained versions

Branch `demo/replay-retrain` only. Retraining on this laptop takes about 5–8 minutes per gate, so for
the recording the Retrain button and the agent's retrain do not train. They move the lab to its next
version, trained earlier on the GPU, and report that version's own recorded results, line by line, over
about 25 seconds:
- each gate's held-out error, for the new model, the current one and template copy;
- the route the gate took (new model, kept current model or template copy);
- the verdict (promoted or not, and why).

Gating is not replayed: each version's models run on the files as usual. Teaching (Keep, Teach the
models) is real too.

## Load the data

1. Put GJ's lab in `models/labs/mo280/` (outside git), as `ml/registry.py` writes it:
   - `template.json`: the mouse gate tree.
   - `v001/version.json`, `v002/version.json`, and so on, oldest first. `gates` maps each gate to its
     model folder (relative to `MODELS_DIR`), and those folders must be there with `model.pt` and
     `recipe.json`.
   - Each retrained version's `metrics` holds `promoted`, `reason`, `held_out_error_pp`
     (`current` / `candidate`) and `gates` (per gate: `route`, `held_out_error_pp`,
     `template_copy_pp`, `current_pp`, `n_held_out`), as `finetune.retrain` records them.
2. Start on v001, the human models: `echo v001 > models/labs/mo280/current`.
3. Check it: `GET /labs/mo280/models` lists the versions. Its `status` says `offline` if any model
   file is missing.

## Run

```bash
cd flow-er-demo/backend   # this branch's worktree
RETRAIN_REPLAY=1 REPLAY_SECONDS=25 uv run uvicorn flow_er.main:app --app-dir src --port 8041 --reload --reload-dir src
cd flow-er-demo/frontend && FLOW_ER_API=http://localhost:8041 FLOW_ER_PORT=5181 pnpm exec vite
```

For a clean home page, also set `WORKSPACE_DIR` (an empty folder) and `MODELS_DIR` (holding only labs
`als` and `mo280`, plus the model folders they point to). Set them as environment variables: they
override `backend/.env`.

In the app, on "mouse mo280 four organs":
1. Run. The models gate every file with v001 (pin the project to the lab with
   `PATCH /experiments/{id} {"lab": "mo280"}` if it does not pick it up).
2. Review a file.
3. Teach 2 mice. Retraining needs kept gates from at least 2 individuals.
4. Retrain. The replay ends on the next version, now current.
5. Run this step again: the same files, gated by that version.

Between takes, go back with `POST /labs/mo280/models/v001/current`. Each Retrain moves one version on;
when there is none left, it says so.

## Watch the models learn (rounds view)

This needs no model files. It reads the numbers of a fine-tuning run recorded earlier: each version's %
against the expert's, gate by gate, on samples no version was taught. It plays the rounds, v001 (the
human models) → v002 → v003 → v004:
- the mean error falls (lab mo280: 25.09 → 5.32 → 4.00 → 1.35 pp, against 3.00 for template copy);
- a curve per tissue;
- every gate's dot moves onto the expert's line. Pick one never-taught sample on the left to see its
  gates, or "Every sample" for each gate's mean error.

Load GJ's run once:

```bash
cd flow-er-demo/backend
uv run python -m flow_er.workspace.rounds ~/Downloads/loop_comparison/mouse_lab_mo280 mo280
```

This writes `models/rounds/mo280.json` (outside git). Open the view from the data lab's top bar or the
model step's panel: "Watch the models learn". It plays once by itself (about 3 s a round). Click a
round to hold it, or press Play again.
