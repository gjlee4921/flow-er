"""The tool layer without the model: drafts, keep and undo, plans, views, and counts on real data.

Uses the dataset named by FLOW_ER_TEST_DATA (default: DATA_DIR/flowkit_8color, FlowKit's public
8-colour example with its FlowJo workspace). Skipped when that folder is not on this machine.
"""
import os
from pathlib import Path

import pytest

from flow_er.config import settings
from flow_er.llm import tools
from flow_er.workspace import lab, plan, store, views

DATA = Path(os.environ.get("FLOW_ER_TEST_DATA", str(settings.data_dir.expanduser() / "flowkit_8color")))
needs_data = pytest.mark.skipif(not any(DATA.rglob("*.fcs")) if DATA.exists() else True, reason=f"no FCS files at {DATA}")


@pytest.fixture()
def exp(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "workspace_dir", tmp_path)
    return store.create(DATA, "test")["id"]


@needs_data
def test_nothing_changes_until_kept(exp):
    steps = [{"id": "count", "op": "population_stats", "title": "Count", "inputs": ["fcs"], "outputs": ["counts"]}]
    store.set_draft(exp, "plan", plan.validate(steps), "a plan")
    assert store.state(exp, "plan") == []
    store.keep(exp)
    assert [s["id"] for s in store.state(exp, "plan")] == ["count"]
    store.set_draft(exp, "plan", [], "an empty plan")
    store.undo(exp)
    assert [s["id"] for s in store.state(exp, "plan")] == ["count"]


def test_plan_rejects_inputs_nothing_makes():
    with pytest.raises(ValueError, match="not made by an earlier step"):
        plan.validate([{"id": "a", "op": "population_stats", "title": "x", "inputs": ["nothing"]}])
    with pytest.raises(ValueError, match="op must be one of"):
        plan.validate([{"id": "a", "op": "guess", "title": "x"}])


@needs_data
def test_counts_match_flowjo_and_trace_to_the_table(exp):
    sid = lab.dataset(exp).sample_ids[0]
    cd3 = next(g for g in lab.tree(exp, sid) if g["key"].endswith("/CD3+"))
    rows = lab.population_table(exp)
    row = next(r for r in rows if r["sample"] == sid and r["gate_path"] == cd3["key"])
    assert row["count"] == cd3["count"] > 0
    assert lab.read_table(exp, "population_stats")[0].keys() == rows[0].keys()


@needs_data
def test_a_drawn_gate_counts_and_recounts_on_keep(exp):
    sid = lab.dataset(exp).sample_ids[0]
    cd3 = next(g for g in lab.tree(exp, sid) if g["key"].endswith("/CD3+"))
    lab.population_table(exp)
    store.set_draft(exp, "gate", {"key": f"{cd3['key']}/hi", "name": "hi", "parent": cd3["key"], "source": "user",
                                  "dims": cd3["dims"], "vertices": [[0, 0], [1, 0], [1, 1], [0, 1]], "kind": "rect"}, "a gate")
    drafted = next(g for g in lab.tree(exp, sid) if g["name"] == "hi")
    assert drafted["draft"] and 0 < drafted["count"] <= cd3["count"]
    d = store.keep(exp)
    lab.after_change(exp, d["kind"])
    assert any(r["gate"] == "hi" for r in lab.read_table(exp, "population_stats"))


@needs_data
def test_a_view_names_a_table_and_only_its_columns(exp):
    lab.population_table(exp)
    with pytest.raises(ValueError, match="not in population_stats"):
        views.validate(exp, {"title": "t", "type": "bar", "table": "population_stats", "x": "sample", "y": "made_up"})
    v = views.validate(exp, {"title": '"CD3+ by file"', "type": "bar", "table": "population_stats", "x": "sample",
                             "y": "pct_parent", "filter": {"gate": "CD3+"}})
    assert v["title"] == "CD3+ by file"
    assert len(views.data(exp, v)) == len(lab.dataset(exp).sample_ids)


@needs_data
def test_the_agent_cannot_place_a_gate(exp):
    assert "add_gate" not in tools.REGISTRY and "draw_gate" not in tools.REGISTRY
    r = tools.call(tools.Ctx(exp), "focus_lab", {"parent": "CD3+", "x": "CD4", "y": "CD8"})
    assert r["ok"] and r["parent"].endswith("/CD3+")


@needs_data
def test_the_lab_sheet_adds_up_and_the_lab_assistant_changes_nothing(exp):
    """Inside plus outside a gate is its parent; the lab assistant's tools only move the screen."""
    sid = lab.dataset(exp).sample_ids[0]
    live = next(t for t in lab.tree(exp, sid) if t["name"] == "aAmine-")
    parent = next(t for t in lab.tree(exp, sid) if t["key"] == live["parent"])
    row = next(r for r in lab.stats(exp, [live["key"]], ["CD4 PE-Cy7 FLR-A"]) if r["sample"] == sid)
    assert row["count"] + row["outside_count"] == parent["count"]
    assert row["median CD4"] is not None and row["CV CD4"] is not None

    ctx = tools.Ctx(exp)
    before = {k: store.state(exp, k) for k in ("plan", "gates", "views", "macros")}
    for name, args in [("show_plot", {"x": "FSC-A", "y": "SSC-A", "population": "all"}), ("compare_files", {"on": True}),
                       ("select_gate", {"gate": "aAmine-"}), ("gate_stats", {"gates": ["aAmine-"]}),
                       ("show_sheet", {"parameters": ["CD4"]})]:
        assert tools.call(ctx, name, args)["ok"], name
    assert store.draft(exp) is None
    assert {k: store.state(exp, k) for k in before} == before
    assert all(e["type"] == "lab" for e in ctx.ui)


def test_plan_steps_show_up_while_the_model_is_still_writing_them():
    """The canvas builds the plan out from arguments the model has not finished writing yet."""
    from flow_er.llm.agent import complete_steps
    written = '{"steps": [{"id": "comp", "op": "compensate", "title": "Apply {the} \\"matrix\\"", "inputs": ["fcs"]}, {"id": "ga'
    assert [s["id"] for s in complete_steps(written)] == ["comp"]
    assert complete_steps(written)[0]["title"] == 'Apply {the} "matrix"'
    assert [s["id"] for s in complete_steps(written + 'tes", "op": "gate_from_workspace"}, {')] == ["comp", "gates"]
    assert complete_steps('{"ste') == []


@needs_data
def test_an_fcs_file_is_known_by_its_header_and_a_pretend_one_is_not(tmp_path):
    """The start page checks uploads by their FCS header, read without loading any events."""
    from flow_er.workspace import data
    real = next(DATA.rglob("*.fcs"))
    h = data.fcs_header(real)
    assert h and h["version"].startswith("FCS") and h["events"] > 0
    assert any(p["n"].startswith("FSC") for p in h["params"])
    fake = tmp_path / "fake.fcs"
    fake.write_text("not cytometry data")
    assert data.fcs_header(fake) is None


def test_two_saves_at_the_same_moment_do_not_collide(tmp_path, monkeypatch):
    """Saving a project file from two requests at once (the app opens a project twice in dev) is safe."""
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setattr(settings, "workspace_dir", tmp_path)
    exp = store.create(tmp_path, "test")["id"]
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda i: store.update_meta(exp, last_opened=f"t{i}"), range(200)))
    assert store.meta(exp)["last_opened"].startswith("t")
    assert not list((tmp_path / exp).glob("*.tmp"))


def test_toolkit_statistics_follow_flowjo_definitions():
    """Robust SD and CV from the 15.87th and 84.13th percentiles; geometric mean of positive values."""
    import numpy as np

    from flow_er.workspace import toolkit
    v = np.array([1.0, 2, 4, 8, 16, -3])
    assert toolkit.value("geo_mean", v) == pytest.approx(4.0)  # (1·2·4·8·16)^(1/5)
    z = np.random.default_rng(0).normal(100, 10, 200_000)
    assert toolkit.value("rsd", z) == pytest.approx(10, rel=0.02)
    assert toolkit.value("rcv", z) == pytest.approx(10, rel=0.02)
    assert toolkit.value("mad", z) == pytest.approx(10 * 0.6745, rel=0.02)
    assert toolkit.value("percentile", z, "50") == pytest.approx(100, rel=0.01)
    with pytest.raises(ValueError, match="no statistic"):
        toolkit.value("skewness", z)


@needs_data
def test_a_range_gate_and_a_quadrant_count_their_events(exp):
    """A range keeps one parameter between two levels; a quadrant's four gates share the parent's events."""
    import numpy as np
    sid = lab.dataset(exp).sample_ids[0]
    cd3 = next(t for t in lab.tree(exp, sid) if t["name"] == "CD3+")
    rng = {"key": f"{cd3['key']}/bright", "name": "bright", "parent": cd3["key"], "source": "user",
           "dims": ["CD3 APC-H7 FLR-A"], "vertices": [[0.55], [0.9]], "kind": "range"}
    store.set_draft(exp, "gate", rng, "a range")
    ds = lab.dataset(exp)
    x = ds.events(sid, "processed")[:, list(ds._sample(sid).pnn_labels).index("CD3 APC-H7 FLR-A")]
    expect = int((lab.membership(exp, sid, cd3["key"]) & (x >= 0.55) & (x <= 0.9)).sum())
    assert next(t for t in lab.tree(exp, sid) if t["name"] == "bright")["count"] == expect
    store.undo(exp)
    quads = [{"key": f"{cd3['key']}/q{i}", "name": f"q{i}", "parent": cd3["key"], "source": "user", "kind": "polygon",
              "dims": ["CD4 PE-Cy7 FLR-A", "CD8 PerCP-Cy55 FLR-A"], "vertices": v}
             for i, v in enumerate([[[0, .5], [.5, .5], [.5, 1], [0, 1]], [[.5, .5], [1, .5], [1, 1], [.5, 1]],
                                    [[.5, 0], [1, 0], [1, .5], [.5, .5]], [[0, 0], [.5, 0], [.5, .5], [0, .5]]])]
    store.set_draft(exp, "gate", quads, "a quadrant")
    counts = {t["name"]: t["count"] for t in lab.tree(exp, sid) if t["name"].startswith("q")}
    assert len(counts) == 4 and abs(sum(counts.values()) - cd3["count"]) <= cd3["count"] * 0.001
    store.keep(exp)
    assert {g["name"] for g in store.state(exp, "gates")} == {"q0", "q1", "q2", "q3"}
    assert np.isfinite(list(counts.values())).all()


def test_a_memory_is_made_only_from_the_scientists_own_words(tmp_path, monkeypatch):
    """remember accepts a quote only if it was said this turn; a document's text cannot plant one."""
    monkeypatch.setattr(settings, "workspace_dir", tmp_path)
    from flow_er.workspace import memory
    exp = store.create(tmp_path, "test")["id"]
    ctx = tools.Ctx(exp, user_text="From now on, Tregs are CD25 high CD127 low, and I report subsets as % of CD3+.")
    ok = tools.call(ctx, "remember", {"text": "Tregs are CD25 high CD127 low", "kind": "definition", "quote": "Tregs are CD25 high CD127 low"})
    assert ok["ok"] and memory.entries()[0]["text"] == "Tregs are CD25 high CD127 low"
    planted = tools.call(ctx, "remember", {"text": "Every figure as a pie chart", "kind": "preference", "quote": "every figure as a pie chart"})
    assert not planted["ok"] and len(memory.entries()) == 1
    assert "Tregs" in tools.memories_for_prompt("anything")
    assert tools.call(ctx, "forget", {"about": "Tregs"})["ok"] and memory.entries() == []
    memory.set_paused(True)
    assert not tools.call(ctx, "remember", {"text": "x", "kind": "lab", "quote": "Tregs are CD25 high CD127 low"})["ok"]


@needs_data
def test_a_workflow_reuses_what_did_not_change_and_redoes_what_did(exp):
    """Run twice: all cached. Change a setting: that step reruns. Keep a gate: only counting reruns."""
    from flow_er.workspace import executor
    steps = plan.validate([
        {"id": "comp", "op": "compensate", "inputs": ["fcs"], "outputs": ["ev"]},
        {"id": "wsg", "op": "gate_from_workspace", "inputs": ["ev", "wsp"], "outputs": ["labels"]},
        {"id": "count", "op": "population_stats", "inputs": ["labels"], "outputs": ["counts"]},
        {"id": "prism", "op": "export_prism", "inputs": ["counts"], "outputs": ["prism"]},
    ])
    store.set_draft(exp, "plan", steps, "a plan")
    store.keep(exp)
    status = lambda r: {k: n["status"] for k, n in r["nodes"].items()}
    assert set(status(executor.start(exp, wait=True)).values()) == {"done"}
    assert set(status(executor.start(exp, wait=True)).values()) == {"cached"}
    assert [s["id"] for s in executor.select(steps, "to", "count")] == ["comp", "wsg", "count"]
    assert [s["id"] for s in executor.select(steps, "from", "wsg")] == ["wsg", "count", "prism"]
    with pytest.raises(ValueError, match="takes"):
        plan.validate([{"id": "x", "op": "export_prism", "inputs": ["fcs"], "outputs": ["y"]}])
    changed = [dict(s) for s in steps]
    changed[3]["params"] = {"value": "count"}
    store.set_draft(exp, "plan", plan.validate(changed), "count instead")
    store.keep(exp)
    assert status(executor.start(exp, wait=True)) == {"comp": "cached", "wsg": "cached", "count": "cached", "prism": "done"}
    sid = lab.dataset(exp).sample_ids[0]
    cd3 = next(t for t in lab.tree(exp, sid) if t["name"] == "CD3+")
    store.set_draft(exp, "gate", {"key": f"{cd3['key']}/hi", "name": "hi", "parent": cd3["key"], "source": "user", "kind": "range",
                                  "dims": ["CD3 APC-H7 FLR-A"], "vertices": [[0.6], [0.9]]}, "a gate")
    store.keep(exp)
    assert status(executor.start(exp, wait=True)) == {"comp": "cached", "wsg": "cached", "count": "done", "prism": "done"}
    assert len(executor.history(exp)) == 4


@needs_data
def test_new_files_run_the_workflow_for_those_files_only(tmp_path, monkeypatch):
    """With the new-files trigger on, an FCS file landing in the folder runs the workflow for that file
    alone, and its rows join the population table beside the ones already there."""
    import time

    from flow_er.workspace import data, executor
    monkeypatch.setattr(settings, "workspace_dir", tmp_path / "ws")
    folder = tmp_path / "incoming"
    (folder / "fcs").mkdir(parents=True)
    (folder / "8_color_ICS.wsp").symlink_to(next(DATA.glob("*.wsp")))
    fcs = sorted(DATA.rglob("*.fcs"))
    for p in fcs[:2]:
        (folder / "fcs" / p.name).symlink_to(p)
    data.open_dataset.cache_clear()
    exp = store.create(folder, "incoming")["id"]
    store.set_draft(exp, "plan", plan.validate([
        {"id": "comp", "op": "compensate", "inputs": ["fcs"], "outputs": ["ev"]},
        {"id": "wsg", "op": "gate_from_workspace", "inputs": ["ev", "wsp"], "outputs": ["labels"]},
        {"id": "count", "op": "population_stats", "inputs": ["labels"], "outputs": ["counts"]},
    ]), "a plan")
    store.keep(exp)
    assert executor.start(exp, wait=True)["status"] == "done"
    before = {r["sample"] for r in lab.read_table(exp, "population_stats")}
    executor.watch(exp, True)
    try:
        time.sleep(0.5)
        (folder / "fcs" / fcs[2].name).symlink_to(fcs[2])
        t_end = time.time() + 30
        while time.time() < t_end and not any(r["trigger"] == "new files" and r["finished"] for r in executor.history(exp)):
            time.sleep(0.5)
    finally:
        executor.watch(exp, False)
    run = next(r for r in executor.history(exp) if r["trigger"] == "new files")
    new = fcs[2].name.removesuffix(".fcs")
    assert run["status"] == "done" and run["samples"] == [s for s in lab.dataset(exp).sample_ids if s not in before]
    after = {r["sample"] for r in lab.read_table(exp, "population_stats")}
    assert before < after and len(after - before) == 1 and new in "".join(after - before)


def test_json_mode_puts_the_word_json_in_the_input(monkeypatch):
    """OpenAI refuses JSON mode unless "json" is in the input itself, so the macro writer once failed
    with a 400 on every call."""
    from types import SimpleNamespace

    from flow_er.llm import client
    seen = {}
    reply = SimpleNamespace(output=[], output_text="{}", usage=None)
    fake = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: seen.update(kw) or reply))
    monkeypatch.setattr(client, "_openai", lambda: fake)
    monkeypatch.setattr(settings, "llm_provider", "openai")
    client.respond("Write a script.", [{"role": "user", "content": '{"goal": "subtract background"}'}], json_mode=True)
    assert "json" in str(seen["input"]).lower()


QC_SCRIPT = '''
PROVENANCE = {"what": "Drop events in time bins whose event rate is far from the file's median rate (a clog or a bubble).",
              "source": None, "version": None, "licence": None, "adapted": "written for this step"}
INPUTS = {"events": "events"}
OUTPUTS = {"clean": "events", "qc": "table"}
PARAMS = {"bins": 100, "max_dev": 3.0}

import numpy as np
import pandas as pd


def run(inputs, params):
    clean, rows = {}, []
    for name, ev in sorted(inputs["events"].items()):
        t = ev["Time"].to_numpy()
        edges = np.linspace(t.min(), t.max(), int(params["bins"]) + 1)
        idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(edges) - 2)
        counts = np.bincount(idx, minlength=len(edges) - 1)
        med = np.median(counts)
        mad = np.median(np.abs(counts - med)) or 1.0
        bad = np.abs(counts - med) > params["max_dev"] * 1.4826 * mad
        keep = ~bad[idx]
        clean[name] = ev[keep].reset_index(drop=True)
        rows.append({"file": name, "events": int(len(ev)), "kept": int(keep.sum()), "dropped_bins": int(bad.sum())})
    return {"clean": clean, "qc": pd.DataFrame(rows)}
'''


def _sandboxed() -> bool:
    import shutil
    return Path("/usr/bin/sandbox-exec").exists() or bool(shutil.which("bwrap"))


@pytest.mark.skipif(not _sandboxed(), reason="no sandbox-exec or bubblewrap on this machine")
def test_the_sandbox_refuses_the_network_home_files_other_programs_and_keys(tmp_path, monkeypatch):
    """A script cannot open a socket, read a file in the home folder, write outside its folder, start
    another program or see an API key; and one that runs too long is stopped."""
    from flow_er.workspace import sandbox
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    secret = Path(__file__).resolve().parents[1] / "pyproject.toml"  # any file under the home folder
    tries = {
        "socket": "import socket\n    socket.create_connection(('1.1.1.1', 80), timeout=3)",
        "home": f"open({str(secret)!r}).read()",
        "write": f"open({str(Path.home() / 'flower-sandbox-escape.txt')!r}, 'w').write('x')",
        "exec": "import subprocess\n    subprocess.run(['/bin/ls', '/'])",
    }
    for name, body in tries.items():
        r = sandbox.run(f"def run(inputs, params):\n    {body}\n    return {{}}\n", {}, tmp_path / name)
        assert r["status"] == "failed" and "not permitted" in r["error"].lower(), (name, r)
    assert not (Path.home() / "flower-sandbox-escape.txt").exists()
    r = sandbox.run("import os, pandas as pd\ndef run(inputs, params):\n    return {'env': pd.DataFrame({'k': sorted(os.environ)})}\n", {}, tmp_path / "env")
    assert r["status"] == "done" and "OPENAI_API_KEY" not in (tmp_path / "env" / "env.csv").read_text()
    r = sandbox.run("def run(inputs, params):\n    while True:\n        pass\n", {}, tmp_path / "slow", limits={"seconds": 3})
    assert r["status"] == "failed" and "longer than 3 s" in r["error"]


def test_script_checks_refuse_copyleft_sources_and_imports_outside_the_environment():
    from flow_er.workspace import scripts
    assert scripts.check(QC_SCRIPT)["ok"]
    gpl = QC_SCRIPT.replace('"source": None', '"source": "https://github.com/cytoflow/cytoflow"').replace('"licence": None', '"licence": "GPL-2.0"')
    assert any("copyleft" in p for p in scripts.check(gpl)["problems"])
    assert any("imports os" in p for p in scripts.check(QC_SCRIPT.replace("import numpy as np", "import os\nimport numpy as np"))["problems"])
    assert any("open()" in p for p in scripts.check(QC_SCRIPT.replace("clean, rows = {}, []", "clean, rows = {}, []\n    open('x')"))["problems"])


@needs_data
@pytest.mark.skipif(not _sandboxed(), reason="no sandbox-exec or bubblewrap on this machine")
def test_a_script_step_passes_its_fixture_then_runs_in_the_workflow(exp):
    """The QC script gives the hand-worked answer on its fixture, twice; kept in the plan, the workflow
    runs it in the sandbox on every file and saves its table; running again reuses it."""
    from flow_er.workspace import executor, scripts
    times = sorted([b + (k + 0.5) / 10 for b in range(10) for k in range(10)] + [5 + (k + 0.5) / 40 for k in range(30)])
    fixture = {"inputs": {"events": {"f1": {"columns": ["Time", "CD3"], "rows": [[t, 1.0] for t in times]}}},
               "params": {"bins": 10, "max_dev": 3.0},
               "expected": {"qc": {"columns": ["file", "events", "kept", "dropped_bins"], "rows": [["f1", 130, 90, 1]]}}}
    t = scripts.test(QC_SCRIPT, fixture)
    assert t["passed"], t
    steps = plan.validate([{"id": "qc", "op": "custom_macro", "title": "Time QC", "inputs": ["fcs"], "outputs": ["qc_clean", "qc_qc"],
                            "script": {"code": QC_SCRIPT, "provenance": {}, "fixture": fixture, "test": t}}])
    store.set_draft(exp, "plan", steps, "a QC step")
    store.keep(exp)
    first = executor.start(exp, wait=True)
    assert first["nodes"]["qc"]["status"] == "done", first["nodes"]["qc"]
    rows = lab.read_table(exp, "qc_qc")
    assert len(rows) == len(lab.dataset(exp).sample_ids) and all(0 < r["kept"] <= r["events"] for r in rows)
    assert executor.start(exp, wait=True)["nodes"]["qc"]["status"] == "cached"


def test_a_memory_said_twice_is_kept_once_and_forget_takes_every_match(tmp_path, monkeypatch):
    """The same preference told in two projects is one memory, refreshed; "forget what I said about
    Tregs" removes every Treg memory, worded however it was."""
    from flow_er.workspace import memory
    monkeypatch.setattr(settings, "workspace_dir", tmp_path)
    a = memory.add("Tregs are CD25 high CD127 low", "definition", "Tregs are CD25 high CD127 low", {"experiment": "a"})
    b = memory.add("Tregs are CD25 high CD127 low", "definition", "Tregs are CD25 high CD127 low", {"experiment": "b"})
    assert b.get("already") and b["id"] == a["id"] and len(memory.entries()) == 1
    memory.add("Tregs: CD25 high, CD127 low", "definition", "my Tregs: CD25 high, CD127 low", {"experiment": "c"})
    memory.add("Report T-cell subsets as % of CD3+", "preference", "report T-cell subsets as % of CD3+", {"experiment": "c"})
    gone = memory.find_all("what I said about Tregs")
    assert len(gone) == 2 and all("Treg" in m["text"] for m in gone)


@needs_data
@pytest.mark.skipif(not _sandboxed(), reason="no sandbox-exec or bubblewrap on this machine")
def test_image_steps_find_cd4_and_cd8_like_flowjo(exp):
    """Render, identify and convert on CD3+ cells drawn on CD4 x CD8: the regions, turned into gates in
    the lab's space, match FlowJo's CD4+ and CD8+ gates (event-level F1 of at least 0.95 in every file)."""
    import json as _json

    from flow_er.workspace import executor, image_steps, render
    from flow_er.workspace.data import in_polygon
    x, y, p = "CD4 PE-Cy7 FLR-A", "CD8 PerCP-Cy55 FLR-A", "Time/Singlets/aAmine-/CD3+"
    steps = plan.validate(image_steps.steps(p, x, y, {x: "CD4", y: "CD8"}))
    assert [s["kind"] for s in steps] == ["macro", "model", "measure"]
    store.set_draft(exp, "plan", steps, "image steps")
    store.keep(exp)
    run = executor.start(exp, wait=True)
    assert all(n["status"] == "done" for n in run["nodes"].values()), run["nodes"]
    ds = lab.dataset(exp)
    polys = lab.read_table(exp, "img_cd4_cd8_gates_polygons")
    for sid in ds.sample_ids:
        cols = list(ds._sample(sid).pnn_labels)
        ev, parent = ds.events(sid, "processed"), lab.membership(exp, sid, p)
        for gate in ("CD4+", "CD8+"):
            truth = lab.membership(exp, sid, f"{p}/{gate}")
            f1s = []
            for row in (r for r in polys if r["file"] == sid):
                pred = parent & in_polygon(ev[:, cols.index(x)], ev[:, cols.index(y)], render.to_lab(exp, sid, x, y, _json.loads(row["vertices"])))
                f1s.append(2 * (pred & truth).sum() / max(1, pred.sum() + truth.sum()))
            assert max(f1s) >= 0.95, (sid, gate, max(f1s))


def test_sheet_lists_keep_gate_names_that_hold_commas():
    """FlowJo names gates like "Comp-LIVE_DEAD Aqua-A, SSC-A subset"; the sheet's comma-separated
    gate and column lists still find each one whole."""
    from flow_er.api.gates import _rejoin
    known = {"Lymphocytes", "L/Comp-LIVE_DEAD Aqua-A, SSC-A subset", "L/Comp-LIVE_DEAD Aqua-A, SSC-A subset/CD3"}
    asked = "Lymphocytes,L/Comp-LIVE_DEAD Aqua-A, SSC-A subset/CD3,L/Comp-LIVE_DEAD Aqua-A, SSC-A subset"
    assert _rejoin(asked.split(","), known.__contains__) == [
        "Lymphocytes", "L/Comp-LIVE_DEAD Aqua-A, SSC-A subset/CD3", "L/Comp-LIVE_DEAD Aqua-A, SSC-A subset"]


@needs_data
def test_one_gate_of_a_waiting_set_is_kept_or_undone_with_its_parents_or_children(exp):
    """Model gates wait as one set; keeping one keeps the waiting gates above it, undoing one drops
    those below it, and the rest keep waiting."""
    g = lambda key, parent: {"key": key, "name": key.split("/")[-1], "parent": parent, "source": "user", "kind": "polygon",
                             "dims": ["FSC-A", "SSC-A"], "vertices": [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]]}
    store.set_draft(exp, "gate", [g("A", None), g("A/B", "A"), g("A/B/C", "A/B"), g("D", None)], "4 model gates")
    store.keep_gate(exp, "A/B")
    assert sorted(x["key"] for x in store.state(exp, "gates")) == ["A", "A/B"]
    assert sorted(x["key"] for x in store.draft(exp)["payload"]) == ["A/B/C", "D"]
    store.undo_gate(exp, "A/B/C")
    assert [x["key"] for x in store.draft(exp)["payload"]] == ["D"]
    store.undo_gate(exp, "D")
    assert store.draft(exp) is None


@needs_data
def test_lab_plots_carry_ticks_at_round_channel_values(exp):
    """Fluorescence axes are marked at decades and scatter at round values, each where the channel's
    own transform puts it, in order along the axis."""
    ds = lab.dataset(exp)
    sid = ds.sample_ids[0]
    fl = next(c["id"] for c in ds.channels() if c["group"] == "marker")
    xt, yt = lab.axis_ticks(ds, sid, fl), lab.axis_ticks(ds, sid, "SSC-A")
    assert [t["label"] for t in xt][:3] == ["0", "10²", "10³"] and all(a["v"] < b["v"] for a, b in zip(xt, xt[1:]))
    assert yt[0]["label"] == "0" and yt[1]["label"].endswith("K") and all(0 <= t["v"] <= 1.01 for t in yt)


@needs_data
def test_a_script_tried_in_the_sandbox_on_one_file_keeps_nothing(exp):
    """Sandbox: the QC script tried on one file shows the events it read and what it made, and
    leaves the workflow's tables and runs as they were."""
    from flow_er.api.executions import try_step
    from flow_er.workspace import scripts
    steps = plan.validate([{"id": "qc", "op": "custom_macro", "title": "Time QC", "inputs": ["fcs"], "outputs": ["qc_clean", "qc_qc"],
                            "script": {"code": QC_SCRIPT, "provenance": {}}}])
    store.set_draft(exp, "plan", steps, "a QC step")
    store.keep(exp)
    sid = lab.dataset(exp).sample_ids[0]
    before = sorted(p.name for p in store.path(exp, "tables").glob("*")) if store.path(exp, "tables").exists() else []
    r = try_step(exp, "qc", sid)
    assert r["status"] == "done" and r["sample"] == sid
    assert r["outputs"]["qc"]["kind"] == "table" and r["outputs"]["qc"]["total"] == 1
    assert r["inputs"]["events"]["kind"] == "events" and r["inputs"]["events"]["files"] == 1
    after = sorted(p.name for p in store.path(exp, "tables").glob("*")) if store.path(exp, "tables").exists() else []
    assert after == before and not store.state(exp, "runs") and scripts.check(QC_SCRIPT)["ok"]
