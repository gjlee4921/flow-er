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
