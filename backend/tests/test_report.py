"""The results page: numbers in the words point at table cells, typed numbers are refused, and the
page is built from the files and the kept workflow."""
import os
from pathlib import Path

import pytest

from flow_er.config import settings
from flow_er.llm import tools
from flow_er.workspace import lab, report, store

DATA = Path(os.environ.get("FLOW_ER_TEST_DATA", str(settings.data_dir.expanduser() / "flowkit_8color")))
needs_data = pytest.mark.skipif(not any(DATA.rglob("*.fcs")) if DATA.exists() else True, reason=f"no FCS files at {DATA}")

ROWS = [
    {"sample": "101_E01_008_clean.fcs", "gate": "CD4+", "gate_path": "Time/CD3+/CD4+", "count": 61000, "pct_parent": 61.734},
    {"sample": "101_E03_009_clean.fcs", "gate": "CD4+", "gate_path": "Time/CD3+/CD4+", "count": 60100, "pct_parent": 61.902},
    {"sample": "101_E01_008_clean.fcs", "gate": "IFNg+", "gate_path": "Time/CD3+/CD4+/IFNg+", "count": 120, "pct_parent": 0.197},
    {"sample": "101_E01_008_clean.fcs", "gate": "IFNg+", "gate_path": "Time/CD3+/CD8+/IFNg+", "count": 300, "pct_parent": 1.204},
]


@pytest.fixture()
def exp(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "workspace_dir", tmp_path / "ws")
    e = store.create(tmp_path, "test")["id"]
    lab.save_table(e, "population_stats", ROWS)
    return e


def test_a_reference_resolves_to_its_cell(exp):
    r = report.resolve(exp, "population_stats | gate=CD4+ | sample=E01_008 | pct_parent")
    assert (r["display"], r["value"], r["n"]) == ("61.7%", 61.734, 1)
    assert report.resolve(exp, "population_stats | gate=CD8+/IFNg+ | pct_parent")["display"] == "1.2%"
    assert report.resolve(exp, "population_stats | gate=CD4+/IFNg+ | pct_parent")["display"] == "0.20%"
    assert report.resolve(exp, "population_stats | gate=CD4+ | pct_parent | range")["display"] == "61.7%–61.9%"
    assert report.resolve(exp, "population_stats | gate=CD4+ | sample=E03 | count")["display"] == "60,100"


def test_a_reference_that_is_not_one_number_says_why(exp):
    with pytest.raises(ValueError, match="matches 2 rows"):
        report.resolve(exp, "population_stats | gate=CD4+ | pct_parent")
    with pytest.raises(ValueError, match="no row"):
        report.resolve(exp, "population_stats | gate=CD19+ | pct_parent")
    with pytest.raises(ValueError, match="no column"):
        report.resolve(exp, "population_stats | gate=CD4+ | median_cd3")
    with pytest.raises(ValueError, match="no table"):
        report.resolve(exp, "cytokines | gate=CD4+ | pct_parent")


def test_a_percent_typed_after_a_reference_is_not_shown_twice(exp):
    seg = report.segments(exp, "CD4+ was {{population_stats | gate=CD4+ | pct_parent | mean}}% of CD3+.")
    assert [s.get("text") or s["number"]["display"] for s in seg] == ["CD4+ was ", "61.8%", " of CD3+."]


def test_typed_numbers_are_refused_but_marker_names_and_years_pass():
    assert report.lint("CD4+ T cells were 61.7% of CD3+, 0.25 of them, 12,345 events") == ["61.7%", "0.25", "12,345"]
    assert report.lint("CD127 low Tregs in 3 donors, acquired in 2017 {{population_stats | gate=CD4+ | pct_parent | mean}}") == []


def test_the_agent_drafts_the_words_and_nothing_lands_until_kept(exp):
    ctx = tools.Ctx(exp)
    bad = tools.call(ctx, "write_report", {"title": "CD4+ cells are 61.7% of T cells", "summary": "x"})
    assert not bad["ok"] and "types the number 61.7%" in bad["error"]
    ok = tools.call(ctx, "write_report", {
        "title": "CD4+ T cells hold steady across files",
        "summary": "CD4+ T cells were {{population_stats | gate=CD4+ | sample=E01_008 | pct_parent}} of CD3+ in E01_008.",
        "key_numbers": [{"label": "CD4+ of CD3+, range", "value": "population_stats | gate=CD4+ | pct_parent | range"}]})
    assert ok["ok"] and ok["numbers_shown"] == ["61.7%"]
    assert store.draft(exp)["kind"] == "report" and not store.state(exp, "report")
    store.keep(exp)
    kept = store.state(exp, "report")
    assert kept["title"] == "CD4+ T cells hold steady across files" and kept["key_numbers"][0]["value"].endswith("range")


@needs_data
def test_the_page_is_built_from_the_files(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "workspace_dir", tmp_path)
    e = store.create(DATA, "page")["id"]
    page = report.build(e)
    heads = [m["heading"] for m in page["methods"]]
    assert heads[:2] == ["Samples", "Panel"] and "Gating strategy" in heads and "Software" in heads
    assert all(len(f["sha256"]) == 64 for f in page["data"]["files"]) and page["data"]["files"]
    assert page["written"] is False and page["summary"] == []


def test_an_anatomy_figure_is_a_view_over_an_organ_table(exp):
    lab.save_table(exp, "organ_counts", [
        {"organ": "Lung", "condition": "naive", "mouse": "pooled", "cd8_per_mg": 1100},
        {"organ": "Lung", "condition": "tumour-bearing", "mouse": "M01", "cd8_per_mg": 4042},
        {"organ": "Spleen", "condition": "tumour-bearing", "mouse": "M01", "cd8_per_mg": 8544}])
    ctx = tools.Ctx(exp)
    r = tools.call(ctx, "write_view", {"title": "CD8 T cells by organ", "type": "anatomy", "table": "organ_counts",
                                       "x": "organ", "y": "cd8_per_mg", "color": "condition"})
    assert r["ok"] and r["rows_drawn"] == 3
    assert store.draft(exp)["payload"]["type"] == "anatomy"
    bad = tools.call(ctx, "write_view", {"title": "x", "type": "anatomy", "table": "organ_counts", "x": "tissue", "y": "cd8_per_mg"})
    assert not bad["ok"] and "tissue" in bad["error"]


def test_a_figure_reads_back_as_the_numbers_it_draws(exp):
    lab.save_table(exp, "organ_counts", [
        {"organ": "Lung", "condition": "naive", "cd8_per_mg": 1100},
        {"organ": "Lung", "condition": "tumour-bearing", "cd8_per_mg": 4000},
        {"organ": "Lung", "condition": "tumour-bearing", "cd8_per_mg": 3800},
        {"organ": "Liver", "condition": "naive", "cd8_per_mg": 900},
        {"organ": "Liver", "condition": "tumour-bearing", "cd8_per_mg": 990}])
    ctx = tools.Ctx(exp)
    tools.call(ctx, "write_view", {"title": "CD8 T cells by organ", "type": "anatomy", "table": "organ_counts",
                                   "x": "organ", "y": "cd8_per_mg", "color": "condition"})
    store.keep(exp)
    r = tools.call(ctx, "read_figure", {"figure": "Figure 1"})
    f = r["figures"][0]
    assert r["ok"] and f["type"] == "anatomy" and f["baseline_group"] == "naive"
    lung = next(m for m in f["marks"] if m["x"] == "Lung" and m["group"] == "tumour-bearing")
    assert (lung["mean"], lung["n"], lung["fold_vs_baseline"]) == (3900, 2, 3.55)
    assert tools.call(ctx, "read_figure", {"figure": "by organ"})["figures"][0]["figure"] == "Figure 1"
    assert not tools.call(ctx, "read_figure", {"figure": "Figure 4"})["ok"]


def test_show_figure_opens_it_in_the_lab_and_explains_it(exp):
    lab.save_table(exp, "organ_counts", [{"organ": "Lung", "condition": "naive", "cd8_per_mg": 1100},
                                         {"organ": "Lung", "condition": "tumour-bearing", "cd8_per_mg": 3900}])
    ctx = tools.Ctx(exp)
    tools.call(ctx, "write_view", {"title": "CD8 T cells by organ", "type": "anatomy", "table": "organ_counts",
                                   "x": "organ", "y": "cd8_per_mg", "color": "condition"})
    assert not tools.call(ctx, "show_figure", {"figure": "Figure 1"})["ok"]  # a draft is not in Saved diagrams yet
    store.keep(exp)
    ctx = tools.Ctx(exp)
    r = tools.call(ctx, "show_figure", {"figure": "the mice diagram"})
    assert r["ok"] and r["shown"] == "Figure 1" and r["figure"]["marks"]
    assert {"type": "lab", "view": "saved", "fig": store.state(exp, "views")[0]["id"]} in ctx.ui
