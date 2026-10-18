"""Readouts, statistics, robustness strip and Prism-ready tables.

Every figure is a view over a saved table, so every number on screen traces to a file.
"""
from fastapi import APIRouter, HTTPException

from flow_er.workspace import lab, store, views

router = APIRouter(tags=["results"])


@router.get("/experiments/{exp_id}/tables")
def tables(exp_id: str) -> list[dict]:
    return lab.tables(exp_id)


@router.get("/experiments/{exp_id}/tables/{name}")
def table(exp_id: str, name: str) -> dict:
    rows = lab.read_table(exp_id, name)
    if not rows:
        raise HTTPException(404, f"no table {name}")
    return {"name": name, "rows": rows}


@router.get("/experiments/{exp_id}/views")
def all_views(exp_id: str) -> list[dict]:
    """Kept views, and the draft one last, each with the rows it draws."""
    out = [{"spec": v, "rows": views.data(exp_id, v), "draft": False} for v in store.state(exp_id, "views")]
    d = store.draft(exp_id)
    if d and d["kind"] == "view":
        out.append({"spec": d["payload"], "rows": views.data(exp_id, d["payload"]), "draft": True})
    return out
