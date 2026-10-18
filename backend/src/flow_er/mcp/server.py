"""MCP server exposing the same tools the web app's agent uses.

Run with `uv run python -m flow_er.mcp.server` (stdio) so Claude Code, Claude Science or any MCP
client can drive an experiment. Every tool takes the experiment id first; changes are drafts,
kept or undone with the keep and undo tools, exactly as in the web app.
"""
from __future__ import annotations

import json

from mcp.server.mcpserver import MCPServer

from flow_er.llm import tools
from flow_er.workspace import data, store

mcp = MCPServer("flow-er")


@mcp.tool(description="List the datasets under DATA_DIR and the experiments already made.")
def experiments() -> str:
    return json.dumps({"datasets": data.find_datasets(), "experiments": store.list_all()}, default=str)


def _register(t: tools.Tool) -> None:
    def run(experiment_id: str, arguments: str = "{}") -> str:
        args = json.loads(arguments or "{}")
        ctx = tools.Ctx(experiment_id)
        result = tools.call(ctx, t.name, args)
        return json.dumps({**result, "ui": ctx.ui}, default=str)

    params = ", ".join(sorted(t.params)) or "none"
    mcp.add_tool(run, name=t.name,
                 description=f"{t.description} Arguments as a JSON object; keys: {params}.")


# the data lab assistant's tools only move the lab on screen, which an MCP client does not have
for _t in tools.REGISTRY.values():
    if _t.name not in tools.LAB_ONLY:
        _register(_t)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
