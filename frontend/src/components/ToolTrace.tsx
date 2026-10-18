// The tools a turn used, folded under one line ("Used 3 tools"), as in the main chat. Open, each says
// what it did; given a `kind`, each is marked as a read (looked something up) or a move (changed what
// is on screen), and the folded line counts them.
import { useState } from "react";
import type { TraceItem } from "../api";
import { Icon } from "./icons";

export type ToolKind = "read" | "move";

export function ToolTrace({ trace, line, kind }: {
  trace?: TraceItem[];
  line?: (t: TraceItem) => string;
  kind?: (tool: string) => ToolKind | null;
}) {
  const [open, setOpen] = useState(false);
  if (!trace?.length) return null;
  const count = (k: ToolKind) => (kind ? trace.filter((t) => kind(t.tool) === k).length : 0);
  const reads = count("read"), moves = count("move");
  const split = kind && (reads || moves) ? ` · ${[reads && `${reads} read`, moves && `${moves} moved`].filter(Boolean).join(", ")}` : "";
  return (
    <div className={`act${open ? " open" : ""}`}>
      <button type="button" aria-expanded={open} onClick={() => setOpen(!open)}>
        <Icon.eye />Used {trace.length} tool{trace.length === 1 ? "" : "s"}{split}<Icon.caret className="cv" />
      </button>
      {open && (
        <ul className="calls-list">
          {trace.map((t, i) => {
            const k = kind?.(t.tool);
            return (
              <li key={i} className={t.ok ? "" : "bad"}>
                {k && <span className={`n-pill ${k === "move" ? "blue" : "grey"}`}>{k === "move" ? "Moved" : "Read"}</span>}{" "}
                <code>{t.tool}</code> {line ? line(t) : t.summary}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
