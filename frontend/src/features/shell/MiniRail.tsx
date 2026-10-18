// The sidebar folded to a strip: the profile (settings) on top, one circle per project (hover for its
// title and status, click to open it) and + for a new analysis. On the start page it is all there is;
// inside a project it unfolds into the full navigator (Rail).
import { useState } from "react";
import type { Experiment } from "../../api";
import { useExperiments } from "../../api/hooks";
import { Icon, Mark } from "../../components/icons";
import { ago, ui } from "../../lib";

/** Two letters: the first word's initial, then the last word's number or initial ("als ics screen4" -> A4). */
function initials(title: string) {
  const w = title.split(/[\s_-]+/).filter(Boolean);
  if (!w.length) return "?";
  const last = w[w.length - 1];
  const second = w.length > 1 ? (last.match(/\d+$/)?.[0].slice(-1) ?? last[0]) : (w[0][1] ?? "");
  return (w[0][0] + second).toUpperCase();
}

/** A steady colour per project, so its circle is recognisable from one visit to the next. */
function hue(id: string) {
  let h = 0;
  for (const c of id) h = (h * 31 + c.charCodeAt(0)) % 360;
  return h;
}

export const projectStatus = (p: Experiment) =>
  p.figures ? `${p.figures} figure${p.figures === 1 ? "" : "s"}` : p.plan_kept ? "plan kept" : "not planned yet";
const folderOf = (path: string) => path.split(/[\\/]/).filter(Boolean).pop()?.replace(/[_-]+/g, " ") ?? "";

export default function MiniRail({ exp }: { exp: string | null }) {
  const { data: projects = [] } = useExperiments();
  const live = projects.filter((p) => !p.archived);
  const [tip, setTip] = useState<{ p: Experiment; top: number; left: number } | null>(null);
  const show = (p: Experiment, el: HTMLElement) => {
    const r = el.getBoundingClientRect();
    setTip({ p, top: r.top + r.height / 2, left: r.right + 10 });
  };

  return (
    <aside className="mini-rail" id="ws-rail" aria-label="Projects">
      <button className="mr-me" aria-label="Settings" title="Settings and what flow-er remembers" onClick={() => ui.set({ settingsOpen: true })}>
        <Mark /><span className="me-badge" aria-hidden="true">W</span>
      </button>
      {exp && (
        <button className="mr-ic" aria-label="Show the project's sidebar" title="Show the project's sidebar" onClick={() => ui.set({ railOpen: true })}><Icon.sidebar /></button>
      )}
      <span className="mr-sep" aria-hidden="true" />
      <nav className="mr-projs" aria-label="Projects">
        {live.map((p) => (
          <button key={p.id} className={`mr-proj${p.figures ? " fig" : p.plan_kept ? " plan" : ""}${p.id === exp ? " on" : ""}`}
            style={{ ["--h" as any]: hue(p.id) }} aria-label={`${p.title}: ${projectStatus(p)}`} aria-current={p.id === exp}
            onMouseEnter={(e) => show(p, e.currentTarget)} onMouseLeave={() => setTip(null)}
            onFocus={(e) => show(p, e.currentTarget)} onBlur={() => setTip(null)}
            onClick={() => { setTip(null); if (p.id !== exp) ui.openProject(p.id); }}>
            {initials(p.title)}
          </button>
        ))}
        <button className="mr-add" aria-label="New analysis" title="New analysis"
          onClick={() => { setTip(null); ui.openProject(null); ui.set({ dock: "open" }); }}><Icon.plus /></button>
      </nav>
      {tip && (
        <div className="mr-tip" role="tooltip" style={{ top: tip.top, left: tip.left }}>
          <b>{tip.p.title}</b>
          <span className="mr-st"><i className={`pm-dot ${tip.p.figures ? "fig" : tip.p.plan_kept ? "plan" : ""}`} />{projectStatus(tip.p)}</span>
          <span>{folderOf(tip.p.data_path)} · opened {ago(tip.p.last_opened ?? tip.p.created)}</span>
        </div>
      )}
    </aside>
  );
}
