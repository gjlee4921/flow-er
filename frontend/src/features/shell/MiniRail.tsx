// The sidebar folded to a strip: your own flower (settings) on top, one circle per project, its flower
// as far opened as the project has come (hover for its title and status, click to open it), and + for a
// new analysis. On the start page it is all there is;
// inside a project it unfolds into the full navigator (Rail).
import { useState } from "react";
import type { Experiment } from "../../api";
import { useExperiments } from "../../api/hooks";
import { Bloom, Icon, Me } from "../../components/icons";
import { ago, ME, ui } from "../../lib";

/** How far a project has come, as a flower opening: a bud before a plan, half open with a kept plan,
 *  in full bloom once it has figures. */
const progress = (p: Experiment) => (p.figures ? 1 : p.plan_kept ? 0.6 : 0.22);

/** A steady shade of the flower pink per project (10-38%), so its circle is recognisable from one
 *  visit to the next while staying in the app's palette. */
function tint(id: string) {
  let h = 0;
  for (const c of id) h = (h * 31 + c.charCodeAt(0)) % 997;
  return `${10 + (h % 8) * 4}%`;
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
        <Me name={ME.name} />
      </button>
      {exp && (
        <button className="mr-ic" aria-label="Show the project's sidebar" title="Show the project's sidebar" onClick={() => ui.set({ railOpen: true })}><Icon.sidebar /></button>
      )}
      <span className="mr-sep" aria-hidden="true" />
      <nav className="mr-projs" aria-label="Projects">
        {live.map((p) => (
          <button key={p.id} className={`mr-proj${p.id === exp ? " on" : ""}`}
            style={{ ["--tint" as any]: tint(p.id) }} aria-label={`${p.title}: ${projectStatus(p)}`} aria-current={p.id === exp}
            onMouseEnter={(e) => show(p, e.currentTarget)} onMouseLeave={() => setTip(null)}
            onFocus={(e) => show(p, e.currentTarget)} onBlur={() => setTip(null)}
            onClick={() => { setTip(null); if (p.id !== exp) ui.openProject(p.id); }}>
            <Bloom p={progress(p)} />
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
