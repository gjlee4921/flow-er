// The sidebar folded to a strip: a circle for each of the three most recent projects, its flower as far
// opened as the project has come (hover for its title and status, click to open it), +N for the rest (a
// list to search), + for a new analysis (it goes home with the composer open), and settings at the
// bottom. On the start page it is all there is; inside a project it unfolds into the full navigator (Rail).
import { useEffect, useRef, useState } from "react";
import type { Experiment } from "../../api";
import { useExperiments } from "../../api/hooks";
import { Bloom, Icon } from "../../components/icons";
import { ago, ui } from "../../lib";

/** Circles shown before the rest fold into +N. */
const RECENT = 3;

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

/** The projects behind +N: a list to search, opened beside the strip; ↑ ↓ and Enter pick one, Esc or
 *  a click elsewhere closes it. */
function MoreProjects({ rest, at, onClose }: { rest: Experiment[]; at: { top: number; left: number }; onClose: () => void }) {
  const box = useRef<HTMLDivElement>(null);
  const [q, setQ] = useState("");
  const [act, setAct] = useState(0);
  const shown = rest.filter((p) => p.title.toLowerCase().includes(q.trim().toLowerCase()));
  useEffect(() => {
    const away = (e: PointerEvent) => { if (!box.current?.contains(e.target as Node) && !(e.target as Element).closest?.(".mr-more")) onClose(); };
    document.addEventListener("pointerdown", away);
    return () => document.removeEventListener("pointerdown", away);
  }, [onClose]);
  useEffect(() => setAct(0), [q]);
  const go = (p: Experiment) => { onClose(); ui.openProject(p.id); };
  const keys = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") { e.preventDefault(); onClose(); }
    else if (e.key === "ArrowDown") { e.preventDefault(); setAct((a) => Math.min(shown.length - 1, a + 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setAct((a) => Math.max(0, a - 1)); }
    else if (e.key === "Enter" && shown[act]) { e.preventDefault(); go(shown[act]); }
  };
  // kept on screen: it opens level with +N and moves up when that would run off the bottom
  const top = Math.max(12, Math.min(at.top, window.innerHeight - 440));
  return (
    <div className="proj-menu mr-pop" ref={box} role="menu" aria-label="More projects" style={{ top, left: at.left }} onKeyDown={keys}>
      <input className="pm-q" autoFocus placeholder={`Find one of ${rest.length} more projects`} value={q} aria-label="Find a project" onChange={(e) => setQ(e.target.value)} />
      <div className="pm-list">
        {shown.map((p, i) => (
          <button key={p.id} className={`pm-row${i === act ? " act" : ""}`} role="menuitem" onMouseEnter={() => setAct(i)} onClick={() => go(p)}>
            <span className={`pm-dot ${p.figures ? "fig" : p.plan_kept ? "plan" : ""}`} aria-hidden="true" />
            <span className="pm-t"><b>{p.title}</b><small>{folderOf(p.data_path)} · {projectStatus(p)}</small></span>
            <small className="mr-ago">{ago(p.last_opened ?? p.created)}</small>
          </button>
        ))}
        {!shown.length && <p className="pm-none">No project matches.</p>}
      </div>
    </div>
  );
}

export default function MiniRail({ exp }: { exp: string | null }) {
  const { data: projects = [] } = useExperiments();
  // newest first (the list comes sorted by when each was last opened); the open project always shows
  const live = projects.filter((p) => !p.archived);
  const recent = live.slice(0, RECENT);
  const current = live.find((p) => p.id === exp);
  if (current && !recent.includes(current)) recent.splice(RECENT - 1, 1, current);
  const rest = live.filter((p) => !recent.includes(p));
  const [tip, setTip] = useState<{ p: Experiment; top: number; left: number } | null>(null);
  const [more, setMore] = useState<{ top: number; left: number } | null>(null);
  const show = (p: Experiment, el: HTMLElement) => {
    const r = el.getBoundingClientRect();
    setTip({ p, top: r.top + r.height / 2, left: r.right + 10 });
  };

  return (
    <aside className="mini-rail" id="ws-rail" aria-label="Projects">
      {exp && <>
        <button className="mr-ic" aria-label="Show the project's sidebar" title="Show the project's sidebar" onClick={() => ui.set({ railOpen: true })}><Icon.sidebar /></button>
        <span className="mr-sep" aria-hidden="true" />
      </>}
      <nav className="mr-projs" aria-label="Projects">
        {recent.map((p) => (
          <button key={p.id} className={`mr-proj${p.id === exp ? " on" : ""}`}
            style={{ ["--tint" as any]: tint(p.id) }} aria-label={`${p.title}: ${projectStatus(p)}`} aria-current={p.id === exp}
            onMouseEnter={(e) => show(p, e.currentTarget)} onMouseLeave={() => setTip(null)}
            onFocus={(e) => show(p, e.currentTarget)} onBlur={() => setTip(null)}
            onClick={() => { setTip(null); if (p.id !== exp) ui.openProject(p.id); }}>
            <Bloom p={progress(p)} />
          </button>
        ))}
        {!!rest.length && (
          <button className="mr-more" aria-label={`${rest.length} more projects`} title={`${rest.length} more projects`} aria-haspopup="menu" aria-expanded={!!more}
            onClick={(e) => { setTip(null); const r = e.currentTarget.getBoundingClientRect(); setMore(more ? null : { top: r.top - 6, left: r.right + 10 }); }}>
            +{rest.length > 99 ? "99" : rest.length}
          </button>
        )}
        <button className="mr-add" aria-label="New analysis" title="New analysis"
          onClick={() => { setTip(null); ui.openProject(null); ui.set({ dock: "open" }); }}><Icon.plus /></button>
      </nav>
      <button className="mr-gear" aria-label="Settings" title="Settings and what flow-er remembers" onClick={() => ui.set({ settingsOpen: true })}>
        <Icon.gear />
      </button>
      {more && <MoreProjects rest={rest} at={more} onClose={() => setMore(null)} />}
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
