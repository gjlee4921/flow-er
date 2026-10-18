// The sidebar inside a project, unfolded: the project, the two workbenches, the plan as numbered lanes
// (each runs its own steps; the line joining them fills in as they finish, and the last one flowers),
// what needs the scientist now, the kept figures and the latest runs. Folded, it is the MiniRail.
import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api, type ExperimentState, type Step } from "../../api";
import { refresh, useAutogateFiles, useExperiment, useExperiments } from "../../api/hooks";
import { Bloom, Flower, Icon } from "../../components/icons";
import { ME, ui, useUi, type Page, type Place } from "../../lib";

type Lane = { id: string; n: number; title: string; page: Page; kinds: string[] };
const LANES: Lane[] = [
  { id: "data", n: 1, title: "Raw data", page: "data", kinds: [] },
  { id: "clean", n: 2, title: "Clean up", page: "clean", kinds: ["macro"] },
  { id: "models", n: 3, title: "Learned models", page: "models", kinds: ["model"] },
  { id: "measure", n: 4, title: "Measure", page: "measure", kinds: ["measure"] },
  { id: "results", n: 5, title: "Results", page: "results", kinds: ["view", "report"] },
];

function laneInfo(l: Lane, s: ExperimentState) {
  const steps = s.plan.filter((p) => l.kinds.includes(p.kind));
  const status = Object.fromEntries(s.runs.map((r) => [r.step, r.status]));
  const done = steps.filter((p) => status[p.id] === "done").length;
  if (l.id === "data") {
    return { steps, done: true, sub: `${s.snapshot.samples.length} files${s.snapshot.flowjo_workspace ? " · FlowJo gates" : ""}` };
  }
  if (l.id === "results") {
    const v = s.views.length;
    const ran = steps.length > 0 && done === steps.length;
    const sub = v ? `${v} figure${v === 1 ? "" : "s"}` : ran ? "no figure kept yet" : steps.length ? "after the run" : "ask for a figure";
    return { steps, done: v > 0 || ran, sub };
  }
  if (!steps.length) return { steps, done: false, sub: "nothing planned" };
  const blocked = steps.some((p) => status[p.id] === "blocked");
  return {
    steps, done: done === steps.length,
    sub: `${steps.length} step${steps.length === 1 ? "" : "s"} · ${done === steps.length ? "done" : blocked ? "waiting on models" : done ? `${done} run` : "not run"}`,
  };
}

/** The project label at the top of the sidebar, and the list of projects it drops down. ⌘K (Ctrl+K)
 *  opens it from anywhere; arrow keys and Enter work inside; the open project can be renamed or
 *  archived from it (archived projects leave the list; nothing is deleted). */
function ProjectMenu({ exp, title, sub }: { exp: string | null; title: string; sub: string }) {
  const qc = useQueryClient();
  const { data: projects = [] } = useExperiments();
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const [act, setAct] = useState(0);
  const [mode, setMode] = useState<"list" | "rename" | "archive">("list");
  const [name, setName] = useState(title);
  const box = useRef<HTMLDivElement>(null);
  // ⌘K from inside the data lab steps out of it to show the menu; remember the lab view, so picking
  // another project saves it as it was, and closing the menu goes back into it
  const fromLab = useRef<Place | null>(null);
  const close = () => {
    setOpen(false); setQ(""); setMode("list");
    if (fromLab.current) { ui.set(fromLab.current); fromLab.current = null; }
  };

  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        const s = ui.get();
        if (s.page === "lab") {
          fromLab.current = { page: s.page, panelOpen: s.panelOpen, full: s.full, lab: s.lab, prevPage: s.prevPage };
          ui.leaveLab();
        }
        setOpen((o) => !o); setMode("list"); setQ("");
      }
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, []);
  useEffect(() => {
    if (!open) return;
    // the list as it is now, including projects made in another tab since the page loaded
    qc.invalidateQueries({ queryKey: ["experiments"] });
    const away = (e: PointerEvent) => { if (!box.current?.contains(e.target as Node)) close(); };
    document.addEventListener("pointerdown", away);
    return () => document.removeEventListener("pointerdown", away);
  }, [open]);

  const shown = projects.filter((p) => p.title.toLowerCase().includes(q.trim().toLowerCase()));
  useEffect(() => { setAct(Math.max(0, shown.findIndex((p) => p.id === exp))); }, [open, q]); // eslint-disable-line react-hooks/exhaustive-deps
  const go = (id: string | null) => {
    if (id === exp) { close(); return; }
    const leaving = fromLab.current ?? undefined;
    fromLab.current = null;
    close();
    ui.openProject(id, leaving);
  };
  const folder = (path: string) => path.split("/").filter(Boolean).pop()?.replace(/[_-]+/g, " ") ?? "";
  const status = (p: (typeof projects)[number]) =>
    p.figures ? `${p.figures} figure${p.figures === 1 ? "" : "s"}` : p.plan_kept ? "plan kept" : "not planned yet";

  const rename = async () => {
    if (!exp || !name.trim()) return;
    try { await api.renameProject(exp, name.trim()); } catch (e) { ui.toast((e as Error).message); return; }
    await refresh(qc, exp); await qc.invalidateQueries({ queryKey: ["experiments"] });
    close();
  };
  const archive = async () => {
    if (!exp) return;
    try { await api.archiveProject(exp, true); } catch (e) { ui.toast((e as Error).message); return; }
    await qc.invalidateQueries({ queryKey: ["experiments"] });
    ui.toast(`Archived “${title}”. Nothing was deleted.`);
    close(); ui.openProject(null);
  };
  const keys = (e: React.KeyboardEvent) => {
    if (!open) return;
    if (e.key === "Escape") { e.preventDefault(); mode === "list" ? close() : setMode("list"); return; }
    if (mode !== "list") return;
    if (e.key === "ArrowDown") { e.preventDefault(); setAct((a) => Math.min(shown.length - 1, a + 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setAct((a) => Math.max(0, a - 1)); }
    else if (e.key === "Enter" && shown[act]) { e.preventDefault(); go(shown[act].id); }
  };

  return (
    <div className="proj" ref={box} onKeyDown={keys}>
      <button className="rail-name" id="proj-btn" aria-haspopup="menu" aria-expanded={open} title="Switch project (⌘K)"
        onClick={() => { setOpen(!open); setMode("list"); setQ(""); }}>
        <b><span id="ws-title">{title}</span><Icon.caret /></b>
        <span id="ws-sub">{sub}</span>
      </button>
      {open && (
        <div className="proj-menu" role="menu" aria-label="Projects">
          {mode === "rename" ? (
            <form className="pm-form" onSubmit={(e) => { e.preventDefault(); rename(); }}>
              <label htmlFor="pm-name">Rename this project</label>
              <input id="pm-name" autoFocus value={name} onChange={(e) => setName(e.target.value)} />
              <div className="pm-btns"><button type="button" className="btn sm plain" onClick={() => setMode("list")}>Cancel</button><button className="btn sm ink" disabled={!name.trim()}>Rename</button></div>
            </form>
          ) : mode === "archive" ? (
            <div className="pm-form">
              <p>Archive “{title}”? It leaves this list and the start page; its files, plan and figures are kept.</p>
              <div className="pm-btns"><button className="btn sm plain" onClick={() => setMode("list")}>Cancel</button><button className="btn sm ink" id="pm-archive-yes" onClick={archive}>Archive</button></div>
            </div>
          ) : (
            <>
              <input className="pm-q" autoFocus placeholder="Find a project" value={q} aria-label="Find a project" onChange={(e) => setQ(e.target.value)} />
              <div className="pm-list">
                {shown.map((p, i) => (
                  <button key={p.id} data-id={p.id} className={`pm-row${i === act ? " act" : ""}`} role="menuitemradio" aria-checked={p.id === exp}
                    onMouseEnter={() => setAct(i)} onClick={() => go(p.id)}>
                    <span className={`pm-dot ${p.figures ? "fig" : p.plan_kept ? "plan" : ""}`} aria-hidden="true" />
                    <span className="pm-t"><b>{p.title}</b><small>{folder(p.data_path)} · {status(p)}</small></span>
                    {p.id === exp && <Icon.check />}
                  </button>
                ))}
                {!shown.length && <p className="pm-none">{projects.length ? "No project matches." : "No projects yet."}</p>}
              </div>
              <div className="pm-acts">
                {exp && <button className="pm-row pm-act" role="menuitem" onClick={() => { setName(title); setMode("rename"); }}><Icon.tag /><span>Rename this project</span></button>}
                {exp && <button className="pm-row pm-act" role="menuitem" id="pm-archive" onClick={() => setMode("archive")}><Icon.folder /><span>Archive this project</span></button>}
                <button className="pm-row pm-act" role="menuitem" id="pm-memory" onClick={() => { close(); ui.set({ memoryOpen: true }); }}><Bloom p={1} className="sm" /><span>What flow-er remembers</span></button>
                <button className="pm-row pm-act" role="menuitem" onClick={() => go(null)}><Icon.plus /><span>New analysis</span></button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

/** What is waiting on the scientist in this project: a draft to keep or drop, steps that failed or are
 *  blocked, and files the lab's models have not gated yet. Each opens where it is dealt with. */
function NeedsYou({ exp, s }: { exp: string; s: ExperimentState }) {
  const { data: gated } = useAutogateFiles(exp, !!s.plan.some((p) => p.op === "autogate"));
  const title = Object.fromEntries(s.plan.map((p) => [p.id, p.title]));
  const latest = new Map(s.runs.map((r) => [r.step, r]));
  const stuck = [...latest.values()].filter((r) => r.status === "failed" || r.status === "blocked");
  const ungated = gated?.lab ? gated.files.filter((f) => !f.gated).length : 0;
  const DRAFT = { plan: "A plan", gate: "A gate", view: "A figure", macro: "A clean-up step", report: "A write-up" } as const;
  const items: { k: string; text: string; sub?: string; go: () => void }[] = [];
  if (s.draft) items.push({ k: "draft", text: `${DRAFT[s.draft.kind] ?? "A draft"} waiting to be kept`, sub: s.draft.summary,
    go: () => ui.open(s.draft!.kind === "gate" ? "lab" : s.draft!.kind === "view" || s.draft!.kind === "report" ? "results" : "flow") });
  for (const r of stuck) items.push({ k: r.step, text: `${title[r.step] ?? r.step}: ${r.status}`, sub: r.summary, go: () => ui.open("flow") });
  if (ungated) items.push({ k: "ungated", text: `${ungated} of ${gated!.files.length} files not gated by the models yet`, go: () => ui.open("lab") });
  return (
    <>
      <div className="rail-h">Needs you</div>
      <div className="rail-sec">
        {items.map((it) => (
          <button key={it.k} className="rail-item need" title={it.sub} onClick={it.go}>
            <span className="need-dot" aria-hidden="true" /><span className="nm">{it.text}</span>
          </button>
        ))}
        {!items.length && <span className="rail-none">Nothing waiting</span>}
      </div>
    </>
  );
}

export default function Rail({ exp }: { exp: string | null }) {
  const qc = useQueryClient();
  const { data: s } = useExperiment(exp);
  const page = useUi((u) => u.page);
  const [running, setRunning] = useState<string | null>(null);

  const runLane = async (l: Lane, steps: Step[]) => {
    if (!exp || running) return;
    setRunning(l.id);
    try {
      for (const st of steps) {
        const r = await api.runStep(exp, st.id);
        if (r.status !== "done") ui.toast(r.summary);
      }
    } catch (e) {
      ui.toast((e as Error).message);
    } finally {
      setRunning(null);
      await refresh(qc, exp);
    }
  };

  const lanes = s ? LANES.map((l) => ({ l, ...laneInfo(l, s) })) : [];
  let grown = 0;
  for (let i = 1; i < lanes.length; i++) { if (lanes[i].done) grown = i; else break; }

  return (
    <aside className="ws-rail" id="ws-rail" aria-label="Navigator">
      <div className="rail-top">
        <ProjectMenu exp={exp} title={s?.title ?? "flow-er"} sub={s ? `${s.snapshot.samples.length} files · ${s.snapshot.channels.length} channels` : "Nothing open yet"} />
        <span className="rail-acts">
          <button className="ib sm" aria-label="Start another analysis" title="Start another analysis" onClick={() => { ui.openProject(null); ui.set({ dock: "open" }); }}><Icon.plus /></button>
          <button className="ib sm" aria-label="Fold the sidebar" title="Fold the sidebar" onClick={() => ui.set({ railOpen: false })}><Icon.sidebar /></button>
        </span>
      </div>
      {s && (
        <div className="plan-nav" id="plan-nav">
          <button className="pnav" aria-current={page === "flow"} onClick={() => ui.open("flow")}>
            <span className="pic"><Icon.pipeline /></span>
            <span className="lbl">Pipeline<small>{s.plan.length ? `${s.plan.length} steps` : "not planned yet"}{s.draft?.kind === "plan" ? " · draft waiting" : ""}</small></span>
          </button>
          <button className="pnav" aria-current={page === "lab"} onClick={() => ui.open("lab")}>
            <span className="pic"><Icon.lab /></span>
            <span className="lbl">Data lab<small>events, gates, saved figures</small></span>
          </button>
          <div className="plan-sep">Plan</div>
          <div className="psteps" style={{ ["--grown" as any]: grown / (LANES.length - 1) }}>
            {lanes.map(({ l, steps, done, sub }) => {
              const bloom = l.id === "results" && done;
              const canRun = steps.length > 0 && !done && l.id !== "data" && l.id !== "results";
              return (
                <div key={l.id} className={`prow${running === l.id ? " running" : ""}`}>
                  <button className="pnav step" aria-current={page === l.page} onClick={() => ui.open(l.page)}>
                    <span className={`num${done ? " done" : ""}${bloom ? " bloom" : ""}`} data-n={l.n}>{bloom && <Bloom p={1} />}</span>
                    <span className="lbl">{l.title}<small>{sub}</small></span>
                  </button>
                  {canRun && (
                    <button className="prun" title={`Run ${l.title.toLowerCase()}`} aria-label={`Run ${l.title.toLowerCase()}`} disabled={!!running} onClick={() => runLane(l, steps)}>
                      {running === l.id ? <Flower className="sm" /> : <Icon.play />}
                    </button>
                  )}
                </div>
              );
            })}
          </div>
          <NeedsYou exp={exp!} s={s} />
          {!!s.views.length && (
            <>
              <div className="rail-h">Figures</div>
              <div className="rail-sec">
                {s.views.slice(-6).reverse().map((v) => (
                  <button key={v.id} className="rail-item" title={v.caption || v.title}
                    onClick={() => { ui.lab({ view: "saved", fig: v.id }); ui.open("lab"); }}>
                    <Icon.gallery /><span className="nm">{v.title}</span>
                  </button>
                ))}
                {s.views.length > 6 && <button className="rail-item more" onClick={() => ui.open("results")}><span className="nm">All {s.views.length} in Results</span></button>}
              </div>
            </>
          )}
          {!!s.runs.length && (
            <>
              <div className="rail-h">Recent runs</div>
              <div className="rail-sec">
                {s.runs.slice(-4).reverse().map((r) => (
                  <div key={r.step} className="run-row" title={r.summary}>
                    <span className={`rdot ${r.status}`} /><span className="nm">{s.plan.find((p) => p.id === r.step)?.title ?? r.step}</span><span className="val">{r.at.slice(11, 16)}</span>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}
      <button className="rail-foot" title="Settings and what flow-er remembers" onClick={() => ui.set({ settingsOpen: true })}>
        <span className="rf-gear"><Icon.gear /></span><span className="nm"><b>Settings</b><small>{ME.name} · what flow-er remembers</small></span>
      </button>
    </aside>
  );
}
