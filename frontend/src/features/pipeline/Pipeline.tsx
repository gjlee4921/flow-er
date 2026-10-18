// The plan as a workflow you can run, drawn as a graph (React Flow): sources on the left, then clean
// up, models, measurements and figures, each step a node with what it reads coming in from the left.
// Run the workflow, all of it or from a node's panel; each step lights up as it runs and says what it
// made, and steps that have not changed are reused. A draft plan shows dashed until it is kept, and
// builds itself out as the agent writes it. Past runs are listed; the new-files switch runs it on its own.
import { useEffect, useMemo, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  Background, Handle, Panel, Position, ReactFlow, ReactFlowProvider, useReactFlow, useUpdateNodeInternals, ViewportPortal,
  type Edge as RFEdge, type Node as RFNode, type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { api, type AutogateGate, type Execution, type ExperimentState, type LabJob, type SandboxOut, type SandboxTry, type Script, type Step, type ViewSpec } from "../../api";
import {
  followRun, refresh, useAutogateFiles, useAutogateResult, useExecutions, useGates, useLabModels, useNodeTypes, useSamples, useStepImages, useStepOutput,
} from "../../api/hooks";
import { DensityImage } from "../../components/plots/DensityImage";
import { densCol } from "../../components/plots/LabPlot";
import { Flower, Icon } from "../../components/icons";
import { clamp, css, fmt, gateLabel, shortSample, ui, useUi } from "../../lib";
import { askInChat, sendInChat } from "../agent/Chat";

const COLS = ["source", "macro", "model", "measure", "view"] as const;
const COL_NAME: Record<string, string> = { source: "Raw data", macro: "Clean up", model: "Learned models", measure: "Measure", view: "Results" };
// muted, of similar lightness, so no stage shouts louder than another; the icon says which is which
const KIND_COL: Record<string, string> = { source: "#8a8f98", macro: "#6b7d96", model: "#8b7c9c", measure: "#a08a68", view: "#6b8f87" };
const KIND_ICON = { source: Icon.file, macro: Icon.sliders, model: Icon.sparkle, measure: Icon.hash, view: Icon.axis } as const;
const NW = 210, COLW = 258, ROWH = 92;
// stacked top to bottom when the canvas is narrow (the side panel): one stage per row
const ROWD = 112, GAPD = 26, NARROW = 880;
const SAID: Record<string, string> = { done: "done", cached: "reused", failed: "failed", blocked: "waiting", running: "running", queued: "queued" };

type Card = {
  id: string; kind: string; title: string; sub: string; step?: Step; status: string; outputs: string[]; inputs: string[];
  run?: { status: string; summary?: string }; build?: number; selected?: boolean; down?: boolean;
  [key: string]: unknown;
};

/** One step on the canvas: its stage tile, its name, what it did last, and a mark for how it went. */
function StepCard({ id, data: d }: NodeProps<RFNode<Card>>) {
  const I = KIND_ICON[d.kind as keyof typeof KIND_ICON] ?? Icon.file;
  // the ports move between the sides and the top and bottom when the layout turns; React Flow keeps
  // the old positions unless told, and the lines would start from where the ports used to be
  const remeasure = useUpdateNodeInternals();
  useEffect(() => remeasure(id), [d.down, id, remeasure]);
  const st = d.run?.status;
  // the stage goes in as k-<stage>: React Flow takes any element classed "source" or "target" for a port
  return (
    <div title={d.title} className={`node ${d.status} k-${d.kind}${st ? ` ran-${st}` : ""}${st === "running" ? " running" : ""}${d.build != null ? " build" : ""}${d.selected ? " sel" : ""}`}
      style={{ width: (d.w as number | undefined) ?? NW, ["--k" as string]: KIND_COL[d.kind] ?? KIND_COL.source, ["--i" as string]: d.build ?? 0 }}>
      {d.inputs.length > 0 && <Handle type="target" position={d.down ? Position.Top : Position.Left} className="port in" isConnectable={false} />}
      {d.outputs.length > 0 && <Handle type="source" position={d.down ? Position.Bottom : Position.Right} className="port out" isConnectable={false} />}
      <span className="nic"><I /></span>
      <span style={{ minWidth: 0 }}><b>{d.title}</b><small>{d.run?.summary || d.sub}</small></span>
      {d.status === "draft" ? <span className="badge">draft</span>
        : st === "running" ? <span className="badge run" title="Running"><i className="spin" /></span>
        : st === "done" || st === "cached" || (!st && d.status === "done") ? <span className={`badge ok${st === "cached" ? " reused" : ""}`} title={st === "cached" ? "Reused: nothing it reads has changed" : "Done"}><Icon.check /></span>
        : st === "failed" ? <span className="badge warn" title={d.run?.summary}>failed</span>
        : st === "blocked" || d.status === "blocked" ? <span className="badge warn">waiting</span> : null}
    </div>
  );
}
const NODE_TYPES = { step: StepCard };

function Canvas({ s }: { s: ExperimentState }) {
  const qc = useQueryClient();
  const rf = useReactFlow();
  const draftPlan = s.draft?.kind === "plan" ? (s.draft.payload as Step[]) : null;
  // while the agent is writing a plan, its steps so far; then the draft it lands as
  const preview = useUi((u) => u.planPreview);
  const building = !draftPlan && !!preview?.length;
  const steps = draftPlan ?? (building ? preview! : s.plan);
  const live = useUi((u) => (u.run?.exp === s.id ? u.run : null));
  const busy = useUi((u) => u.agentBusy);
  const { data: history = [] } = useExecutions(s.id);
  const [sel, setSel] = useState<string | null>(null);
  const [showRuns, setShowRuns] = useState(false);
  const [viewRun, setViewRun] = useState<Execution | null>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const outer = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 800, h: 600 });
  const [outerW, setOuterW] = useState(800);
  useEffect(() => {
    const el = outer.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setOuterW(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  // the layout follows the whole panel's width, so opening a step's details never flips it
  const down = outerW < NARROW;
  const lastRun = useMemo(() => Object.fromEntries(s.runs.map((r) => [r.step, r])), [s.runs]);
  // in a draft, only the steps it adds or changes are drafts; the rest are the kept steps as they were
  const kept = useMemo(() => Object.fromEntries(s.plan.map((k) => [k.id, k])), [s.plan]);
  const same = (a: Step, b?: Step) => !!b && JSON.stringify([a.op, a.inputs, a.outputs, a.params ?? {}, a.script?.code]) === JSON.stringify([b.op, b.inputs, b.outputs, b.params ?? {}, b.script?.code]);

  const cards = useMemo<Card[]>(() => {
    const src: Card[] = [{ id: "fcs", kind: "source", title: "FCS files", sub: `${s.snapshot.samples.length} files`, status: "done", outputs: ["fcs"], inputs: [] }];
    if (s.snapshot.flowjo_workspace) src.push({ id: "wsp", kind: "source", title: "FlowJo workspace", sub: s.snapshot.flowjo_workspace, status: "done", outputs: ["wsp"], inputs: [] });
    if (s.snapshot.uploads.length) src.push({ id: "uploads", kind: "source", title: "Added in the chat", sub: `${s.snapshot.uploads.length} files`, status: "done", outputs: ["uploads"], inputs: [] });
    const stepCards: Card[] = steps.map((st) => ({
      id: st.id, kind: st.kind, title: st.title, sub: st.op === "custom_macro" ? "script" : st.op.replace(/_/g, " "), step: st,
      status: draftPlan && !same(st, kept[st.id]) ? "draft" : building ? "building" : lastRun[st.id]?.status ?? "planned", outputs: st.outputs, inputs: st.inputs,
    }));
    // the Results lane: the figures, each drawn from the table it reads (a waiting one dashed), or a
    // card saying how to get one; figures are not plan steps, so nothing else would put them here
    const draftView = s.draft?.kind === "view" ? (s.draft.payload as ViewSpec) : null;
    const figs = [...s.views.filter((v) => v.id !== draftView?.id), ...(draftView ? [draftView] : [])];
    const figCards: Card[] = figs.map((v, i) => ({
      id: `fig:${v.id}`, kind: "view", title: v.title, sub: `Figure ${i + 1} · ${v.type === "table" ? "table" : `${v.type} chart`}`,
      status: v === draftView ? "draft" : "done", outputs: [], inputs: [v.table], fig: true,
    }));
    if (!figCards.length && steps.length && !building)
      figCards.push({ id: "fig:none", kind: "view", title: "No figures yet", sub: "Ask for a figure or a write-up in the chat", status: "planned", outputs: [], inputs: [], fig: true });
    return src.concat(stepCards, figCards);
  }, [s, steps, draftPlan, building, lastRun, kept]); // eslint-disable-line react-hooks/exhaustive-deps

  const { pos, widths } = useMemo(() => {
    const lane: Record<string, number> = {};
    for (const n of cards) {
      // a stage per lane; a script sits with the latest stage it reads from (one reading the counts,
      // the measured ones or the models', goes under Measure), so its lines run forward instead of climbing back
      let c = Math.max(0, COLS.indexOf(n.kind as (typeof COLS)[number]));
      if (n.step?.op === "custom_macro") {
        for (const i of n.inputs) {
          const from = cards.find((m) => m.outputs.includes(i));
          if (from && lane[from.id] != null) c = Math.max(c, from.step?.op === "autogate" ? COLS.indexOf("measure") : lane[from.id]);
        }
      }
      lane[n.id] = c;
    }
    const out: Record<string, [number, number]> = {}, w: Record<string, number> = {};
    const byLane: Record<number, Card[]> = {};
    for (const n of cards) (byLane[lane[n.id]] ??= []).push(n);
    if (down) {
      // stacked: each card about as wide as its title needs (within bounds), each row centred on the
      // panel's middle; rows need not line up at the edges
      const fit = (n: Card) => Math.round(Math.max(160, Math.min(236, 92 + n.title.length * 4.6)));
      const rowW = (row: Card[]) => row.reduce((t, n) => t + fit(n), 0) + (row.length - 1) * GAPD;
      const widest = Math.max(NW, ...Object.values(byLane).map(rowW));
      for (const [c, row] of Object.entries(byLane)) {
        let x = 30 + (widest - rowW(row)) / 2;
        for (const n of row) { out[n.id] = [x, 40 + Number(c) * ROWD]; w[n.id] = fit(n); x += fit(n) + GAPD; }
      }
    } else {
      for (const [c, row] of Object.entries(byLane)) row.forEach((n, r) => { out[n.id] = [30 + Number(c) * COLW, 40 + r * ROWH]; w[n.id] = NW; });
    }
    return { pos: out, widths: w };
  }, [cards, down]);

  // the order a draft's new steps appear in: by column, then row; kept for as long as they are shown
  const built = useRef(new Map<string, number>());
  cards.filter((n) => (n.status === "draft" || n.status === "building") && !built.current.has(n.id))
    .sort((a, b) => pos[a.id][0] - pos[b.id][0] || pos[a.id][1] - pos[b.id][1])
    .forEach((n, k) => built.current.set(n.id, k));

  const shownRun = viewRun?.nodes ?? live?.nodes;
  const nodes: RFNode<Card>[] = cards.map((n) => ({
    id: n.id, type: "step", position: { x: pos[n.id][0], y: pos[n.id][1] }, draggable: false,
    sourcePosition: down ? Position.Bottom : Position.Right, targetPosition: down ? Position.Top : Position.Left,
    data: { ...n, run: shownRun?.[n.id], build: built.current.get(n.id), selected: sel === n.id, down, w: widths[n.id] },
  }));
  const edges: RFEdge[] = [];
  for (const n of cards) for (const inp of n.inputs) {
    const from = cards.find((m) => m.outputs.includes(inp));
    if (!from) continue;
    const st = shownRun?.[n.id]?.status;
    const k = built.current.get(n.id);
    edges.push({ id: `${from.id}-${n.id}`, source: from.id, target: n.id, animated: st === "running",
      className: `edge${st === "done" || st === "cached" || (!st && lastRun[n.id]?.status === "done") ? " done" : ""}${building || n.status === "draft" ? " planned" : ""}${k != null ? " build" : ""}`,
      style: k != null ? ({ ["--i" as string]: k } as React.CSSProperties) : undefined });
  }

  // fit when the steps or the canvas change: the whole graph if it fits at a readable size, otherwise
  // a readable size that keeps the start in view (or, while the agent builds it, the newest step)
  const layoutKey = cards.map((n) => `${n.kind}:${n.id}:${Math.round(widths[n.id] ?? 0)}`).join("|");
  useEffect(() => {
    const { w, h } = size;
    const ids = Object.keys(pos);
    const maxX = Math.max(...ids.map((id) => pos[id][0] + (widths[id] ?? NW))) + 30, maxY = Math.max(...ids.map((id) => pos[id][1])) + 72 + 20;
    const top = 58, bottom = down ? 20 : 56;
    const k = clamp(Math.min((w - 24) / maxX, (h - top - bottom) / maxY), 0.62, 1);
    const fitsX = maxX * k <= w - 24, fitsY = maxY * k <= h - top - bottom;
    // centred on the panel's middle when it fits; while the agent builds, follow the newest step
    let x = fitsX || down ? (w - maxX * k) / 2 : building ? w - 12 - maxX * k : 0;
    let y = fitsY || !building || !down ? top : h - bottom - maxY * k;
    // the step whose details are open stays in view above (or beside) them
    if (sel && pos[sel]) {
      const ny = y + pos[sel][1] * k, nx = x + pos[sel][0] * k;
      if (ny < top || ny + 72 * k > h - bottom) y = (h - 72 * k) / 2 - pos[sel][1] * k;
      if (nx < 8 || nx + (widths[sel] ?? NW) * k > w - 8) x = (w - (widths[sel] ?? NW) * k) / 2 - pos[sel][0] * k;
    }
    rf.setViewport({ x, y, zoom: k }, { duration: 450 });
  }, [layoutKey, size.w, size.h, sel, down]); // eslint-disable-line react-hooks/exhaustive-deps

  const runAll = async (mode = "all", node?: string) => {
    setViewRun(null);
    try { const r = await api.runWorkflow(s.id, mode, node); followRun(s.id, r.id, qc); }
    catch (e) { ui.toast((e as Error).message); }
  };
  const toggleActive = async () => {
    try { await api.setActive(s.id, !s.active); await refresh(qc, s.id); ui.toast(!s.active ? "The workflow now runs on its own when new FCS files arrive." : "New files no longer start the workflow."); }
    catch (e) { ui.toast((e as Error).message); }
  };
  const running = live?.status === "running";
  const doneCount = live ? Object.values(live.nodes).filter((n) => ["done", "cached", "failed", "blocked"].includes(n.status)).length : 0;

  if (!steps.length) {
    return (
      <div className="empty">
        <div style={{ display: "grid", gap: 10, justifyItems: "center" }}>
          <h2>No analysis planned yet</h2>
          <p>Tell me in the chat what the experiment is for. I will lay the analysis out here as steps: arithmetic as macros, judgement calls as models, then the measurements and the figures.</p>
        </div>
        <button className="btn ink big-btn" onClick={() => sendInChat("Plan the analysis")}>Plan the analysis</button>
      </div>
    );
  }

  const selCard = cards.find((n) => n.id === sel);
  return (
    <div className={`flow-split${down ? " down" : ""}${selCard?.step ? " docked" : ""}`} ref={outer}>
    <div className={`flow-wrap flow-rf${showRuns ? " side" : ""}`} ref={wrap}>
      <ReactFlow nodes={nodes} edges={edges} nodeTypes={NODE_TYPES} minZoom={0.3} maxZoom={1.6} proOptions={{ hideAttribution: true }}
        nodesConnectable={false} nodesDraggable={false} zoomOnScroll={false} panOnScroll zoomActivationKeyCode="Meta"
        onNodeClick={(_, n) => (n.data.fig ? ui.open("results") : setSel(sel === n.id ? null : n.id))} onPaneClick={() => setSel(null)}>
        <Background gap={22} size={1.2} color="var(--sep-2)" />
        <ViewportPortal>
          {COLS.map((c, i) => <div key={c} className="lane-label" style={{ position: "absolute", transform: down ? `translate(30px, ${16 + i * ROWD}px)` : `translate(${30 + i * COLW}px, 12px)` }}>{COL_NAME[c]}</div>)}
        </ViewportPortal>
        <Panel position="top-left" className="flow-tools">
          <span className="grpb">
            <button onClick={() => rf.zoomOut({ duration: 200 })} aria-label="Zoom out">−</button>
            <button onClick={() => rf.fitView({ padding: 0.12, duration: 400 })}>Fit</button>
            <button onClick={() => rf.zoomIn({ duration: 200 })} aria-label="Zoom in">+</button>
          </span>
          <button className="btn sm ink" id="flow-run" disabled={!!draftPlan || building || !s.plan.length || running}
            title={draftPlan ? "Keep the plan first" : "Run every step; steps that have not changed are reused"} onClick={() => runAll()}>
            {running ? <Flower className="sm" /> : <Icon.play />}{running ? `Running ${doneCount} of ${Object.keys(live!.nodes).length}` : "Run workflow"}
          </button>
          <button className={`btn sm${showRuns ? " on" : ""}`} id="flow-runs" onClick={() => setShowRuns(!showRuns)}>Runs{history.length ? ` · ${history.length}` : ""}</button>
          <button className={`flow-switch${s.active ? " on" : ""}`} id="flow-active" role="switch" aria-checked={!!s.active} onClick={toggleActive}
            title="Run the workflow on its own when new FCS files arrive in the data folder">
            <i className="sw" aria-hidden="true" /><span>{s.active ? "Runs on new files" : "Run on new files"}</span>
          </button>
          <button className="btn sm" onClick={() => askInChat("Add a step that ")}>+ Add a step</button>
        </Panel>
      </ReactFlow>
      {busy ? <div className="flow-banner building" role="status"><Flower className="sm" />{busy}…</div>
        : viewRun ? <div className="flow-banner" role="status">Showing the run at {viewRun.started.slice(11, 19)} ({viewRun.trigger}). <button className="link" onClick={() => setViewRun(null)}>Back to now</button></div>
        : draftPlan ? <div className="flow-banner">{s.plan.length ? `${cards.filter((c) => c.status === "draft").length || "No"} step${cards.filter((c) => c.status === "draft").length === 1 ? "" : "s"} changed or added, dashed. Keep the change to run it.` : "This plan is a draft. Keep it to run its steps."}</div>
        : live && !running ? <div className="flow-banner" role="status">{runLine(live)}</div> : null}
      {showRuns && (
        <aside className="runs-panel" aria-label="Past runs">
          <div className="insp-h"><b>Runs</b><button className="ib sm" aria-label="Close" onClick={() => setShowRuns(false)}><Icon.close /></button></div>
          <div className="runs-list">
            {history.map((r) => (
              <button key={r.id} className={`run-item${viewRun?.id === r.id ? " on" : ""}`} onClick={() => setViewRun(viewRun?.id === r.id ? null : r)}>
                <span className="ri-h"><b>{r.started.slice(11, 19)}</b><span className={`ri-st ${r.status}`}>{r.status}</span><span className="ri-tr">{r.trigger}</span></span>
                <span className="ri-dots">{Object.entries(r.nodes).map(([k, n]) => <i key={k} className={`d ${n.status}`} title={`${n.title}: ${SAID[n.status] ?? n.status}`} />)}</span>
              </button>
            ))}
            {!history.length && <p className="small muted">Not run yet.</p>}
          </div>
        </aside>
      )}
    </div>
      {selCard?.step && <NodePanel s={s} card={selCard} draft={!!draftPlan || building} onClose={() => setSel(null)} onRun={runAll} run={shownRun?.[selCard.id]} />}
    </div>
  );
}

/** A script step's code with where it came from and the test it passed, as the spec asks before it
 *  can be kept: provenance, licence, a hand-worked answer on a fixture, and the same output twice. */
function ScriptView({ script: sc }: { script: Script }) {
  const p = sc.provenance ?? {};
  const t = sc.test;
  return (
    <div className="np-script">
      {p.what && <p>{p.what}</p>}
      <dl className="np-prov">
        <dt>Source</dt>
        <dd>{p.source ? <><a href={p.source} target="_blank" rel="noreferrer noopener">{p.source.replace(/^https?:\/\/(www\.)?/, "")}</a>{p.version ? ` · ${p.version}` : ""}</> : "Written for this step"}</dd>
        <dt>Licence</dt><dd>{p.licence ? <span className="pill">{p.licence}</span> : "None needed"}</dd>
        {p.adapted && <><dt>Adapted</dt><dd>{p.adapted}</dd></>}
        <dt>Test</dt>
        <dd className={t?.passed ? "ok" : "bad"}>{t?.passed
          ? `Passed in the sandbox: the hand-worked answer on a small fixture, and the same output twice${t.seconds != null ? ` (${t.seconds} s)` : ""}.`
          : `Not passed${t?.problems?.length ? `: ${t.problems.join("; ")}` : ""}.`}</dd>
      </dl>
      <pre className="np-code"><code>{sc.code.trim()}</code></pre>
    </div>
  );
}

/** One line on how a run went: what was redone, what was reused because nothing it reads changed. */
export function runLine(r: { nodes: Record<string, { status: string }> }): string {
  const ns = Object.values(r.nodes);
  const n = (st: string) => ns.filter((x) => x.status === st).length;
  const step = (k: number) => `${k} step${k === 1 ? "" : "s"}`;
  if (n("failed")) return `${step(n("failed"))} failed. ${n("done") + n("cached")} of ${ns.length} finished.`;
  if (n("blocked")) return `${step(n("blocked"))} waiting. ${n("done") + n("cached")} of ${ns.length} finished.`;
  if (!n("done")) return ns.length === 1 ? "Nothing changed, so the step was reused." : `Nothing changed, so all ${ns.length} steps were reused.`;
  return n("cached") ? `Redid ${step(n("done"))}; reused ${n("cached")} that had not changed.` : `Ran ${step(n("done"))}.`;
}

/** The density images behind a step, for one file at a time: what a render step drew, or what a
 *  script read and found (regions outlined). A step that made polygons can open them as draft gates. */
function StepImages({ s, step }: { s: ExperimentState; step: Step }) {
  const qc = useQueryClient();
  const [file, setFile] = useState<string | undefined>();
  const { data } = useStepImages(s.id, step.id, file, step.op === "render_image" || step.op === "custom_macro");
  const { data: samples } = useSamples(s.id);
  if (!data?.images.length) return null;
  const density = data.images.find((i) => i.kind === "density");
  const regions = data.images.find((i) => i.kind === "regions");
  const name = (id?: string) => samples?.channels.find((c) => c.id === id)?.name ?? id;
  const polygons = step.outputs.some((o) => /polygons$/.test(o));
  const openGates = async () => {
    try {
      const r = await api.regionGates(s.id, step.id, data.file);
      await refresh(qc, s.id);
      ui.lab({ sample: r.sample, parent: r.parent || null, x: r.x, y: r.y, mode: "density", view: "plot", sel: r.gates[0] ?? null });
      ui.open("lab");
      ui.toast(`${r.gates.length} gates from the image are waiting: keep or undo them in the lab.`);
    } catch (e) { ui.toast((e as Error).message); }
  };
  const short = (f: string) => f.replace(/^101_DEN084Y5_15_|_clean\.fcs$|\.fcs$/g, "");
  return (
    <div className="np-images">
      {data.files.length > 1 && (
        <select className="np-file" value={data.file} onChange={(e) => setFile(e.target.value)} aria-label="File">
          {data.files.map((f) => <option key={f} value={f}>{short(f)}</option>)}
        </select>
      )}
      {density && <DensityImage grid={density.grid} regions={regions?.grid} x={name(density.meta.x)} y={name(density.meta.y)} />}
      {density && <p className="small muted">{(density.meta.parent || "All events").split("/").pop()} · {density.meta.events?.toLocaleString()} events · recipe {density.meta.recipe}</p>}
      {polygons && <button className="btn sm" id="open-region-gates" onClick={openGates}>Open as gates in the data lab</button>}
    </div>
  );
}

/** The autogate step's output: one file at a time, every gate with the model that answered, how sure
 *  it was (unsure ones flagged for review), and FlowJo's % and an F1 where FlowJo gates exist. The
 *  gates open in the data lab as drafts beside FlowJo's; kept ones teach the lab's next version. */
function AutogatePanel({ s }: { s: ExperimentState }) {
  const qc = useQueryClient();
  const { data: files } = useAutogateFiles(s.id);
  const [file, setFile] = useState<string | undefined>();
  const sample = file ?? files?.files.find((f) => f.gated)?.sample ?? files?.files[0]?.sample;
  const { data: res, isFetching } = useAutogateResult(s.id, sample);
  const { data: lab } = useLabModels(files?.lab);
  const short = (f: string) => f.replace(/^101_DEN084Y5_15_|_clean\.fcs$|\.fcs$/g, "");
  const review = async () => {
    if (!sample) return;
    try {
      const r = await api.autogateReview(s.id, sample);
      await refresh(qc, s.id);
      ui.lab({ sample: r.sample, parent: r.parent, x: r.x, y: r.y, mode: "density", view: "plot", sel: r.gate });
      ui.open("lab");
      ui.toast(`${r.gates.length} model gates are waiting beside FlowJo's: keep or undo them.`);
    } catch (e) { ui.toast((e as Error).message); }
  };
  const teach = async () => {
    if (!sample) return;
    try {
      const r = await api.autogateAccept(s.id, sample);
      qc.invalidateQueries({ queryKey: ["lab", files?.lab, "models"] });
      const n = r.taught.individuals.length;
      ui.toast(`Taught lab ${r.lab} ${r.gates} gates from ${short(sample)} (${r.individual})${r.from_scientist ? `, ${r.from_scientist} as you left them in the lab` : ""}. `
        + `It has kept gates from ${n} ${n === 1 ? "donor" : "donors"}${n < 2 ? "; teach a file from another donor before retraining" : ""}.`);
    } catch (e) { ui.toast((e as Error).message); }
  };
  const [job, setJob] = useState<LabJob | null>(null);
  const retrain = async () => {
    if (!files?.lab) return;
    try {
      let j = await api.retrainLab(files.lab);
      setJob(j);
      while (j.status === "running") {
        await new Promise((z) => setTimeout(z, 3000));
        j = await api.job(j.id);
        setJob(j);
      }
      qc.invalidateQueries({ queryKey: ["lab", files.lab, "models"] });
      ui.toast(j.status === "done" ? `Retrain finished: ${j.summary?.reason}` : `Retrain stopped: ${j.error}`);
    } catch (e) { ui.toast((e as Error).message); }
  };
  const gates = res?.gates ?? [];
  const f1s = gates.map((g) => g.f1).filter((v): v is number => typeof v === "number");
  return (
    <div className="ag-panel">
      <div className="ag-head">
        {files && files.files.length > 1 && (
          <select className="np-file" id="ag-file" value={sample} onChange={(e) => setFile(e.target.value)} aria-label="File">
            {files.files.map((f) => <option key={f.sample} value={f.sample}>{short(f.sample)}{f.gated ? "" : " (not gated yet)"}</option>)}
          </select>
        )}
        <span className="small muted">{files?.lab ? `Lab ${files.lab} · ${files.version}` : "Template copy (no lab)"}{res?.seconds ? ` · ${res.seconds} s` : ""}</span>
      </div>
      {isFetching && !res && <p className="np-status"><Flower className="sm" /> Gating {sample && short(sample)}…</p>}
      {!!gates.length && (
        <div className="n-tblwrap np-tbl"><table className="n-tbl ag-tbl">
          {/* five narrow columns, so the panel shows them all: the model's % beside FlowJo's, and a dot for "review this" */}
          <thead><tr><th>Gate</th><th>By</th><th title="How sure the model is (0 to 1); a dot means review this gate">Sure</th><th title="% of parent: the model's, then FlowJo's">% · FlowJo</th><th title="Overlap with FlowJo's gate, cell by cell (1 is identical)">F1</th></tr></thead>
          <tbody>{gates.map((g) => (
            <tr key={g.gate} className={g.low_confidence ? "ag-unsure" : ""}>
              <td title={g.gate}>{gateLabel(g.gate.split("/").pop() ?? "")}</td>
              <td title={g.model}>{g.model.includes("/unito/") ? "U-Net" : g.model === "template_copy" ? "Copy" : g.model}</td>
              <td>{g.confidence == null ? "—" : g.confidence.toFixed(2)}{g.low_confidence && <i className="ag-dot" title="Review this gate" aria-label="review" />}</td>
              <td>{g.percent_of_parent.toFixed(2)}<span className="ag-fj"> · {g.flowjo_pct == null ? "—" : g.flowjo_pct.toFixed(2)}</span></td>
              <td>{g.f1 == null ? "—" : g.f1.toFixed(2)}</td>
            </tr>
          ))}</tbody>
        </table>
        <p className="small muted">{gates.filter((g) => g.model.includes("/unito/")).length} of {gates.length} gates by a model{f1s.length ? ` · median F1 against FlowJo ${[...f1s].sort((a, b) => a - b)[Math.floor(f1s.length / 2)].toFixed(3)}` : ""}</p></div>
      )}
      <div className="ag-actions">
        <button className="btn sm ink" id="ag-review" onClick={review} disabled={!gates.length}>Review in the data lab</button>
        {files?.lab && <button className="btn sm" id="ag-teach" onClick={teach} disabled={!gates.length} title="Keep these gates as training examples for the lab's next version">Teach the models</button>}
      </div>
      {!!lab?.versions.length && (
        <div className="ag-models"><b className="small">Models for lab {lab.lab}</b>
          {lab.taught && <p className="small muted" id="ag-taught">Taught so far: {lab.taught.samples} {lab.taught.samples === 1 ? "file" : "files"} from {lab.taught.individuals.length} {lab.taught.individuals.length === 1 ? "donor" : "donors"}{lab.taught.individuals.length ? ` (${lab.taught.individuals.join(", ")})` : ""}</p>}
          {lab.taught && <button className="btn sm" id="ag-retrain" onClick={retrain} disabled={job?.status === "running" || lab.taught.individuals.length < 2}
            title="Train new models on the kept gates; a version is used only if it is better">
            {job?.status === "running" ? <><Flower className="sm" /> Retraining…</> : "Retrain on the kept gates"}</button>}
          {/* why the button is off, said plainly rather than only on hover */}
          {lab.taught && lab.taught.individuals.length < 2 && !job && (
            <p className="small muted" id="ag-why">Retraining compares on a donor it did not learn from, so it needs kept gates from 2 donors; {lab.taught.individuals.length ? "1 so far. Teach a file from another donor." : "none yet. Review a file, then Teach the models."}</p>
          )}
          {job && <p className="small muted" id="ag-job">{job.status === "running" ? job.lines.at(-1) ?? "starting" : job.status === "done" ? job.summary?.reason : job.error}</p>}
          <details className="ag-versions">
            <summary>Versions ({lab.versions.length}){lab.versions.find((v) => v.current) ? ` · ${lab.versions.find((v) => v.current)!.version} in use` : ""}</summary>
            <ul>{lab.versions.map((v) => <li key={v.version} className={v.current ? "on" : ""}><span>{v.version}</span>{v.current && <span className="pill">current</span>}<small>{v.notes}</small></li>)}</ul>
          </details>
        </div>
      )}
    </div>
  );
}

/** One gate as the model drew it on one file: the density it saw (101 × 101, rows = y) with its
 *  polygon on top, pink when it is sure and orange dashed when it asks for review. */
function GateTile({ g, onOpen }: { g: AutogateGate; onOpen: () => void }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const S = 132;
  useEffect(() => {
    const c = ref.current, grid = g.plot?.density;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = S * dpr; c.height = S * dpr;
    const ctx = c.getContext("2d")!;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = css("--plot") || "#fff"; ctx.fillRect(0, 0, S, S);
    if (grid?.length) {
      const ny = grid.length, nx = grid[0].length, cw = S / nx, ch = S / ny;
      const mx = Math.max(...grid.flat().map((v) => Math.log1p(v))) || 1;
      for (let r = 0; r < ny; r++) for (let k = 0; k < nx; k++) {
        const v = grid[r][k];
        if (v) { ctx.fillStyle = densCol(Math.log1p(v) / mx); ctx.fillRect(k * cw, S - (r + 1) * ch, cw + 0.5, ch + 0.5); }
      }
    }
    const poly = g.polygon?.display;
    if (poly?.length) {
      // display units to pixels: a pixel holds round(100 v), so v sits at the centre of pixel 100 v
      const X = (v: number) => (Math.max(-0.01, Math.min(1.01, v)) * 100 + 0.5) * (S / 101);
      const Y = (v: number) => S - (Math.max(-0.01, Math.min(1.01, v)) * 100 + 0.5) * (S / 101);
      ctx.beginPath(); poly.forEach(([a, b], i) => (i ? ctx.lineTo(X(a), Y(b)) : ctx.moveTo(X(a), Y(b)))); ctx.closePath();
      ctx.fillStyle = g.low_confidence ? "rgba(196,90,0,.08)" : "rgba(215,127,156,.12)"; ctx.fill();
      ctx.strokeStyle = g.low_confidence ? css("--c-flag") : css("--fl-core"); ctx.lineWidth = 1.6;
      ctx.setLineDash(g.low_confidence ? [4, 3] : []); ctx.stroke(); ctx.setLineDash([]);
    }
  }, [g]);
  const name = gateLabel(g.gate.split("/").pop() ?? "");
  return (
    <button className={`gt${g.low_confidence ? " unsure" : ""}`} onClick={onOpen} title={`${g.gate}\nOpen it in the data lab`}>
      <canvas ref={ref} style={{ width: S, height: S }} aria-label={`${name}: the model's gate`} />
      <span className="gt-n">{name}</span>
      <span className="gt-v">{g.percent_of_parent.toFixed(1)}%{g.confidence != null && <> · sure {g.confidence.toFixed(2)}</>}{g.flowjo_pct != null && <span className="gt-fj"> · FlowJo {g.flowjo_pct.toFixed(1)}%</span>}</span>
    </button>
  );
}

/** Live: the models at work on the data. Every gate of one file, drawn on the events it was drawn on;
 *  during a run it follows each file as the models finish it. */
function LiveGates({ s, running }: { s: ExperimentState; running: boolean }) {
  const { data: files } = useAutogateFiles(s.id, true, running ? 1500 : false);
  const [pick, setPick] = useState<string | undefined>();
  const all = files?.files ?? [];
  const gated = all.filter((f) => f.gated);
  const sample = pick ?? (running ? gated.at(-1)?.sample : undefined) ?? gated[0]?.sample ?? all[0]?.sample;
  const { data: res, isFetching } = useAutogateResult(s.id, sample);
  const gates = (res?.gates ?? []).filter((g) => g.plot?.density?.length);
  const open = (g: AutogateGate) => { ui.lab({ sample, parent: g.parent || null, x: g.x, y: g.y, view: "plot", mode: "density" }); ui.open("lab"); };
  return (
    <div className="live">
      <div className="live-h">
        <select className="np-file" id="live-file" value={sample ?? ""} onChange={(e) => setPick(e.target.value)} aria-label="File">
          {all.map((f) => <option key={f.sample} value={f.sample}>{shortSample(f.sample)}{f.gated ? "" : " (not gated yet)"}</option>)}
        </select>
        <span className="live-p" role="status">{running && <i className="live-dot" />}{running ? "Gating" : "Gated"} {gated.length} of {all.length} files
          {files?.lab ? <> · lab {files.lab} <b className="live-v">{files.version}</b></> : null}</span>
        {pick && running && <button className="link" onClick={() => setPick(undefined)}>Follow the run</button>}
      </div>
      <div className="live-bar"><i style={{ width: `${all.length ? (100 * gated.length) / all.length : 0}%` }} /></div>
      {isFetching && !res && <p className="np-status"><Flower className="sm" /> Gating {sample && shortSample(sample)}…</p>}
      {!!gates.length && <div className="gt-grid">{gates.map((g) => <GateTile key={g.gate} g={g} onOpen={() => open(g)} />)}</div>}
      <p className="small muted">Each tile is the population the model saw, on the two channels it gates, with the gate it drew: pink when sure, orange dashed when it asks for review. Click one to open it in the data lab.</p>
    </div>
  );
}

/** Sandbox: a script tried on one file's data, to see what it does before or without a run. Nothing it
 *  makes is kept. */
function ScriptSandbox({ s, step }: { s: ExperimentState; step: Step }) {
  const [sample, setSample] = useState(s.snapshot.samples[0]?.id);
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<SandboxTry | null>(null);
  const go = async () => {
    setBusy(true);
    try { setRes(await api.tryStep(s.id, step.id, sample)); } catch (e) { ui.toast((e as Error).message); }
    setBusy(false);
  };
  const show = (name: string, o: SandboxOut) => (
    <div key={name} className="sb-out">
      <b className="small">{name}</b>
      {o.kind === "table" ? (
        <div className="n-tblwrap np-tbl"><table className="n-tbl">
          <thead><tr>{o.columns.slice(0, 6).map((c) => <th key={c}>{c}</th>)}</tr></thead>
          <tbody>{o.rows.slice(0, 8).map((r, i) => <tr key={i}>{r.slice(0, 6).map((v, j) => <td key={j}>{/\.fcs$/i.test(v) ? shortSample(v) : v}</td>)}</tr>)}</tbody>
        </table><p className="small muted">{o.for_file ? `${o.for_file} of ${o.total} rows are this file's` : `${o.total} row${o.total === 1 ? "" : "s"}`}{o.columns.length > 6 ? `, ${o.columns.length} columns` : ""}</p></div>
      ) : <p className="small">{fmt(o.events)} events in {o.files} file{o.files === 1 ? "" : "s"}</p>}
    </div>
  );
  return (
    <div className="sandbox">
      <div className="live-h">
        <select className="np-file" id="sb-file" value={sample} onChange={(e) => { setSample(e.target.value); setRes(null); }} aria-label="File">
          {s.snapshot.samples.map((x) => <option key={x.id} value={x.id}>{shortSample(x.id)}</option>)}
        </select>
        <button className="btn sm ink" id="sb-run" onClick={go} disabled={busy}>{busy ? <><Flower className="sm" /> Trying…</> : <><Icon.play />Try on this file</>}</button>
      </div>
      <p className="small muted">Runs the script on this file's data in the sandbox (no network, no other files) and shows what it reads and makes. Nothing is kept: the workflow is unchanged.</p>
      {res?.status === "done" && (
        <>
          <p className="np-status"><span className="ri-st done">tried</span> in {res.seconds} s on {shortSample(res.sample)}</p>
          {res.inputs && <div className="sb-sec"><span className="sb-k">It read</span>{Object.entries(res.inputs).map(([n, o]) => show(n, o))}</div>}
          {res.outputs && <div className="sb-sec"><span className="sb-k">It made</span>{Object.entries(res.outputs).map(([n, o]) => show(n, o))}</div>}
        </>
      )}
      {res && res.status !== "done" && <p className="np-status"><span className="ri-st failed">{res.status}</span> {res.summary}</p>}
    </div>
  );
}

/** A step's panel, as in n8n: what comes in, its settings, and what it made last time. */
function NodePanel({ s, card, draft, onClose, onRun, run }: { s: ExperimentState; card: Card; draft: boolean; onClose: () => void;
  onRun: (mode: string, node: string) => void; run?: { status: string; summary?: string } }) {
  const qc = useQueryClient();
  const step = card.step!;
  const { data: types } = useNodeTypes();
  const { data: out } = useStepOutput(s.id, step.id);
  const { data: samples } = useSamples(s.id);
  const { data: gates } = useGates(s.id, s.snapshot.samples[0]?.id);
  const [tab, setTab] = useState<"input" | "settings" | "code" | "output" | "live" | "sandbox">(step.script && !out?.ran ? "code" : "output");
  // while the models gate the files, the Live view is the thing to watch
  useEffect(() => { if (run?.status === "running" && step.op === "autogate") setTab("live"); }, [run?.status, step.op]);
  const type = types?.types[step.op];
  const allSteps = s.plan;
  const from = (inp: string) => allSteps.find((x) => x.outputs.includes(inp));
  const portType = (inp: string) => types?.sources[inp] ?? (from(inp) ? types?.types[from(inp)!.op]?.makes : "?");
  const setParam = async (k: string, v: string) => {
    try { await api.setParams(s.id, step.id, { [k]: v }); await refresh(qc, s.id); ui.toast("Changed. Keep it to make it part of the workflow."); }
    catch (e) { ui.toast((e as Error).message); }
  };
  const rows = out?.rows ?? [];
  const cols = rows.length ? Object.keys(rows[0]).slice(0, 6) : [];
  return (
    <aside className="inspector node-panel node-dock" aria-label={card.title}>
      <div className="insp-h"><span className="np-t"><b>{card.title}</b><small>{type?.name ?? step.op}</small></span><button className="ib sm" aria-label="Close" onClick={onClose}><Icon.close /></button></div>
      <div className="np-tabs" role="tablist">
        {(step.script ? (["input", "code", "output", "sandbox"] as const)
          : step.op === "autogate" ? (["input", "settings", "output", "live"] as const)
          : (["input", "settings", "output"] as const)).map((t) => (
          <button key={t} role="tab" aria-selected={tab === t} className={`np-tab${t === "live" || t === "sandbox" ? " np-tab-live" : ""}`} id={`np-tab-${t}`} onClick={() => setTab(t)}>
            {{ input: "Input", settings: "Settings", code: "Code", output: "Output", live: "Live", sandbox: "Sandbox" }[t]}</button>
        ))}
      </div>
      <div className="insp-b">
        {tab === "input" && (
          <>
            {type && <p className="small muted">{type.about}</p>}
            {step.why && <p>{step.why}</p>}
            <div className="np-ports">{step.inputs.map((inp) => (
              <div key={inp} className="np-port"><span className="pt-t">{portType(inp)}</span><b>{from(inp)?.title ?? ({ fcs: "FCS files", wsp: "FlowJo workspace", uploads: "Added in the chat" } as Record<string, string>)[inp] ?? inp}</b><small>{inp}</small></div>
            ))}{!step.inputs.length && <p className="small muted">Reads nothing.</p>}</div>
            <p className="small muted">Makes {step.outputs.join(", ") || "nothing"} ({type?.makes ?? "?"}).</p>
          </>
        )}
        {tab === "settings" && (
          type?.params && Object.keys(type.params).length ? (
            <div className="np-params">{Object.entries(type.params).map(([k, p]) => (
              <label key={k} className="np-param"><span>{p.label}</span>
                <select id={`param-${k}`} value={step.params?.[k] ?? p.default ?? ""} disabled={draft} onChange={(e) => setParam(k, e.target.value)}>
                  {(p.choices === "channels" ? (samples?.channels ?? []).map((c) => [c.id, c.name] as const)
                    : p.choices === "gates" ? [["", "All events"] as const, ...(gates?.gates ?? []).map((g) => [g.key, g.key] as const)]
                    : (p.enum ?? []).map((v) => [v, p.names?.[v] ?? v] as const)).map(([v, label]) => <option key={v} value={v}>{label}</option>)}
                </select>
              </label>
            ))}<p className="small muted">A change is a draft of the plan: keep it, and the next run redoes this step and what reads from it; everything else is reused.</p></div>
          ) : <p className="small muted">This step has no settings.</p>
        )}
        {tab === "code" && step.script && <ScriptView script={step.script} />}
        {tab === "live" && <LiveGates s={s} running={run?.status === "running"} />}
        {tab === "sandbox" && <ScriptSandbox s={s} step={step} />}
        {tab === "output" && (
          <>
            <StepImages s={s} step={step} />
            {step.op === "autogate" && out?.ran && <AutogatePanel s={s} />}
            {run?.status === "running" ? <p className="np-status"><Flower className="sm" /> Running…</p>
              : out?.ran ? <p className="np-status"><span className={`ri-st ${run?.status ?? out.status}`}>{SAID[run?.status ?? out.status ?? ""] ?? out.status}</span> {run?.summary ?? out.summary}</p>
              : <p className="small muted">Not run yet.</p>}
            {out?.at && <p className="small muted">Last run {out.at.replace("T", " ")}{out.execution ? ` · run ${out.execution}` : ""}</p>}
            {!!rows.length && step.op !== "autogate" && (
              <div className="n-tblwrap np-tbl"><table className="n-tbl">
                <thead><tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
                <tbody>{rows.slice(0, 8).map((r, i) => <tr key={i}>{cols.map((c) => <td key={c}>{typeof r[c] === "number" ? fmt(r[c], Number.isInteger(r[c]) ? 0 : 2) : String(r[c]).replace(/^101_DEN084Y5_15_|_clean\.fcs$/g, "")}</td>)}</tr>)}</tbody>
              </table><p className="small muted">{out?.table} · {out?.total_rows} rows</p></div>
            )}
          </>
        )}
      </div>
      {!draft && (
        <div className="np-run">
          {/* one button: the step, and anything before it that changed (unchanged steps are reused) */}
          <button className="btn sm ink" onClick={() => onRun("to", step.id)} title="Runs this step, and any step before it that changed; unchanged steps are reused"><Icon.play />Run this step</button>
        </div>
      )}
    </aside>
  );
}

export default function Pipeline({ s }: { s: ExperimentState }) {
  return <ReactFlowProvider><Canvas s={s} /></ReactFlowProvider>;
}
