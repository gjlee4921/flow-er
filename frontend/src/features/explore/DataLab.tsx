// The data lab: look at the events yourself. Files and gates on the left, the plot in the middle,
// the selected gate's numbers beside it and the same gate in every file below. Gates you draw are
// drafts until kept; gates from FlowJo or the models are labelled as such. Saved diagrams lists the
// figures kept so far, each with the table behind it.
import { useEffect, useMemo, useRef, useState } from "react";
import { useQueries, useQueryClient } from "@tanstack/react-query";
import { api, type ExperimentState, type GateNode } from "../../api";
import { refresh, useGates, useModels, usePlot, useSamples, useToolkit, useViews } from "../../api/hooks";
import { Bloom, Flower, Icon, Mark } from "../../components/icons";
import { LabHist, LabPlot, ViewChart } from "../../components/plots";
import { Resizer, widthOf } from "../../components/Resizer";
import { fmt, POP, shortSample, ui, useUi, type LabState } from "../../lib";
import { sendInChat } from "../agent/Chat";
import LabAgent from "./LabAgent";
import LabSheet, { sheetCsv } from "./LabSheet";

const SRC = { flowjo: "FlowJo", user: "you", model: "model" } as const;
const pillClass = (s: string) => (s === "user" ? "orange" : s === "model" ? "blue" : "grey");

export default function DataLab({ s }: { s: ExperimentState }) {
  const qc = useQueryClient();
  const lab = useUi((u) => u.lab);
  const { data: meta } = useSamples(s.id);
  const { data: models } = useModels();
  const samples = meta?.samples ?? [];
  const channels = meta?.channels ?? [];
  const sample = lab.sample ?? samples[0]?.id;
  const { data: treeData } = useGates(s.id, sample);
  // depth first, so every gate sits directly under its own parent
  const tree = useMemo(() => {
    const all = treeData?.gates ?? [];
    const kids = (k: string | null) => all.filter((g) => (g.parent ?? null) === k);
    const out: GateNode[] = [];
    const walk = (k: string | null) => kids(k).forEach((g) => { out.push(g); walk(g.key); });
    walk(null);
    return out;
  }, [treeData]);
  const byKey = useMemo(() => Object.fromEntries(tree.map((g) => [g.key, g])), [tree]);
  const children = tree.filter((g) => (g.parent ?? null) === (lab.parent ?? null));
  const short = (id?: string) => channels.find((c) => c.id === id)?.short ?? id ?? "";

  // sensible axes: the channels the first gate under this population is drawn on
  useEffect(() => {
    if (!tree.length || (lab.x && lab.y)) return;
    const first = children.find((g) => g.dims.length === 2) ?? tree.find((g) => g.dims.length === 2);
    if (first) ui.lab({ x: first.dims[0], y: first.dims[1] });
  }, [tree.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const { data: plot, isFetching } = usePlot(s.id, { sample, x: lab.x, y: lab.y, parent: lab.parent, stage: lab.stage });
  const sel = lab.sel ? byKey[lab.sel] : undefined;

  // the selected gate in every file
  const perFile = useQueries({
    queries: samples.map((x) => ({ queryKey: ["exp", s.id, "gates", x.id], queryFn: () => api.gates(s.id, x.id), enabled: !!lab.sel })),
  });
  const across = samples.map((x, i) => ({ id: x.id, g: perFile[i]?.data?.gates.find((g) => g.key === lab.sel) }));

  const select = (g: GateNode) => {
    ui.lab({ sel: g.key, parent: g.parent ?? null, ...(g.dims.length === 2 ? { x: g.dims[0], y: g.dims[1] } : {}) });
  };
  const enter = (key: string) => {
    const kids = tree.filter((g) => g.parent === key && g.dims.length === 2);
    ui.lab({ parent: key, sel: key, ...(kids[0] ? { x: kids[0].dims[0], y: kids[0].dims[1] } : {}) });
  };
  const draw = async (vertices: number[][], kind: "rect" | "polygon") => {
    if (!lab.x || !lab.y) return;
    const n = tree.filter((g) => g.source === "user").length + 1;
    try {
      const r = await api.drawGate(s.id, { name: `Gate ${n}`, parent: lab.parent, dims: [lab.x, lab.y], vertices, kind });
      ui.lab({ sel: r.gate.key, tool: "select" });
      ui.toast(`Gate ${n}: ${r.gate.pct_parent.toFixed(1)}% of the parent. Keep it to make it part of the analysis.`);
    } catch (e) { ui.toast((e as Error).message); }
    await refresh(qc, s.id);
  };
  const decide = async (keep: boolean) => {
    try { keep ? await api.keep(s.id) : await api.undo(s.id); } catch (e) { ui.toast((e as Error).message); }
    if (!keep) ui.lab({ sel: null });
    await refresh(qc, s.id);
  };

  // saved diagrams: the figures kept so far (drafts stay on the Results page until kept)
  const { data: views = [] } = useViews(s.id);
  const saved = views.filter((v) => !v.draft);
  // which one is open lives in the lab's state, so the report and the lab assistant can open one too
  const figId = lab.fig ?? null;
  const setFigId = (id: string | null) => ui.lab({ fig: id });
  const fig = saved.find((v) => v.spec.id === figId);
  const askForFigure = () => { ui.leaveLab(); sendInChat("Draw a figure that answers the experiment's question"); };
  const lead = (v: (typeof saved)[number]) => (v.spec.caption ? `${v.spec.caption.split(". ")[0].replace(/\.$/, "")}.` : `From ${v.spec.table}.`);

  const name = (id?: string) => channels.find((c) => c.id === id)?.name ?? id ?? "";
  const has = (id: string) => channels.some((c) => c.id === id);
  const PRESETS = [
    { label: "Scatter", title: "Forward against side scatter: size and granularity", x: "FSC-A", y: "SSC-A" },
    { label: "Singlets", title: "FSC-A against FSC-H: single cells sit on the diagonal", x: "FSC-A", y: "FSC-H" },
  ].filter((q) => has(q.x) && has(q.y));
  const GROUPS = [["scatter", "Scatter"], ["marker", "Markers"], ["time", "Time"]] as const;

  const parentName = lab.parent ? byKey[lab.parent]?.name ?? lab.parent : "All events";
  const tool = lab.tool;
  const { data: kit } = useToolkit();

  // the plot types, the three used most recently first (kept in this browser)
  const [recent, setRecent] = useState<string[]>(() => { try { return JSON.parse(localStorage.getItem("flower.recentPlots") || "[]"); } catch { return []; } });
  const [typeMenu, setTypeMenu] = useState(false);
  const typeBox = useRef<HTMLDivElement>(null);
  const plotTypes = useMemo(() => {
    const all = kit?.plot_types ?? [];
    const top = recent.map((id) => all.find((t) => t.id === id)).filter(Boolean).slice(0, 3) as typeof all;
    return { top, rest: all.filter((t) => !top.includes(t)) };
  }, [kit, recent]);
  const pickType = (id: string) => {
    const next = [id, ...recent.filter((r) => r !== id)].slice(0, 6);
    setRecent(next);
    try { localStorage.setItem("flower.recentPlots", JSON.stringify(next)); } catch { /* private window */ }
    ui.lab({ mode: id as typeof lab.mode, tool: id === "hist" ? (tool === "range" ? "range" : "select") : tool === "range" ? "select" : tool });
    setTypeMenu(false);
  };
  useEffect(() => {
    if (!typeMenu) return;
    const away = (e: PointerEvent) => { if (!typeBox.current?.contains(e.target as Node)) setTypeMenu(false); };
    document.addEventListener("pointerdown", away);
    return () => document.removeEventListener("pointerdown", away);
  }, [typeMenu]);

  // gate tools have no buttons: the assistant picks one, or a key does (V select, R, P, Q, G)
  useEffect(() => {
    const keys: Record<string, LabState["tool"]> = { v: "select", r: "rect", p: "poly", q: "quadrant", g: "range" };
    const k = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey || (e.target as Element)?.closest?.("input,textarea,select,[contenteditable]")) return;
      const t = keys[e.key.toLowerCase()];
      if (!t) return;
      ui.lab({ tool: t, view: "plot", ...(t === "range" ? { mode: "hist" as const } : ui.get().lab.mode === "hist" && t !== "select" ? { mode: "density" as const } : {}) });
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, []);
  const drawing = tool !== "select" ? kit?.gate_tools.find((g) => g.id === tool) : undefined;

  const drawRange = async (vertices: number[][]) => {
    if (!lab.x) return;
    const n = tree.filter((g) => g.source === "user").length + 1;
    try {
      const r = await api.drawGate(s.id, { name: `Range ${n}`, parent: lab.parent, dims: [lab.x], vertices, kind: "range" });
      ui.lab({ sel: r.gate.key, tool: "select" });
      ui.toast(`Range ${n}: ${r.gate.pct_parent.toFixed(1)}% of the parent. Keep it to make it part of the analysis.`);
    } catch (e) { ui.toast((e as Error).message); }
    await refresh(qc, s.id);
  };
  const quadrant = async (centre: number[]) => {
    if (!lab.x || !lab.y) return;
    try {
      const r = await api.drawQuadrant(s.id, { parent: lab.parent, dims: [lab.x, lab.y], centre });
      ui.lab({ tool: "select", sel: r.gates[1]?.key ?? null });
      ui.toast(`Quadrant: ${r.gates.map((g) => `${g.name} ${g.pct_parent.toFixed(1)}%`).join(", ")}. Keep them to make them part of the analysis.`);
    } catch (e) { ui.toast((e as Error).message); }
    await refresh(qc, s.id);
  };

  // Export: a PNG of the view on screen, or the sheet as a CSV file
  const exportView = async () => {
    const save = (url: string, name: string) => { const a = document.createElement("a"); a.href = url; a.download = name; a.click(); };
    const stem = [sample ? shortSample(sample) : "flow-er", short(lab.x), lab.mode === "hist" ? "" : short(lab.y)].filter(Boolean).join("-").replace(/[^\w+-]+/g, "_");
    if (lab.view === "table") {
      const csv = sheetCsv.get?.();
      if (csv) save(URL.createObjectURL(new Blob([csv], { type: "text/csv" })), "flow-er-sheet.csv");
      return;
    }
    const svgToCanvas = (el: SVGSVGElement, w: number, h: number) => new Promise<HTMLCanvasElement>((done) => {
      const clone = el.cloneNode(true) as SVGSVGElement;
      const css = getComputedStyle(document.documentElement);
      clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
      clone.querySelectorAll(".grid").forEach((n) => n.setAttribute("style", `stroke:${css.getPropertyValue("--sep")}`));
      clone.querySelectorAll(".tick, .axis-t").forEach((n) => n.setAttribute("style", `fill:${css.getPropertyValue("--text-2")};font:11px sans-serif`));
      clone.querySelectorAll(".h-area").forEach((n) => n.setAttribute("style", "fill:rgba(0,113,227,.16)"));
      clone.querySelectorAll(".h-line").forEach((n) => n.setAttribute("style", "fill:none;stroke:#0071e3;stroke-width:1.6"));
      const img = new Image();
      img.onload = () => { const c = document.createElement("canvas"); c.width = w * 2; c.height = h * 2; const x = c.getContext("2d")!; x.fillStyle = "#fff"; x.fillRect(0, 0, c.width, c.height); x.drawImage(img, 0, 0, c.width, c.height); done(c); };
      img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(new XMLSerializer().serializeToString(clone))}`;
    });
    const parts: HTMLCanvasElement[] = [];
    const scope = document.querySelector(lab.compare ? ".lab-grid" : ".lab-v-plot");
    for (const el of Array.from(scope?.querySelectorAll<HTMLElement>(lab.mode === "hist" ? "svg.lab-hist" : "canvas") ?? [])) {
      parts.push(el instanceof HTMLCanvasElement ? el : await svgToCanvas(el as unknown as SVGSVGElement, 600, (el as unknown as SVGSVGElement).viewBox.baseVal.height));
    }
    if (!parts.length) return;
    const out = document.createElement("canvas");
    out.width = parts.reduce((a, c) => a + c.width, 0); out.height = Math.max(...parts.map((c) => c.height));
    const ctx = out.getContext("2d")!; ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, out.width, out.height);
    parts.reduce((x, c) => { ctx.drawImage(c, x, 0); return x + c.width; }, 0);
    save(out.toDataURL("image/png"), `${stem}.png`);
    ui.toast("Saved a PNG of the view.");
  };
  // ⌘J (Ctrl+J) opens and closes the lab assistant
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "j") { e.preventDefault(); ui.lab({ ask: !ui.get().lab.ask }); } };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, []);
  const screen = {
    file: sample ? shortSample(sample) : null, population: parentName, x: name(lab.x), y: name(lab.y),
    plot: { density: "density", dots: "dots", hist: "histogram", contour: "contour" }[lab.mode], stage: lab.stage, gate_tool: tool,
    tab: { plot: "plot", table: "sheet", saved: "saved diagrams" }[lab.view], every_file_side_by_side: lab.compare,
    selected_gate: sel?.name ?? null,
  };
  return (
    <div className="ws-page lab-page" id="pg-lab">
      <header className="lab-top">
        <div className="lt-l">
          <button className="n-ic" id="lab-exit" title="Back to the chat (Esc)" aria-label="Back to the chat" onClick={() => ui.leaveLab()}><Icon.back /></button>
          <Mark small />
          <div className="lt-name"><b>Data lab</b><span id="lab-crumb">{sample ? shortSample(sample) : ""} · {parentName}</span></div>
        </div>
        <div className="lt-views" role="tablist">
          {([["plot", "Plot", Icon.axis], ["table", "Sheet", Icon.table], ["saved", "Saved diagrams", Icon.gallery]] as const).map(([v, t, I]) => (
            <button key={v} className="n-tab" role="tab" aria-selected={lab.view === v} onClick={() => { ui.lab({ view: v }); if (v === "saved") setFigId(null); }}>
              <I />{t}
            </button>
          ))}
        </div>
        <div className="lt-r">
          <button className={`btn sm lab-ask-btn${lab.ask ? " on" : " plain"}`} id="lab-ask" aria-pressed={lab.ask} title="The lab assistant: ask for any view of the events (⌘J)"
            onClick={() => ui.lab({ ask: !lab.ask })}><Bloom p={1} className="sm" /><span>Ask</span></button>
          <button className="btn sm ink" disabled={!models?.available} title={models?.message}>Run the model</button>
        </div>
      </header>

      <div className="lab-tools" role="toolbar" aria-label="Tools">
        <div className="tb-g" role="group" aria-label="Plot type"><div className="tb-btns">
          <div className="tb-type" ref={typeBox}>
            <button className="tb" id="plot-type" aria-haspopup="menu" aria-expanded={typeMenu} title="Plot type" onClick={() => setTypeMenu(!typeMenu)}>
              {lab.mode === "hist" ? <Icon.hist /> : lab.mode === "dots" ? <Icon.dots /> : <Icon.density />}
              <span>{kit?.plot_types.find((t) => t.id === lab.mode)?.name ?? "Density"}</span><Icon.caret />
            </button>
            {typeMenu && (
              <div className="tb-pop" role="menu" aria-label="Plot types">
                {plotTypes.top.length > 0 && <div className="tp-h">Recent</div>}
                {[...plotTypes.top, ...(plotTypes.top.length ? [null] : []), ...plotTypes.rest].map((t, i) => t ? (
                  <button key={t.id} role="menuitemradio" aria-checked={lab.mode === t.id} className="tp-row" onClick={() => pickType(t.id)}>
                    <b>{t.name}</b><span>{t.about}</span>
                  </button>
                ) : <div key={`sep${i}`} className="tp-h">All plot types</div>)}
              </div>
            )}
          </div>
        </div></div>
        <div className="tb-g" role="group" aria-label="Axes"><div className="tb-btns">
          {(["y", "x"] as const).map((ax, i) => (
            <span key={ax} style={{ display: "contents" }}>
              {i === 1 && <button className="tb only" title="Swap the axes" aria-label="Swap the axes" disabled={lab.mode === "hist"} onClick={() => ui.lab({ x: lab.y, y: lab.x })}><Icon.swap /></button>}
              <label className={`tb-sel${ax === "y" && lab.mode === "hist" ? " off" : ""}`}><span>{ax.toUpperCase()}</span>
                <select aria-label={`${ax.toUpperCase()} axis`} value={lab[ax] ?? ""} disabled={ax === "y" && lab.mode === "hist"} onChange={(e) => ui.lab({ [ax]: e.target.value })}>
                  {GROUPS.map(([g, t]) => (
                    <optgroup key={g} label={t}>
                      {channels.filter((c) => c.group === g).map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
                    </optgroup>
                  ))}
                </select>
              </label>
            </span>
          ))}
          {PRESETS.map((q) => (
            <button key={q.label} className="tb preset" title={q.title} aria-pressed={lab.x === q.x && lab.y === q.y}
              onClick={() => ui.lab({ x: q.x, y: q.y, mode: lab.mode === "hist" ? "density" : lab.mode })}><span>{q.label}</span></button>
          ))}
        </div></div>
        <div className="tb-g" role="group" aria-label="Files"><div className="tb-btns">
          <button className="tb" title="One file" aria-pressed={!lab.compare} onClick={() => ui.lab({ compare: false })}><Icon.file /><span>One file</span></button>
          <button className="tb" title="Every file side by side" aria-pressed={lab.compare} onClick={() => ui.lab({ compare: true, view: "plot" })}><Icon.gallery /><span>All files</span></button>
        </div></div>
        <div className="tb-g" role="group" aria-label="Export"><div className="tb-btns">
          <button className="tb" id="lab-export" title={lab.view === "table" ? "Download the sheet as CSV" : "Save a PNG of the view"} onClick={exportView}>
            <Icon.upload /><span>Export</span>
          </button>
        </div></div>
      </div>

      <div className={`lab-shell${lab.ask ? " asking" : ""}`}>
        <nav className="lab-nav" aria-label="Files and gates"><div className="ln-scroll">
          <div className="ln-sec">
            <div className="n-h">Files</div>
            <div className="n-tree">
              {samples.map((x, i) => (
                <button key={x.id} className="n-row file" aria-selected={x.id === sample} title={x.file} onClick={() => ui.lab({ sample: x.id })}>
                  <span className="tw" /><span className="nm"><Icon.file />{shortSample(x.id)}</span>
                  <span className="val">{lab.sel && across[i]?.g ? `${across[i].g!.pct_parent.toFixed(1)}%` : fmt(x.events)}</span>
                </button>
              ))}
            </div>
          </div>
          <div className="ln-sec">
            <div className="n-h">Gates</div>
            <div className="n-tree">
              <button className="n-row" aria-selected={!lab.parent && !lab.sel} onClick={() => ui.lab({ parent: null, sel: null })}>
                <span className="tw" /><span className="nm">All events</span><span className="val">{plot && !lab.parent ? fmt(plot.n_parent) : ""}</span>
              </button>
              {tree.map((g) => {
                const depth = g.key.split("/").length;
                // a model gate shows its name and a "model" tag rather than "(model)" in the name, so deep trees stay readable
                const m = g.name.match(/^(.*) \((model(?: \d+)?)\)$/);
                return (
                  <button key={g.key} className="n-row" aria-selected={lab.sel === g.key} style={{ paddingLeft: 6 + Math.min(depth, 8) * 8 }}
                    onClick={() => select(g)} onDoubleClick={() => enter(g.key)} title={`${g.name}: click to show it on its own channels; double-click to look inside it`}>
                    <span className="tw" /><span className="nm">{m ? m[1] : g.name}</span>
                    {m ? <span className={`n-pill ${g.draft ? "orange" : "gray"}`}>{m[2]}</span> : g.draft && <span className="n-pill orange">new</span>}
                    <span className="val">{g.pct_parent.toFixed(1)}%</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div></nav>
        <Resizer className="rsz-labnav" cssVar="--labnavw" side="left" label="Files and gates width" measure={widthOf(".lab-nav")} min={200} max={() => 460} />

        <main className="lab-stage">
          {lab.view === "plot" && lab.compare ? (
            <section className="lab-view lab-v-doc lab-grid-v"><div>
              <div className="n-h">Every file<span>{lab.mode === "hist" ? name(lab.x) : `${name(lab.y)} against ${name(lab.x)}`} · {parentName}</span></div>
              <div className="lab-grid">
                {samples.map((x) => <FileCell key={x.id} exp={s.id} sample={x.id} on={x.id === sample} xLabel={name(lab.x)} yLabel={name(lab.y)} />)}
              </div>
            </div></section>
          ) : lab.view === "plot" ? (
            <section className="lab-view lab-v-plot">
              {lab.mode === "hist" ? (
                <div className="lab-plot lab-plot-h"><LabHist data={plot} xLabel={name(lab.x)} gates={children} x={lab.x} tool={tool} selected={lab.sel}
                  onSelect={(k) => ui.lab({ sel: k })} onDraw={drawRange} /></div>
              ) : (
                <LabPlot data={plot} gates={children} x={lab.x ?? ""} y={lab.y ?? ""} xLabel={name(lab.x)} yLabel={name(lab.y)}
                  mode={lab.mode} stage={lab.stage} tool={tool} selected={lab.sel} busy={isFetching}
                  onSelect={(k) => ui.lab({ sel: k })} onEnter={enter} onDraw={draw} onQuadrant={quadrant} />
              )}
              {drawing && (
                <div className="draw-hint" role="status"><b>{drawing.name}</b> {drawing.about} <span>Esc or V to stop</span></div>
              )}
              <Resizer className="rsz-labside" cssVar="--labsidew" side="right" label="Details width" measure={widthOf(".lab-under")} min={260} max={() => 720} />
              <div className="lab-under" id="lab-under">
                <div className="ls-sec ls-details">
                  <div className="n-h">Details<span>{sample ? shortSample(sample) : ""}</span></div>
                  {sel ? (
                    <>
                      <div className="ln-big"><b>{sel.pct_parent.toFixed(2)}%</b><span>{sel.name} of {sel.parent ? byKey[sel.parent]?.name : "all events"}</span></div>
                      <div className="n-props">
                        <div className="n-prop"><span className="k"><Icon.hash />Events</span><span className="v">{fmt(sel.count)}</span></div>
                        <div className="n-prop"><span className="k"><Icon.hash />% of all</span><span className="v">{sel.pct_total.toFixed(3)}</span></div>
                        <div className="n-prop"><span className="k"><Icon.axis />Drawn on</span><span className="v">{sel.dims.map(short).join(" × ") || "—"}</span></div>
                        <div className="n-prop"><span className="k"><Icon.tag />From</span><span className="v"><span className={`n-pill ${pillClass(sel.source)}`}>{SRC[sel.source]}</span></span></div>
                      </div>
                      {sel.draft && <div className="row-btns ln-keep"><button className="btn sm ink" onClick={() => decide(true)}>Keep this gate</button><button className="btn sm plain" onClick={() => decide(false)}>Undo</button></div>}
                    </>
                  ) : (
                    <>
                      <div className="ln-big"><b>{plot ? fmt(plot.n_parent) : "…"}</b><span>events in {parentName}</span></div>
                      <p className="n-note">Click a gate in the tree or on the plot for its numbers. Double-click to look inside it.</p>
                    </>
                  )}
                </div>
              </div>
              <div className="lab-extra" id="lab-extra">
                {lab.sel && (
                  <div className="ls-block"><div className="n-h">{sel?.name ?? ""} in every file</div>
                    {/* one row per file, so any number of files fits the panel; a file without this gate shows a dash, not 0% */}
                    <div className="across">{(() => {
                      const mx = Math.max(1e-9, ...across.map((q) => q.g?.pct_parent ?? 0));
                      return across.map((a) => {
                        const v = a.g?.pct_parent;
                        return (
                          <button key={a.id} className={`ab${a.id === sample ? " on" : ""}`} title={`${a.id}: ${v == null ? "no such gate in this file" : `${v.toFixed(2)}%`}`} onClick={() => ui.lab({ sample: a.id })}>
                            <span className="ab-n">{shortSample(a.id)}</span>
                            <span className="ab-t"><i style={{ width: v == null ? 0 : `${Math.max(1.5, (v / mx) * 100)}%` }} /></span>
                            <b>{v == null ? "—" : `${v.toFixed(1)}%`}</b>
                          </button>
                        );
                      });
                    })()}</div></div>
                )}
                <div className="ls-key">
                  {lab.stage === "labels" ? (
                    <div className="legend lab-leg">{children.map((g, i) => <span key={g.key}><i style={{ background: POP[i % POP.length] }} />{g.name}</span>)}<span><i style={{ background: "rgba(120,122,132,.6)" }} />the rest</span></div>
                  ) : lab.mode === "density" ? (
                    <div className="dens-key"><span>fewer events</span><i /><span>more</span></div>
                  ) : null}
                  <p className="hint" id="lab-stats">
                    {tool === "rect" ? "Drag on the plot to draw a rectangle gate." : tool === "poly" ? "Click to add points; click the first point or double-click to close. Esc cancels."
                      : <><b>{plot ? fmt(plot.n_parent) : "…"}</b> events in {parentName}{lab.stage === "raw" ? ", uncompensated" : ""}. {plot && plot.n_shown < plot.n_parent ? `${fmt(plot.n_shown)} drawn.` : ""}</>}
                  </p>
                </div>
              </div>
            </section>
          ) : lab.view === "saved" ? (
            <section className="lab-view lab-v-doc lab-saved"><div>
              {fig ? (
                <>
                  <button className="btn sm plain saved-back" onClick={() => setFigId(null)}><Icon.back />All saved diagrams</button>
                  <figure className="rfig">
                    <div className="rfig-h"><span><b>Figure {saved.indexOf(fig) + 1}</b> · {fig.spec.title}</span></div>
                    <ViewChart view={fig} />
                    <figcaption>{fig.spec.caption || `From ${fig.spec.table}.`}</figcaption>
                  </figure>
                  {fig.spec.type !== "table" && (
                    <>
                      <div className="n-h">The table behind it<span>{fig.spec.table} · {fig.rows.length} rows</span></div>
                      <ViewChart view={{ ...fig, spec: { ...fig.spec, type: "table", columns: [] } }} />
                    </>
                  )}
                </>
              ) : (
                <>
                  <div className="n-h">Saved diagrams<span>{saved.length ? `${saved.length} kept · each drawn from a saved table` : "none kept yet"}</span></div>
                  <div className="n-gal">
                    {saved.map((v, i) => (
                      // a div, not a button: the small chart inside has buttons of its own (its legend)
                      <div key={v.spec.id} className="n-card" role="button" tabIndex={0} onClick={() => setFigId(v.spec.id)}
                        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setFigId(v.spec.id); } }}>
                        <span className="cover" aria-hidden="true"><ViewChart view={v} /></span>
                        <b>{v.spec.title}</b>
                        <span>Figure {i + 1} · {lead(v)}</span>
                      </div>
                    ))}
                    <button className="n-card add" onClick={askForFigure}>
                      <span className="cover plus">+</span><b>{saved.length ? "Ask for another" : "Ask for a figure"}</b>
                      <span>Describe the view you want and the agent writes it from a saved table.</span>
                    </button>
                  </div>
                </>
              )}
            </div></section>
          ) : (
            <LabSheet s={s} channels={channels} tree={tree} />
          )}
        </main>
        {lab.ask && <>
          <Resizer className="rsz-labask" cssVar="--labaskw" side="right" label="Lab assistant width" measure={widthOf(".lab-ask")} min={280} max={() => 560} />
          <LabAgent exp={s.id} screen={screen} />
        </>}
      </div>
      {isFetching && !plot && <div className="lab-loading"><Flower /> Reading the events…</div>}
    </div>
  );
}

/** One file in the side-by-side grid: the same axes and population as the lab, its own events. */
function FileCell({ exp, sample, on, xLabel, yLabel }: { exp: string; sample: string; on: boolean; xLabel: string; yLabel: string }) {
  const lab = useUi((u) => u.lab);
  const { data: plot } = usePlot(exp, { sample, x: lab.x, y: lab.y, parent: lab.parent, stage: lab.stage });
  const { data: g } = useGates(exp, sample);
  const kids = (g?.gates ?? []).filter((q) => (q.parent ?? null) === (lab.parent ?? null));
  const picked = lab.sel ? g?.gates.find((q) => q.key === lab.sel) : undefined;
  const open = () => ui.lab({ sample, compare: false });
  return (
    <figure className={`lab-cell${on ? " on" : ""}`}>
      <figcaption>
        <button className="lc-name" onClick={open} title="Open this file on its own">{shortSample(sample)}</button>
        <span>{plot ? `${fmt(plot.n_parent)} events` : "…"}{picked ? ` · ${picked.name} ${picked.pct_parent.toFixed(1)}%` : ""}</span>
      </figcaption>
      {lab.mode === "hist" ? <LabHist data={plot} xLabel={xLabel} height={300} /> : (
        <LabPlot data={plot} gates={kids} x={lab.x ?? ""} y={lab.y ?? ""} xLabel={xLabel} yLabel={yLabel} mode={lab.mode} stage={lab.stage}
          tool="select" selected={lab.sel} onSelect={(k) => ui.lab({ sel: k })} onEnter={() => open()} onDraw={() => {}} />
      )}
    </figure>
  );
}
