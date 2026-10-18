// The models at work. While a run gates the files with the lab's models, the data lab shows each file
// as soon as it is done, gate by gate, on the events each gate was drawn on (the density the model saw)
// with the gate it drew. Watching keeps nothing: the gates become drafts only when you review a file.
import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type AutogateGate, type ExperimentState } from "../../api";
import { refresh, useAutogateResult } from "../../api/hooks";
import { Flower, Icon, Mark } from "../../components/icons";
import { densCol } from "../../components/plots/LabPlot";
import { css, gateLabel, shortSample, ui } from "../../lib";

/** One gate, large: the parent population on the gate's two channels (101 × 101, rows = y) and the
 *  model's polygon traced in, pink when it is sure and orange dashed when it asks for review. */
function LiveCanvas({ g, dur }: { g: AutogateGate; dur: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const box = useRef<HTMLDivElement>(null);
  const [S, setS] = useState(460);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setS(Math.max(240, Math.min(560, el.clientWidth - 8, el.clientHeight - 40))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  useEffect(() => {
    const c = ref.current, grid = g.plot?.density;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = S * dpr; c.height = S * dpr;
    const ctx = c.getContext("2d")!;
    ctx.scale(dpr, dpr);
    const paint = () => {
      ctx.fillStyle = css("--plot") || "#fff"; ctx.fillRect(0, 0, S, S);
      if (!grid?.length) return;
      const ny = grid.length, nx = grid[0].length, cw = S / nx, ch = S / ny;
      const mx = Math.max(...grid.flat().map((v) => Math.log1p(v))) || 1;
      for (let r = 0; r < ny; r++) for (let k = 0; k < nx; k++) {
        const v = grid[r][k];
        if (v) { ctx.fillStyle = densCol(Math.log1p(v) / mx); ctx.fillRect(k * cw, S - (r + 1) * ch, cw + 0.5, ch + 0.5); }
      }
      ctx.strokeStyle = css("--sep-2"); ctx.strokeRect(0.5, 0.5, S - 1, S - 1);
    };
    // the polygon traces itself in, as if drawn: a pixel holds round(100 v), so v sits mid-pixel 100 v
    const poly = g.polygon?.display ?? [];
    const P = (v: number) => (Math.max(-0.01, Math.min(1.01, v)) * 100 + 0.5) * (S / 101);
    const pts = poly.map(([a, b]) => [P(a), S - P(b)]);
    const t0 = performance.now();
    let raf = 0;
    const frame = (now: number) => {
      const f = Math.min(1, (now - t0) / dur);
      paint();
      if (pts.length > 1) {
        const n = Math.max(2, Math.ceil(f * (pts.length + 1)));
        ctx.beginPath(); pts.slice(0, n).forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        if (f === 1) { ctx.closePath(); ctx.fillStyle = g.low_confidence ? "rgba(196,90,0,.10)" : "rgba(215,127,156,.14)"; ctx.fill(); }
        ctx.strokeStyle = g.low_confidence ? css("--c-flag") : css("--fl-core"); ctx.lineWidth = 2;
        ctx.setLineDash(g.low_confidence ? [6, 4] : []); ctx.stroke(); ctx.setLineDash([]);
      }
      if (f < 1) raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [g, S, dur]);
  return (
    <div className="ll-plot" ref={box}>
      <div className="ll-axes" style={{ width: S }}>
        <canvas ref={ref} style={{ width: S, height: S }} aria-label={`${gateLabel(g.gate.split("/").pop() ?? "")}: the model's gate`} />
        <span className="ll-x">{g.x}</span><span className="ll-y">{g.y}</span>
      </div>
    </div>
  );
}

export default function LiveLab({ s }: { s: ExperimentState }) {
  const qc = useQueryClient();
  const { data: live } = useQuery({
    queryKey: ["exp", s.id, "autogate-live"], queryFn: () => api.autogateLive(s.id),
    refetchInterval: (q) => (q.state.data && !q.state.data.running ? false : 700),
  });
  const files = live?.files ?? [], done = live?.done ?? [];
  const [fi, setFi] = useState(0);
  const [gi, setGi] = useState(0);
  const [paused, setPaused] = useState(false);
  const file = done.length ? done[Math.min(fi, done.length - 1)] : undefined;
  const { data: res } = useAutogateResult(s.id, file);
  const gates = (res?.gates ?? []).filter((x) => x.plot?.density?.length);
  const g = gates[Math.min(gi, Math.max(0, gates.length - 1))];
  const behind = done.length - 1 - fi;
  // slower while the models keep pace, quicker when they are ahead; an outline always closes before the next
  const step = behind > 1 ? 320 : 620;

  // step through the gates of each finished file in turn; faster when the models are ahead
  useEffect(() => {
    if (paused || !gates.length) return;
    const t = window.setTimeout(() => {
      if (gi < gates.length - 1) setGi(gi + 1);
      else if (fi < done.length - 1) { setFi(fi + 1); setGi(0); }
    }, step);
    return () => window.clearTimeout(t);
  }, [gi, fi, gates.length, done.length, paused, step]);

  const finished = !!live && !live.running && done.length > 0 && fi >= done.length - 1 && gates.length > 0 && gi >= gates.length - 1;
  const unsure = gates.filter((x) => x.low_confidence).length;
  const stop = () => ui.lab({ live: false });
  const review = async () => {
    if (!file) return;
    try {
      const r = await api.autogateReview(s.id, file);
      await refresh(qc, s.id);
      ui.lab({ live: false, sample: r.sample, parent: r.parent, x: r.x, y: r.y, mode: "density", view: "plot", sel: r.gate });
      ui.toast(`${r.gates.length} model gates are waiting beside FlowJo's: keep or undo them.`);
    } catch (e) { ui.toast((e as Error).message); }
  };
  const pick = (k: number) => { setGi(k); setPaused(true); };

  return (
    <div className="ws-page lab-page live-lab" id="pg-lab">
      <header className="lab-top">
        <div className="lt-l">
          <button className="n-ic" id="lab-exit" title="Back (Esc)" aria-label="Back" onClick={() => { stop(); ui.leaveLab(); }}><Icon.back /></button>
          <Mark small />
          <div className="lt-name"><b>Data lab</b><span>Watching lab {live?.lab ?? "…"} {live?.version ?? ""} gate the files</span></div>
        </div>
        <div className="ll-prog" role="status">
          {live?.running && <i className="live-dot" />}
          <span>{live?.running ? "Gating" : "Gated"} {done.length} of {files.length || "…"} files</span>
          <span className="ll-bar"><i style={{ width: `${files.length ? (100 * done.length) / files.length : 0}%` }} /></span>
        </div>
        <div className="lt-r ll-acts">
          <button className="btn sm" onClick={() => setPaused(!paused)}>{paused ? "Resume" : "Pause"}</button>
          {behind > 0 && <button className="btn sm" onClick={() => { setFi(done.length - 1); setGi(0); setPaused(false); }}>Skip to the latest</button>}
          <button className="btn sm plain" id="ll-stop" onClick={stop}>Stop watching</button>
        </div>
      </header>

      <div className="lab-shell">
        <nav className="lab-nav" aria-label="Files and gates"><div className="ln-scroll">
          <div className="ln-sec">
            <div className="n-h">Files<span>{done.length}/{files.length}</span></div>
            <div className="n-tree">
              {files.map((f) => {
                const at = done.indexOf(f);
                return (
                  <button key={f} className="n-row" aria-selected={f === file} disabled={at < 0} onClick={() => { setFi(at); setGi(0); setPaused(true); }}>
                    <span className="tw">{at >= 0 ? <Icon.check /> : f === live?.current ? <i className="live-dot" /> : null}</span>
                    <span className="nm">{shortSample(f)}</span>
                    <span className="val">{at >= 0 && !(live?.fresh ?? []).includes(f) ? "reused" : f === live?.current ? "gating" : ""}</span>
                  </button>
                );
              })}
            </div>
          </div>
          <div className="ln-sec">
            <div className="n-h">Gates the models drew<span>{file ? shortSample(file) : ""}</span></div>
            <div className="n-tree">
              {gates.map((x, k) => (
                <button key={x.gate} className="n-row" aria-selected={k === gi} style={{ paddingLeft: 6 + Math.min(x.gate.split("/").length, 8) * 8 }} onClick={() => pick(k)} title={x.gate}>
                  <span className="tw">{k < gi ? <Icon.check /> : null}</span>
                  <span className="nm">{gateLabel(x.gate.split("/").pop() ?? "")}</span>
                  {x.low_confidence && <span className="n-pill orange">review</span>}
                  <span className="val">{k <= gi ? `${x.percent_of_parent.toFixed(1)}%` : ""}</span>
                </button>
              ))}
            </div>
          </div>
        </div></nav>

        <main className="lab-stage ll-stage">
          {g ? <LiveCanvas key={`${file}|${g.gate}`} g={g} dur={Math.round(step * 0.55)} />
            : <div className="lab-loading"><Flower /> {live?.running ? "The models are gating the first file…" : "Waiting for a run to gate the files…"}</div>}
          <aside className="ll-side">
            {g && (
              <>
                <div className="n-h">The model's gate<span>{file ? shortSample(file) : ""}</span></div>
                <div className="ln-big"><b>{g.percent_of_parent.toFixed(2)}%</b><span>{gateLabel(g.gate.split("/").pop() ?? "")} of {g.parent ? gateLabel(g.parent.split("/").pop() ?? "") : "all events"}</span></div>
                <div className="n-props">
                  <div className="n-prop"><span className="k">Events</span><span className="v">{g.count.toLocaleString()}</span></div>
                  <div className="n-prop"><span className="k">Drawn by</span><span className="v">{g.model.includes("/unito/") ? `U-Net (${live?.version ?? ""})` : g.model === "template_copy" ? "a copy of the template" : g.model}</span></div>
                  <div className="n-prop"><span className="k">How sure</span><span className="v">{g.confidence == null ? "—" : g.confidence.toFixed(2)}{g.low_confidence ? " · asks for review" : ""}</span></div>
                  {g.flowjo_pct != null && <div className="n-prop"><span className="k">FlowJo</span><span className="v">{g.flowjo_pct.toFixed(2)}%{g.f1 != null ? ` · overlap ${g.f1.toFixed(2)}` : ""}</span></div>}
                </div>
                <p className="n-note">The density is the population the model saw, on the two channels it gates; the outline is the gate it drew. Pink: sure. Orange, dashed: it asks for review.</p>
              </>
            )}
            {finished && (
              <div className="ll-end" role="status">
                <b>The models gated {done.length} file{done.length === 1 ? "" : "s"}.</b>
                <span>{unsure ? `${unsure} gate${unsure === 1 ? "" : "s"} on this file ask${unsure === 1 ? "s" : ""} for review.` : "Every gate on this file is sure."} Review puts them beside FlowJo's for you to keep or undo.</span>
                <div className="row-btns">
                  <button className="btn sm ink" id="ll-review" onClick={review}>Review this file</button>
                  <button className="btn sm" onClick={() => { stop(); ui.leaveLab(); }}>Back to the pipeline</button>
                </div>
              </div>
            )}
          </aside>
        </main>
      </div>
    </div>
  );
}
