// The models learning, round by round (demo): how each version of a lab did on samples no version was
// taught, gate by gate, from a fine-tuning run recorded earlier. Each round moves every gate's model %
// toward the expert's, and the error above falls with it. Read-only: nothing here trains or keeps anything.
import { useEffect, useMemo, useState } from "react";
import type { Rounds } from "../../api";
import { useLabRounds } from "../../api/hooks";
import { Flower, Icon, Mark } from "../../components/icons";
import { SERIES, gateLabel, ui } from "../../lib";

const STEP_MS = 2800;
const OFF_PP = 5; // a gate this far from the expert is drawn as wrong
const ROUTE: Record<string, string> = { new: "new model", template: "template copy", current: "kept its model" };
const TISSUES = ["Liver", "Aorta", "Blood", "Spleen"];

const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);

/** Mean error against the expert, per round, for all samples or one tissue, with template copy dashed. */
function Curve({ d, at }: { d: Rounds; at: number }) {
  const W = 420, H = 150, L = 34, R = 14, T = 12, B = 26;
  const keys = d.rounds.map((r) => r.key);
  const scopes = ["all", ...TISSUES.filter((t) => d.means.v001?.[t] != null)];
  const top = Math.max(...keys.flatMap((k) => scopes.map((s) => d.means[k]?.[s] ?? 0))) * 1.08 || 1;
  const X = (i: number) => L + (i / Math.max(1, keys.length - 1)) * (W - L - R);
  const Y = (v: number) => T + (1 - v / top) * (H - T - B);
  const path = (s: string) => keys.map((k, i) => `${i ? "L" : "M"}${X(i)},${Y(d.means[k]?.[s] ?? 0)}`).join(" ");
  return (
    <svg className="rl-curve" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Mean error against the expert, round by round">
      {[0, 0.5, 1].map((f) => <g key={f}><line className="grid" x1={L} x2={W - R} y1={Y(top * f / 1.08)} y2={Y(top * f / 1.08)} />
        <text className="tick" x={L - 6} y={Y(top * f / 1.08) + 4} textAnchor="end">{Math.round(top * f / 1.08)}</text></g>)}
      <line className="rl-ref" x1={L} x2={W - R} y1={Y(d.means.template?.all ?? 0)} y2={Y(d.means.template?.all ?? 0)} />
      <text className="rl-ref-t" x={W - R} y={Y(d.means.template?.all ?? 0) - 5} textAnchor="end">template copy</text>
      {scopes.slice(1).map((s, i) => <path key={s} d={path(s)} fill="none" stroke={SERIES[i % SERIES.length]} strokeWidth={1.4} opacity={0.55} />)}
      <path d={path("all")} fill="none" stroke="var(--fl-core)" strokeWidth={3} />
      {keys.map((k, i) => <g key={k}>
        <circle cx={X(i)} cy={Y(d.means[k]?.all ?? 0)} r={i === at ? 6.5 : 3.5} fill={i <= at ? "var(--fl-core)" : "var(--plot)"} stroke="var(--fl-core)" strokeWidth={2} className="rl-pt" />
        <text className="tick" x={X(i)} y={H - 8} textAnchor="middle" fontWeight={i === at ? 700 : 400}>{k}</text>
      </g>)}
    </svg>
  );
}

function CurveKey({ d }: { d: Rounds }) {
  const tissues = TISSUES.filter((t) => d.means.v001?.[t] != null);
  return (
    <div className="rl-key">
      <span><i className="all" />every sample</span>
      {tissues.map((t, i) => <span key={t}><i style={{ background: SERIES[i % SERIES.length] }} />{t.toLowerCase()}</span>)}
      <span><i className="ref" />template copy</span>
    </div>
  );
}

export default function RoundsLab({ lab }: { lab: string }) {
  const { data: d, error } = useLabRounds(lab);
  const [at, setAt] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [item, setItem] = useState("all");

  // play through the rounds once, a little longer on the first so the starting point reads
  useEffect(() => {
    if (!playing || !d) return;
    if (at >= d.rounds.length - 1) { setPlaying(false); return; }
    const t = window.setTimeout(() => setAt(at + 1), at === 0 ? STEP_MS + 800 : STEP_MS);
    return () => window.clearTimeout(t);
  }, [playing, at, d]);

  const r = d?.rounds[at];
  const k = r?.key ?? "v001";
  // per gate: the model's [%, expert %, F1] on the picked sample, or the mean error over every sample
  const rows = useMemo(() => {
    if (!d) return [];
    const items = item === "all" ? d.items.map((i) => i.id) : [item];
    return d.gates.map((g) => {
      const vals = (key: string) => items.map((i) => d.values[i]?.[g.path]?.[key]).filter(Boolean) as [number, number, number | null][];
      const m = vals(k), t = vals("template");
      return { g, one: item === "all" ? null : m[0], ref: item === "all" ? null : t[0],
        err: mean(m.map(([p, e]) => Math.abs(p - e))), refErr: mean(t.map(([p, e]) => Math.abs(p - e))), first: mean(vals("v001").map(([p, e]) => Math.abs(p - e))), has: m.length > 0 };
    }).filter((x) => x.has); // a gate this sample's tissue does not have is left out, not counted as 0
  }, [d, item, k]);
  if (error) return <div className="ws-page lab-page"><div className="lab-loading">{(error as Error).message}</div></div>;
  if (!d || !r) return <div className="ws-page lab-page"><div className="lab-loading"><Flower /> Reading the rounds…</div></div>;

  const scopeErr = item === "all" ? d.means[k]?.all ?? 0 : mean(rows.map((x) => x.err));
  const refErr = item === "all" ? d.means.template?.all ?? 0 : mean(rows.map((x) => x.refErr));
  const startErr = item === "all" ? d.means.v001?.all ?? 0 : mean(rows.map((x) => x.first));
  const scale = Math.max(1, ...rows.map((x) => x.first), ...rows.map((x) => x.refErr));
  const right = rows.filter((x) => x.err <= OFF_PP).length;
  const never = d.items.map((i) => `${i.tissue.toLowerCase()} ${i.mouse}`);
  const pick = (i: number) => { setAt(i); setPlaying(false); };

  return (
    <div className="ws-page lab-page rounds-lab" id="pg-lab">
      <header className="lab-top">
        <div className="lt-l">
          <button className="n-ic" id="rl-exit" title="Back to the data lab" aria-label="Back" onClick={() => ui.lab({ rounds: null })}><Icon.back /></button>
          <Mark small />
          <div className="lt-name"><b>Data lab</b><span>Lab {d.lab}: the models learn, round by round</span></div>
        </div>
        <div className="rl-steps" role="tablist" aria-label="Rounds">
          {d.rounds.map((x, i) => (
            <button key={x.key} role="tab" aria-selected={i === at} className={`rl-step${i < at ? " past" : ""}`} onClick={() => pick(i)}>
              <b>{x.label}</b><span>{i === 0 ? "start" : `round ${i}`}</span>
            </button>
          ))}
        </div>
        <div className="lt-r">
          <button className="btn sm" id="rl-play" onClick={() => { if (at >= d.rounds.length - 1) setAt(0); setPlaying(!playing || at >= d.rounds.length - 1); }}>
            {playing ? "Pause" : at >= d.rounds.length - 1 ? "Play again" : "Play"}</button>
        </div>
      </header>

      <div className="lab-shell">
        <nav className="lab-nav" aria-label="Samples"><div className="ln-scroll">
          <div className="ln-sec">
            <div className="n-h">Never taught<span>{d.items.length}</span></div>
            <div className="n-tree">
              <button className="n-row" aria-selected={item === "all"} onClick={() => setItem("all")}>
                <span className="tw" /><span className="nm">Every sample</span><span className="val">{(d.means[k]?.all ?? 0).toFixed(1)}</span>
              </button>
              {TISSUES.filter((t) => d.items.some((i) => i.tissue === t)).map((t) => (
                <div key={t}>
                  <div className="rl-tissue">{t}<span>{(d.means[k]?.[t] ?? 0).toFixed(1)} pp</span></div>
                  {d.items.filter((i) => i.tissue === t).map((i) => (
                    <button key={i.id} className="n-row" aria-selected={item === i.id} onClick={() => setItem(i.id)} style={{ paddingLeft: 18 }}>
                      <span className="tw" /><span className="nm">mouse {i.mouse}</span>
                    </button>
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div></nav>

        <main className="lab-stage rl-stage">
          <section className="rl-hero">
            <div className="rl-big">
              <span className="k">{item === "all" ? "Every never-taught sample" : `${item.split(" ")[0]} of mouse ${item.split(" ")[1]}`} · mean error against the expert</span>
              <b key={`${k}-${item}`} className="rl-num">{scopeErr.toFixed(2)}<small> pp</small></b>
              <span className="rl-sub">
                {at === 0 ? "The human ALS models, on mouse tissue they have never seen" : `from ${startErr.toFixed(1)} pp at the start`}
                {" · "}template copy {refErr.toFixed(2)} pp{" · "}{right} of {rows.length} gates within {OFF_PP} pp
              </span>
            </div>
            <div className="rl-curvebox"><Curve d={d} at={at} /><CurveKey d={d} /></div>
          </section>

          <div className="rl-round" role="status">
            <b>{r.label}</b>
            {at === 0 ? <span>Each mouse gate starts from the human model for the same kind of plot.</span> : <>
              <span>Taught {r.taught}.</span>
              {r.reason && <span className="rl-verdict">{r.promoted ? "Promoted" : "Not promoted"}: {r.reason.replace(/->/g, "→").replace(/(\d+\.\d)\d+/g, "$1")}</span>}
              {r.routes && <span className="rl-routes">{Object.entries(r.routes).map(([route, n]) => <i key={route} className={`rl-route ${route}`}>{n} {ROUTE[route] ?? route}</i>)}</span>}
            </>}
          </div>

          <div className="rl-ladder" aria-label="Gate by gate">
            <div className="rl-lh"><span>Gate</span><span>{item === "all" ? "Error against the expert (pp)" : "% of parent: the model ● against the expert │ (template copy ○)"}</span><span /></div>
            {rows.map(({ g, one, ref, err, refErr: te }) => (
              <div key={g.path} className={`rl-row${err > OFF_PP ? " off" : ""}`} title={g.path}>
                <span className="rl-g" style={{ paddingLeft: 4 + g.depth * 9 }}>{gateLabel(g.name)}<small>{g.parent ? ` · ${gateLabel(g.parent)}` : ""}</small></span>
                {one ? (
                  <span className="rl-track">
                    <i className="rl-exp" style={{ left: `${one[1]}%` }} />
                    {ref && <i className="rl-ghost" style={{ left: `${ref[0]}%` }} />}
                    <i className="rl-dot" style={{ left: `${one[0]}%` }} />
                  </span>
                ) : (
                  <span className="rl-track err">
                    <i className="rl-bar" style={{ width: `${Math.min(100, (err / scale) * 100)}%` }} />
                    <i className="rl-exp ref" style={{ left: `${Math.min(100, (te / scale) * 100)}%` }} />
                  </span>
                )}
                <span className="rl-v">{one ? `${one[0].toFixed(1)}% · ${one[1].toFixed(1)}%` : `${err.toFixed(1)}`}</span>
              </div>
            ))}
          </div>
          <p className="n-note rl-note">Recorded earlier on a GPU, from GJ's fine-tuning run. Every version is scored on samples none of them was taught ({never.join(", ")}), as % of parent against the expert's FlowJo gates, each gate on the expert's parent population.</p>
        </main>
      </div>
    </div>
  );
}
