// A figure the agent wrote, drawn from the rows of a saved table. Every mark has its numbers on
// hover; the legend hides and shows series; clicking a category picks it out.
import { useMemo, useState } from "react";
import type { View } from "../../api";
import { esc, fmt, SERIES, shortSample } from "../../lib";
import AnatomyMap from "./AnatomyMap";
import { selKey, setSelection, useSelection } from "./selection";
import { Tip, useTip } from "./Tip";

const W = 640;
const label = (v: unknown) => {
  const s = String(v ?? "");
  if (/\.fcs$/i.test(s)) return shortSample(s);
  if (s.includes("/")) return s.split("/").pop()!;
  return s;
};
const num = (v: unknown) => (typeof v === "number" ? v : Number(v));
const nice = (v: number) => (Math.abs(v) >= 100 ? fmt(v) : Math.abs(v) >= 1 ? fmt(v, 2) : fmt(v, 3));

/** Round tick values from lo to hi (lo is 0 unless some values are negative), always through 0. */
function ticks(lo: number, hi: number, n = 4) {
  const span = hi - lo;
  if (span <= 0) return [0];
  const step = Math.pow(10, Math.floor(Math.log10(span / n)));
  const s = [1, 2, 2.5, 5, 10].map((k) => k * step).find((k) => span / k <= n) ?? step * 10;
  const out = [];
  for (let v = Math.ceil(lo / s - 1e-9) * s; v <= hi * 1.0001 + 1e-12; v += s) out.push(+v.toFixed(10));
  return out;
}

/** Up to two lines for a long axis title, split at the space nearest the middle. */
function lines(t: string, max = 34): string[] {
  if (t.length <= max) return [t];
  const mid = t.length / 2;
  const at = [...t.matchAll(/ /g)].map((m) => m.index!).sort((a, b) => Math.abs(a - mid) - Math.abs(b - mid))[0];
  return at == null ? [t] : [t.slice(0, at), t.slice(at + 1)];
}

export default function ViewChart({ view }: { view: View }) {
  const { spec, rows } = view;
  const tip = useTip();
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const hot = useSelection(); // shared with every other figure: picking a mark here picks it out there too

  const groups = useMemo(() => {
    if (!spec.color) return ["_"];
    return [...new Set(rows.map((r) => String(r[spec.color!])))];
  }, [rows, spec.color]);
  const colour = (g: string) => SERIES[groups.indexOf(g) % SERIES.length];
  const xs = useMemo(() => [...new Set(rows.map((r) => String(r[spec.x ?? ""])))], [rows, spec.x]);

  if (!rows.length) return <p className="note">The table has no rows for this view yet.</p>;
  if (spec.type === "table") return <TableView view={view} />;
  if (spec.type === "anatomy") return <AnatomyMap view={view} />;
  // a picked label dims the other marks, but only when this figure has it (an organ picked elsewhere
  // leaves a per-file chart alone)
  const keyOf = (r: Record<string, any>) => [selKey(label(r[spec.x ?? ""])), spec.color ? selKey(label(r[spec.color])) : ""];
  const has = hot != null && rows.some((r) => keyOf(r).includes(hot));
  const dimOf = (r: Record<string, any>) => (has && !keyOf(r).includes(hot!) ? 0.25 : 1);

  const shown = rows.filter((r) => !spec.color || !hidden.has(String(r[spec.color])));
  const yv = (r: Record<string, any>) => num(r[spec.y!]);
  // the y range covers every value, below zero too (a response can be negative), with a little room
  const vals = shown.map(yv).filter(Number.isFinite);
  const top = Math.max(0, ...vals), bottom = Math.min(0, ...vals);
  const pad = Math.max(1e-9, top - bottom) * 0.1;
  const ymax = top > 0 || bottom === 0 ? top + pad : 0, ymin = bottom < 0 ? bottom - pad : 0;
  const ytitle = lines(String(spec.y_label || spec.y || ""));
  const H = 300, L = ytitle.length > 1 ? 70 : 56, R = 16, T = 18, B = 58;
  const pw = W - L - R, ph = H - T - B;
  const Y = (v: number) => T + ph - ((v - ymin) / Math.max(1e-9, ymax - ymin)) * ph;
  const tipFor = (r: Record<string, any>) =>
    [`<b>${esc(label(r[spec.x ?? ""]))}</b>`, spec.color ? esc(label(r[spec.color])) : "",
     `${esc(spec.y_label || spec.y)}: <b>${nice(yv(r))}</b>`, r.count != null && spec.y !== "count" ? `${fmt(num(r.count))} events` : "",
     r.n != null ? `n = ${r.n}` : ""].filter(Boolean).join("<br>");

  let marks: JSX.Element[] = [];
  const band = pw / Math.max(1, xs.length);
  const xc = (x: string) => L + band * (xs.indexOf(x) + 0.5);
  const visibleGroups = groups.filter((g) => !hidden.has(g));

  if (spec.type === "bar") {
    const bw = Math.min(42, (band * 0.72) / Math.max(1, visibleGroups.length));
    marks = shown.map((r, i) => {
      const g = spec.color ? String(r[spec.color]) : "_";
      const x0 = xc(String(r[spec.x!])) - (bw * visibleGroups.length) / 2 + bw * visibleGroups.indexOf(g);
      const dim = dimOf(r);
      // from the zero line, up for positive values and down for negative ones
      return <rect key={i} x={x0 + 1} y={Math.min(Y(yv(r)), Y(0))} width={bw - 2} height={Math.max(1, Math.abs(Y(0) - Y(yv(r))))} rx={3}
        fill={colour(g)} opacity={dim} onMouseMove={(e) => tip.show(tipFor(r), e)} onMouseLeave={tip.hide}
        onClick={() => setSelection(label(r[spec.x!]))} style={{ cursor: "pointer" }} />;
    });
  } else if (spec.type === "dot" || spec.type === "dumbbell") {
    if (spec.type === "dumbbell" && visibleGroups.length === 2) {
      marks = xs.map((x) => {
        const a = shown.find((r) => String(r[spec.x!]) === x && String(r[spec.color!]) === visibleGroups[0]);
        const b = shown.find((r) => String(r[spec.x!]) === x && String(r[spec.color!]) === visibleGroups[1]);
        if (!a || !b) return <g key={x} />;
        return <line key={`l-${x}`} x1={xc(x)} x2={xc(x)} y1={Y(yv(a))} y2={Y(yv(b))} stroke="var(--sep-2)" strokeWidth={2.5} />;
      });
    }
    // each group gets its own column within a category, and dots that would overlap there step
    // aside (a small beeswarm), so every donor stays visible
    const sub = spec.type === "dot" ? Math.min(56, band / (visibleGroups.length + 1)) : 0;
    const placed = new Map<string, [number, number][]>();
    const at = new Map<number, number>();
    [...shown.keys()].sort((a, b) => yv(shown[a]) - yv(shown[b])).forEach((i) => {
      const r = shown[i], g = spec.color ? String(r[spec.color]) : "_";
      const cx = xc(String(r[spec.x!])) + (visibleGroups.indexOf(g) - (visibleGroups.length - 1) / 2) * sub, cy = Y(yv(r));
      const cell = placed.get(`${r[spec.x!]}|${g}`) ?? [];
      const dx = spec.type === "dot" ? [0, 9, -9, 18, -18, 27, -27].find((d) => cell.every(([x, y]) => Math.hypot(x - (cx + d), y - cy) >= 12.5)) ?? 0 : 0;
      cell.push([cx + dx, cy]);
      placed.set(`${r[spec.x!]}|${g}`, cell);
      at.set(i, cx + dx);
    });
    marks = marks.concat(shown.map((r, i) => {
      const g = spec.color ? String(r[spec.color]) : "_";
      const dim = dimOf(r);
      return <circle key={`d-${i}`} cx={at.get(i)} cy={Y(yv(r))} r={6} fill={colour(g)} opacity={dim}
        stroke="var(--plot)" strokeWidth={1.5} onMouseMove={(e) => tip.show(tipFor(r), e)} onMouseLeave={tip.hide}
        onClick={() => setSelection(label(r[spec.x!]))} style={{ cursor: "pointer" }} />;
    }));
  } else if (spec.type === "line") {
    marks = visibleGroups.flatMap((g) => {
      const pts = shown.filter((r) => !spec.color || String(r[spec.color]) === g).map((r) => [xc(String(r[spec.x!])), Y(yv(r)), r] as const);
      return [<polyline key={`p-${g}`} points={pts.map(([x, y]) => `${x},${y}`).join(" ")} fill="none" stroke={colour(g)} strokeWidth={2.4} />,
        ...pts.map(([x, y, r], i) => <circle key={`c-${g}-${i}`} cx={x} cy={y} r={4} fill={colour(g)}
          onMouseMove={(e) => tip.show(tipFor(r), e)} onMouseLeave={tip.hide} />)];
    });
  } else if (spec.type === "scatter") {
    const xv = (r: Record<string, any>) => num(r[spec.x!]);
    const xmax = Math.max(1e-9, ...shown.map(xv).filter(Number.isFinite)) * 1.08;
    marks = shown.map((r, i) => {
      const g = spec.color ? String(r[spec.color]) : "_";
      return <circle key={i} cx={L + (xv(r) / xmax) * pw} cy={Y(yv(r))} r={5} fill={colour(g)} opacity={0.85}
        onMouseMove={(e) => tip.show(tipFor(r), e)} onMouseLeave={tip.hide} />;
    });
  } else if (spec.type === "heatmap") {
    const cols = visibleGroups, cw = pw / Math.max(1, xs.length), chh = ph / Math.max(1, cols.length);
    const max = Math.max(1e-9, ...shown.map(yv));
    marks = shown.map((r, i) => {
      const g = String(r[spec.color!]);
      const k = yv(r) / max;
      return <rect key={i} x={L + cw * xs.indexOf(String(r[spec.x!]))} y={T + chh * cols.indexOf(g)} width={cw - 2} height={chh - 2} rx={3}
        fill="var(--c-trt)" opacity={0.1 + 0.9 * k} onMouseMove={(e) => tip.show(tipFor(r), e)} onMouseLeave={tip.hide} />;
    });
  }

  const yt = spec.type === "heatmap" ? [] : ticks(ymin, ymax);
  const step = yt.length > 1 ? yt[1] - yt[0] : 1;
  const tickLabel = (t: number) => fmt(t, step >= 1 ? 0 : Math.min(3, Math.ceil(-Math.log10(step))));
  return (
    <div className="vchart">
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={spec.title}>
        {yt.map((t) => <g key={t}><line className="grid" x1={L} x2={W - R} y1={Y(t)} y2={Y(t)} /><text className="tick" x={L - 8} y={Y(t) + 4} textAnchor="end">{tickLabel(t)}</text></g>)}
        {ymin < 0 && spec.type !== "heatmap" && <line className="zero" x1={L} x2={W - R} y1={Y(0)} y2={Y(0)} />}
        {marks}
        {spec.type !== "scatter" && xs.map((x) => (
          <text key={x} className="tick" x={spec.type === "heatmap" ? L + (pw / xs.length) * (xs.indexOf(x) + 0.5) : xc(x)} y={H - B + 18}
            textAnchor="middle" style={{ cursor: "pointer", fontWeight: hot === selKey(label(x)) ? 600 : 400 }} onClick={() => setSelection(label(x))}>{label(x)}</text>
        ))}
        {spec.type === "heatmap" && visibleGroups.map((g, i) => <text key={g} className="tick" x={L - 8} y={T + (ph / visibleGroups.length) * (i + 0.5) + 4} textAnchor="end">{label(g)}</text>)}
        <text className="axis-t" x={L + pw / 2} y={H - 8} textAnchor="middle">{spec.x_label || spec.x}</text>
        {spec.type !== "heatmap" && (
          <text className="axis-t" x={14} y={T + ph / 2} textAnchor="middle" transform={`rotate(-90 14 ${T + ph / 2})`}>
            {ytitle.map((t, i) => <tspan key={i} x={14} dy={i ? 14 : 0}>{t}</tspan>)}
          </text>
        )}
      </svg>
      {spec.color && (
        <div className="legend">
          {groups.map((g) => (
            <button key={g} className={`lg-t${hidden.has(g) ? " off" : ""}`} title={`Click to ${hidden.has(g) ? "show" : "hide"}`}
              onClick={() => { const n = new Set(hidden); n.has(g) ? n.delete(g) : n.add(g); setHidden(n); }}>
              <i style={{ background: colour(g) }} />{label(g)}
            </button>
          ))}
        </div>
      )}
      <Tip state={tip.state} />
    </div>
  );
}

function TableView({ view }: { view: View }) {
  const cols = view.spec.columns.length ? view.spec.columns : Object.keys(view.rows[0] ?? {});
  return (
    <div className="n-tblwrap">
      <table className="n-tbl">
        <thead><tr>{cols.map((c) => <th key={c} className={typeof view.rows[0]?.[c] === "number" ? "n-r" : ""}>{c}</th>)}</tr></thead>
        <tbody>{view.rows.map((r, i) => (
          <tr key={i}>{cols.map((c) => <td key={c} className={typeof r[c] === "number" ? "n-r" : ""}>{typeof r[c] === "number" ? nice(r[c]) : label(r[c])}</td>)}</tr>
        ))}</tbody>
      </table>
    </div>
  );
}
