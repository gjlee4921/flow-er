// The data lab's sheet: one row per gate per file, with any statistic in the toolkit as a column.
// Filter the rows, sort by any column, colour a column by its values, chart a column as bars with
// each file as a dot, copy it all as CSV, and click a row to see that gate in that file. The lab
// assistant drives the same sheet (its columns, filter, sort, colour and chart live in the lab state).
import { useEffect, useMemo, useState } from "react";
import type { Channel, ExperimentState, GateNode, StatRow } from "../../api";
import { useStats, useToolkit } from "../../api/hooks";
import { Flower, Icon } from "../../components/icons";
import { fmt, shortSample, ui, useUi } from "../../lib";

type Col = { key: string; label: string; num: boolean; spec?: string; show: (r: StatRow) => string };
/** The sheet as CSV, for the toolbar's Export while the Sheet is open. */
export const sheetCsv: { get: (() => string) | null } = { get: null };
const BASE: Col[] = [
  { key: "sample", label: "File", num: false, show: (r) => shortSample(r.sample) },
  { key: "gate", label: "Gate", num: false, show: (r) => r.gate },
  { key: "count", label: "Count", num: true, show: (r) => fmt(r.count) },
  { key: "pct_parent", label: "% parent", num: true, show: (r) => r.pct_parent.toFixed(2) },
  { key: "pct_total", label: "% total", num: true, show: (r) => r.pct_total.toFixed(3) },
  { key: "source", label: "From", num: false, show: (r) => ({ flowjo: "FlowJo", user: "you", model: "model" } as Record<string, string>)[String(r.source)] ?? "" },
];
const BUILT_IN = new Set(["count", "pct_parent", "pct_total"]);

/** Bars with each file as a dot: one bar per gate, its height the mean across the files shown. */
function SheetChart({ rows, col }: { rows: StatRow[]; col: Col }) {
  const gates = [...new Set(rows.map((r) => r.gate_path))];
  const val = (r: StatRow) => Number(r[col.key]);
  const groups = gates.map((g) => {
    const rs = rows.filter((r) => r.gate_path === g && Number.isFinite(val(r)));
    return { gate: rs[0]?.gate ?? g, rs, mean: rs.length ? rs.reduce((a, r) => a + val(r), 0) / rs.length : 0 };
  }).filter((g) => g.rs.length);
  if (!groups.length) return null;
  const W = 640, H = 230, L = 54, R = 14, T = 16, B = 44, pw = W - L - R, ph = H - T - B;
  const top = Math.max(...groups.flatMap((g) => g.rs.map(val)), 1e-9) * 1.12;
  const step = [1, 2, 2.5, 5].map((m) => m * Math.pow(10, Math.floor(Math.log10(top / 4)))).find((s) => top / s <= 5) ?? top / 4;
  const ticks = Array.from({ length: Math.floor(top / step) + 1 }, (_, i) => i * step);
  const band = pw / groups.length, bw = Math.min(56, band * 0.5);
  const Y = (v: number) => T + ph - (v / top) * ph;
  const files = [...new Set(rows.map((r) => r.sample))];
  return (
    <figure className="sheet-chart">
      <figcaption><b>{col.label}</b> by gate · bar is the mean across files, each dot one file</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${col.label} by gate`}>
        {ticks.map((t) => <g key={t}><line className="grid" x1={L} x2={W - R} y1={Y(t)} y2={Y(t)} /><text className="tick" x={L - 6} y={Y(t) + 4} textAnchor="end">{t >= 100 ? fmt(t) : +t.toFixed(2)}</text></g>)}
        {groups.map((g, i) => {
          const cx = L + band * (i + 0.5);
          return (
            <g key={g.gate}>
              <rect className="sc-bar" x={cx - bw / 2} y={Y(g.mean)} width={bw} height={Y(0) - Y(g.mean)}><title>{`${g.gate}: mean ${fmt(g.mean, 2)}`}</title></rect>
              {g.rs.map((r) => {
                const k = files.indexOf(r.sample), jitter = (k - (files.length - 1) / 2) * Math.min(9, bw / Math.max(1, files.length));
                return <circle key={r.sample} className="sc-dot" cx={cx + jitter} cy={Y(val(r))} r={4}><title>{`${shortSample(r.sample)} · ${g.gate}: ${fmt(val(r), 2)}`}</title></circle>;
              })}
              <text className="tick" x={cx} y={H - B + 16} textAnchor="middle">{g.gate}</text>
            </g>
          );
        })}
      </svg>
    </figure>
  );
}

export default function LabSheet({ s, channels, tree }: { s: ExperimentState; channels: Channel[]; tree: GateNode[] }) {
  const lab = useUi((u) => u.lab);
  const { data: kit } = useToolkit();
  const { data, isFetching } = useStats(s.id, lab.sheetGates, lab.sheetSpec);
  const [adding, setAdding] = useState(false);
  const [stat, setStat] = useState("median");
  const [param, setParam] = useState("");
  const [pct, setPct] = useState("90");
  const [anc, setAnc] = useState("");
  const [menu, setMenu] = useState<string | null>(null);
  const order = useMemo(() => Object.fromEntries(tree.map((g, i) => [g.key, i])), [tree]);

  const cols: Col[] = [...BASE, ...(data?.columns ?? []).map((c) => ({
    key: c.label, label: c.label, num: true, spec: c.spec,
    show: (r: StatRow) => (r[c.label] == null ? "—" : fmt(Number(r[c.label]), Math.abs(Number(r[c.label])) >= 100 ? 1 : 2)),
  }))];
  const colOf = (key: string | null) => cols.find((c) => c.key === key || c.label === key);

  const rows = useMemo(() => {
    const f = lab.sheetFilter.trim().toLowerCase();
    const out = (data?.rows ?? []).filter((r) => !f || `${shortSample(r.sample)} ${r.gate}`.toLowerCase().includes(f));
    const sc = lab.sheetSort ? colOf(lab.sheetSort.key) : null;
    if (sc) {
      const dir = lab.sheetSort!.desc ? -1 : 1;
      out.sort((a, b) => {
        const x = a[sc.key], y = b[sc.key];
        return (typeof x === "number" && typeof y === "number" ? x - y : String(x ?? "").localeCompare(String(y ?? ""))) * dir;
      });
    } else {
      out.sort((a, b) => (order[a.gate_path] ?? -1) - (order[b.gate_path] ?? -1) || a.sample.localeCompare(b.sample));
    }
    return out;
  }, [data, lab.sheetFilter, lab.sheetSort, order]); // eslint-disable-line react-hooks/exhaustive-deps

  // a colour scale on one column: light to the flower's rose, from its lowest value to its highest
  const tint = useMemo(() => {
    const c = colOf(lab.sheetColour);
    if (!c?.num) return null;
    const vals = rows.map((r) => Number(r[c.key])).filter(Number.isFinite);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    return { key: c.key, at: (v: number) => (hi > lo ? (v - lo) / (hi - lo) : 0.5) };
  }, [rows, lab.sheetColour]); // eslint-disable-line react-hooks/exhaustive-deps

  const csv = () => {
    const esc = (v: unknown) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v ?? ""));
    return [cols.map((c) => esc(c.label)).join(","), ...rows.map((r) => cols.map((c) => esc(c.key === "sample" ? shortSample(r.sample) : r[c.key])).join(","))].join("\n");
  };
  useEffect(() => { sheetCsv.get = csv; return () => { sheetCsv.get = null; }; });
  const copy = async () => {
    try { await navigator.clipboard.writeText(csv()); ui.toast(`Copied ${rows.length} rows as CSV.`); } catch { ui.toast("The browser did not allow copying."); }
  };
  const open = (r: StatRow) => {
    const g = tree.find((t) => t.key === r.gate_path);
    ui.lab({ view: "plot", compare: false, sample: r.sample, sel: g?.key ?? null, parent: g?.parent ?? null,
      ...(g?.dims.length === 2 ? { x: g.dims[0], y: g.dims[1] } : {}) });
  };
  const sortBy = (key: string) => {
    const s0 = lab.sheetSort;
    ui.lab({ sheetSort: s0?.key !== key ? { key, desc: false } : s0.desc ? null : { key, desc: true } });
  };
  const needs = kit?.statistics.find((x) => x.id === stat)?.needs ?? [];
  const add = () => {
    if (needs.includes("parameter") && !param) return;
    const spec = `${stat}:${needs.includes("parameter") ? param : ""}${needs.includes("p") ? `:${pct}` : needs.includes("ancestor") ? `:${anc}` : ""}`;
    if (!lab.sheetSpec.includes(spec)) ui.lab({ sheetSpec: [...lab.sheetSpec, spec] });
    setAdding(false);
  };
  const groups = [["scatter", "Scatter"], ["marker", "Markers"], ["time", "Time"]] as const;
  const chartCol = colOf(lab.sheetChart);

  return (
    <section className="lab-view lab-v-doc lab-sheet"><div>
      <div className="sheet-bar">
        <label className="tb-sel"><span>Gates</span>
          <select value={lab.sheetGates[0] ?? ""} onChange={(e) => ui.lab({ sheetGates: e.target.value ? [e.target.value] : [] })} aria-label="Gates in the sheet">
            <option value="">Every gate</option>
            {tree.map((g) => <option key={g.key} value={g.key}>{"  ".repeat(g.key.split("/").length - 1)}{g.name}</option>)}
          </select>
        </label>
        <label className="st-find sheet-filter"><Icon.search /><input id="sheet-filter" value={lab.sheetFilter} onChange={(e) => ui.lab({ sheetFilter: e.target.value })} placeholder="Filter rows" aria-label="Filter rows" /></label>
        <div className="sheet-add">
          <button className="btn sm plain" id="sheet-add" onClick={() => setAdding(!adding)} aria-expanded={adding}><Icon.plus />Add a statistic</button>
          {adding && (
            <form className="sheet-pop" onSubmit={(e) => { e.preventDefault(); add(); }}>
              <label>Statistic
                <select id="add-stat" value={stat} onChange={(e) => setStat(e.target.value)}>
                  {kit?.statistics.filter((x) => !BUILT_IN.has(x.id)).map((x) => <option key={x.id} value={x.id}>{x.name}</option>)}
                </select>
              </label>
              {needs.includes("parameter") && (
                <label>Of
                  <select id="add-param" value={param} onChange={(e) => setParam(e.target.value)}>
                    <option value="">Choose a parameter…</option>
                    {groups.map(([g, t]) => <optgroup key={g} label={t}>{channels.filter((c) => c.group === g).map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}</optgroup>)}
                  </select>
                </label>
              )}
              {needs.includes("p") && <label>Percentile<input id="add-p" type="number" min={1} max={99} value={pct} onChange={(e) => setPct(e.target.value)} /></label>}
              {needs.includes("ancestor") && (
                <label>Of the gate
                  <select id="add-anc" value={anc} onChange={(e) => setAnc(e.target.value)}>
                    <option value="">All events</option>
                    {tree.map((g) => <option key={g.key} value={g.key}>{g.name}</option>)}
                  </select>
                </label>
              )}
              <div className="pm-btns"><button type="button" className="btn sm plain" onClick={() => setAdding(false)}>Cancel</button>
                <button className="btn sm ink" disabled={needs.includes("parameter") && !param}>Add column</button></div>
            </form>
          )}
        </div>
        <span className="sheet-n">{isFetching ? <Flower className="sm" /> : null}{rows.length} rows · every event · statistics on compensated values</span>
        <button className="btn sm plain" onClick={copy} disabled={!rows.length}><Icon.copy />Copy CSV</button>
      </div>

      {chartCol && <SheetChart rows={rows} col={chartCol} />}

      <div className="n-tblwrap"><table className="n-tbl">
        <thead><tr>{cols.map((c) => (
          <th key={c.key} className={`${c.num ? "n-r" : ""}${tint?.key === c.key ? " tinted" : ""}`} aria-sort={lab.sheetSort?.key === c.key ? (lab.sheetSort.desc ? "descending" : "ascending") : "none"}>
            <span className="th-in">
              <button className="th-sort" onClick={() => sortBy(c.key)}>{c.label}{lab.sheetSort?.key === c.key ? (lab.sheetSort.desc ? " ↓" : " ↑") : ""}</button>
              {c.num && (
                <span className="th-menu">
                  <button className="th-more" aria-label={`More for ${c.label}`} onClick={() => setMenu(menu === c.key ? null : c.key)}>⋯</button>
                  {menu === c.key && (
                    <span className="th-pop" role="menu">
                      <button role="menuitem" onClick={() => { ui.lab({ sheetColour: lab.sheetColour === c.key ? null : c.key }); setMenu(null); }}>{lab.sheetColour === c.key ? "Remove colour scale" : "Colour scale"}</button>
                      <button role="menuitem" onClick={() => { ui.lab({ sheetChart: lab.sheetChart === c.key ? null : c.key }); setMenu(null); }}>{lab.sheetChart === c.key ? "Hide chart" : "Chart: bars with each file as a dot"}</button>
                      {c.spec && <button role="menuitem" onClick={() => { ui.lab({ sheetSpec: lab.sheetSpec.filter((x) => x !== c.spec) }); setMenu(null); }}>Remove column</button>}
                    </span>
                  )}
                </span>
              )}
            </span>
          </th>
        ))}</tr></thead>
        <tbody>{rows.map((r) => (
          <tr key={r.sample + r.gate_path} onClick={() => open(r)} title="Show this gate in this file">
            {cols.map((c) => {
              const t = tint?.key === c.key && Number.isFinite(Number(r[c.key])) ? tint.at(Number(r[c.key])) : null;
              return <td key={c.key} className={c.num ? "n-r" : ""} style={t != null ? { background: `color-mix(in srgb, var(--fl) ${Math.round(8 + t * 42)}%, transparent)` } : undefined}>{c.show(r)}</td>;
            })}
          </tr>
        ))}</tbody>
      </table></div>
    </div></section>
  );
}
