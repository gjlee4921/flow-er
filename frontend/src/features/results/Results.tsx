// The results page (a short paper over the saved tables, in Report.tsx) and the population table itself.
import { useMemo, useState } from "react";
import { type ExperimentState } from "../../api";
import { useTable } from "../../api/hooks";
import { fmt, shortSample } from "../../lib";
import { sendInChat } from "../agent/Chat";
import { Report } from "./Report";

export function Results({ s }: { s: ExperimentState }) {
  return <Report s={s} />;
}

export function Measure({ s }: { s: ExperimentState }) {
  const { data: t } = useTable(s.id, s.tables.some((x) => x.name === "population_stats") ? "population_stats" : null);
  const rows = t?.rows ?? [];
  const gateNames = useMemo(() => [...new Set(rows.map((r) => r.gate_path as string))], [rows]);
  const [gate, setGate] = useState<string>("");
  const shown = gate ? rows.filter((r) => r.gate_path === gate) : rows;
  const events = s.snapshot.samples.reduce((a, x) => a + x.events, 0);
  return (
    <div className="ws-page-in">
      <div className="kpis">
        <div className="kpi"><b>{s.snapshot.samples.length}</b><span>files</span></div>
        <div className="kpi"><b>{fmt(events / 1e6, 2)}M</b><span>events</span></div>
        <div className="kpi"><b>{gateNames.length || "—"}</b><span>gates counted</span></div>
        <div className="kpi"><b>{s.tables.length}</b><span>saved tables</span></div>
      </div>
      {!rows.length ? (
        <div className="empty-in"><p>Nothing measured yet. Run the measure step of the plan, and every gate in every file is counted here.</p>
          <button className="btn sm" onClick={() => sendInChat("Run the plan")}>Run the plan</button></div>
      ) : (
        <>
          <div className="n-h">Population table<span>tables/population_stats.csv · {rows.length} rows</span></div>
          <div className="filter-row">
            <select value={gate} onChange={(e) => setGate(e.target.value)} aria-label="Gate">
              <option value="">Every gate</option>
              {gateNames.map((g) => <option key={g} value={g}>{g}</option>)}
            </select>
          </div>
          <div className="n-tblwrap"><table className="n-tbl">
            <thead><tr><th>Sample</th><th>Gate</th><th className="n-r">Count</th><th className="n-r">% parent</th><th className="n-r">% total</th><th>From</th></tr></thead>
            <tbody>{shown.map((r, i) => (
              <tr key={i}><td>{shortSample(String(r.sample))}</td><td title={String(r.gate_path)}>{r.gate}</td>
                <td className="n-r">{fmt(Number(r.count))}</td><td className="n-r">{fmt(Number(r.pct_parent), 2)}</td><td className="n-r">{fmt(Number(r.pct_total), 3)}</td>
                <td><span className={`n-pill ${r.source === "user" ? "orange" : r.source === "model" ? "blue" : "grey"}`}>{r.source === "flowjo" ? "FlowJo" : String(r.source)}</span></td></tr>
            ))}</tbody>
          </table></div>
        </>
      )}
    </div>
  );
}
