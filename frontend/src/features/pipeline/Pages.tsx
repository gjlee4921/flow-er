// The smaller pages of the panel: clean up, models, and the raw files.
import type { ExperimentState, Macro } from "../../api";
import { useModels, useSamples } from "../../api/hooks";
import { Icon } from "../../components/icons";
import { fmt } from "../../lib";
import { askInChat } from "../agent/Chat";

export function Clean({ s }: { s: ExperimentState }) {
  const steps = s.plan.filter((p) => p.kind === "macro");
  const draft = s.draft?.kind === "macro" ? (s.draft.payload as Macro) : null;
  const macros: (Macro & { draft?: boolean })[] = [...s.macros, ...(draft ? [{ ...draft, draft: true }] : [])];
  const status = Object.fromEntries(s.runs.map((r) => [r.step, r]));
  return (
    <div className="ws-page-in">
      <p className="page-lede">The arithmetic. Same answer every time, so nothing here can drift.</p>
      <div className="cards">
        {steps.map((st) => (
          <article key={st.id} className="card wide">
            <div className="card-h"><div className="t"><b>{st.title}</b><span className="s">{st.op.replace(/_/g, " ")}</span></div>
              {status[st.id] && <span className="st">{status[st.id].status}</span>}</div>
            {st.why && <p className="note">{st.why}</p>}
            {st.op === "compensate" && (
              <div className="built-on"><span className="io-k">built on</span>
                <a className="chip-k" href="https://github.com/whitews/FlowKit" target="_blank" rel="noopener">FlowKit<span className="lic">BSD-3</span></a></div>
            )}
            {status[st.id] && <p className="small muted">{status[st.id].summary}</p>}
          </article>
        ))}
        {macros.map((m) => (
          <article key={m.id} className={`card wide${m.draft ? " is-draft" : ""}`}>
            <div className="card-h"><div className="t"><b>{m.title}</b><span className="s">{m.draft ? "written, waiting for keep" : "written by the agent"}</span></div></div>
            {m.summary && <p className="note">{m.summary}</p>}
            {!!m.provenance.length && (
              <div className="built-on"><span className="io-k">built on</span>
                {m.provenance.map((p) => <a key={p.name} className="chip-k" href={p.url} target="_blank" rel="noopener" title={p.what}>{p.name}<span className="lic">{p.licence}</span></a>)}</div>
            )}
            <div className="code">{m.code}</div>
          </article>
        ))}
        {!steps.length && !macros.length && (
          <div className="empty-in"><p>No clean-up steps yet. They appear when the analysis is planned, or ask for a macro in the chat.</p>
            <button className="btn sm" onClick={() => askInChat("Write a macro that ")}>Ask for a macro</button></div>
        )}
      </div>
    </div>
  );
}

export function Models({ s }: { s: ExperimentState }) {
  const { data: m } = useModels();
  const steps = s.plan.filter((p) => p.kind === "model");
  const status = Object.fromEntries(s.runs.map((r) => [r.step, r]));
  return (
    <div className="ws-page-in">
      <p className="page-lede">The judgement calls, each trained on labels you made.</p>
      <div className="cards">
        <article className="card wide">
          <div className="card-h"><div className="t"><b>Gating models</b><span className="s">per-gate U-Net, CytoBERT second opinion</span></div>
            <span className="st">{m?.available ? "available" : "being built"}</span></div>
          <p className="note">{m?.message ?? "…"}</p>
        </article>
        {steps.map((st) => (
          <article key={st.id} className="card wide">
            <div className="card-h"><div className="t"><b>{st.title}</b><span className="s">{st.op.replace(/_/g, " ")}{st.gate ? ` · ${st.gate}` : ""}</span></div>
              {status[st.id] && <span className="st">{status[st.id].status}</span>}</div>
            {st.why && <p className="note">{st.why}</p>}
            {status[st.id] && <p className="small muted">{status[st.id].summary}</p>}
          </article>
        ))}
      </div>
    </div>
  );
}

export function RawData({ s }: { s: ExperimentState }) {
  const { data } = useSamples(s.id);
  return (
    <div className="ws-page-in">
      <p className="page-lede">As it came off the instrument. Nothing compensated, gated or corrected.</p>
      <div className="n-h">Files<span>{s.data_path.split("/").pop()}</span></div>
      <div className="n-tblwrap"><table className="n-tbl">
        <thead><tr><th><Icon.file />File</th><th className="n-r">Events</th></tr></thead>
        <tbody>{data?.samples.map((x) => <tr key={x.id}><td className="n-file"><span><Icon.file /><b>{x.file}</b></span></td><td className="n-r">{fmt(x.events)}</td></tr>)}</tbody>
      </table></div>
      <div className="n-h">Channels<span>{data?.workspace ? `gates from ${data.workspace}` : "no FlowJo workspace"}</span></div>
      <div className="chan-list">{data?.channels.map((c) => <span key={c.id} className="n-pill" title={c.id}>{c.short}</span>)}</div>
    </div>
  );
}
