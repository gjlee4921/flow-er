// The results page reads like a short paper about the experiment: a title that says what was found,
// a summary whose numbers point at their table cells, numbered live figures, and methods and a data
// list written by code from the files and the kept workflow. Nothing on it is typed in by hand.
import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api, type ExperimentState, type View } from "../../api";
import { refresh, useViews } from "../../api/hooks";
import { useReport, type BoundNumber, type Segment } from "../../api/report";
import { Tip, useTip, ViewChart } from "../../components/plots";
import { clearSelection, useSelectionLabel } from "../../components/plots/selection";
import { esc, fmt, ui } from "../../lib";
import { askInChat, sendInChat } from "../agent/Chat";
import { GatingStrategy } from "./GatingStrategy";
import "../../styles/report.css";

type TipApi = ReturnType<typeof useTip>;

const behind = (b: BoundNumber) =>
  [`<b>${esc(b.display)}</b> · ${esc(b.column)}${b.agg ? ` (${esc(b.agg)} of ${b.n} rows)` : ""}`,
   `from <b>${esc(b.table)}</b>${b.step ? `, made by step <b>${esc(b.step)}</b>` : ""}`,
   ...b.rows.slice(0, 4).map((r) => esc(Object.values(r).join(" · "))), b.rows.length > 4 ? `… ${b.rows.length - 4} more rows` : ""]
    .filter(Boolean).join("<br>");

/** Text with its bound numbers: hover one to see the table, rows and step behind it. */
function Bound({ seg, tip }: { seg: Segment[]; tip: TipApi }) {
  return <>{seg.map((s, i) => {
    if ("text" in s) return <span key={i}>{s.text}</span>;
    if ("broken" in s) return <span key={i} className="bn broken" title={s.broken}>[number not found]</span>;
    return <span key={i} className="bn" tabIndex={0} onMouseMove={(e) => tip.show(behind(s.number), e)} onMouseLeave={tip.hide}>{s.number.display}</span>;
  })}</>;
}

const csv = (rows: Record<string, unknown>[]) => {
  const cols = Object.keys(rows[0] ?? {});
  const cell = (v: unknown) => (/[",\n]/.test(String(v ?? "")) ? `"${String(v).replace(/"/g, '""')}"` : String(v ?? ""));
  return [cols.join(","), ...rows.map((r) => cols.map((c) => cell(r[c])).join(","))].join("\n");
};

function Figure({ v, n, onDecide }: { v: View; n: number; onDecide: (keep: boolean) => void }) {
  const [wide, setWide] = useState(false);
  const [rowsOpen, setRowsOpen] = useState(false);
  const copy = async () => {
    try { await navigator.clipboard.writeText(csv(v.rows)); ui.toast(`Copied ${v.rows.length} rows as CSV`); }
    catch { ui.toast("Could not reach the clipboard"); }
  };
  return (
    <figure className={`rfig rp-fig${v.draft ? " is-draft" : ""}${wide ? " wide" : ""}`} id={`fig-${n}`}>
      <div className="rfig-h">
        <span><b>Figure {n}</b> · {v.spec.title}{v.draft && <span className="n-pill orange">draft</span>}</span>
        {v.draft
          ? <span className="row-btns"><button className="btn sm ink" onClick={() => onDecide(true)}>Keep</button><button className="btn sm plain" onClick={() => onDecide(false)}>Undo</button></span>
          : <span className="rp-acts">
              <button onClick={() => setWide(!wide)}>{wide ? "Narrow" : "Expand"}</button>
              <button onClick={() => setRowsOpen(!rowsOpen)} aria-expanded={rowsOpen}>Data</button>
              <button onClick={copy}>Copy CSV</button>
              <button onClick={() => { ui.lab({ view: "saved", fig: v.spec.id }); ui.open("lab"); }} title="Open this figure in the data lab's Saved diagrams, where the lab assistant can explain it">Open in lab</button>
              <button onClick={() => askInChat(`About Figure ${n} (${v.spec.title}): `)}>Ask</button>
            </span>}
      </div>
      <ViewChart view={v} />
      <figcaption><b>Figure {n}.</b> {v.spec.caption || v.spec.title}{" "}
        <span className="rp-src">From {v.spec.table}, {v.rows.length} row{v.rows.length === 1 ? "" : "s"}.</span></figcaption>
      {rowsOpen && (
        <div className="n-tblwrap rp-rows"><table className="n-tbl">
          <thead><tr>{Object.keys(v.rows[0] ?? {}).map((c) => <th key={c}>{c}</th>)}</tr></thead>
          <tbody>{v.rows.slice(0, 200).map((r, i) => <tr key={i}>{Object.values(r).map((x, j) => <td key={j}>{typeof x === "number" ? fmt(x, Number.isInteger(x) ? 0 : 3) : String(x)}</td>)}</tr>)}</tbody>
        </table></div>
      )}
    </figure>
  );
}

/** The section being read: the last one whose heading has passed the upper third of the screen. */
function useReading(ready: boolean) {
  const [at, setAt] = useState<string>("summary");
  useEffect(() => {
    if (!ready) return;
    const els = SECTIONS.map(([id]) => document.getElementById(`rp-${id}`)).filter(Boolean) as HTMLElement[];
    const io = new IntersectionObserver(() => {
      const line = window.innerHeight / 3;
      const passed = els.filter((el) => el.getBoundingClientRect().top <= line);
      setAt((passed.at(-1) ?? els[0])?.id.replace("rp-", "") ?? "summary");
    }, { threshold: [0, 0.25, 0.5, 0.75, 1], rootMargin: "0px 0px -60% 0px" });
    els.forEach((el) => io.observe(el));
    // the page scrolls inside the panel, not the window: listen on the nearest ancestor that scrolls
    let scroller: HTMLElement | Window = window;
    for (let n = els[0]?.parentElement; n; n = n.parentElement) {
      const oy = getComputedStyle(n).overflowY;
      if ((oy === "auto" || oy === "scroll") && n.scrollHeight > n.clientHeight) { scroller = n; break; }
    }
    const onScroll = () => {
      const line = window.innerHeight / 3;
      const passed = els.filter((el) => el.getBoundingClientRect().top <= line);
      setAt((passed.at(-1) ?? els[0])?.id.replace("rp-", "") ?? "summary");
    };
    scroller.addEventListener("scroll", onScroll, { passive: true });
    return () => { io.disconnect(); scroller.removeEventListener("scroll", onScroll); };
  }, [ready]);
  return at;
}

const SECTIONS = [["summary", "Summary"], ["figures", "Figures"], ["methods", "Methods"], ["gating", "Gating strategy"], ["data", "Data and code"]] as const;
const size = (b: number) => (b > 1048576 ? `${(b / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1024))} KB`);

export function Report({ s }: { s: ExperimentState }) {
  const qc = useQueryClient();
  const { data: rep, error } = useReport(s.id);
  const { data: views = [] } = useViews(s.id);
  const tip = useTip();
  const picked = useSelectionLabel();
  const reading = useReading(!!rep);
  const decide = async (keep: boolean) => {
    try { keep ? await api.keep(s.id) : await api.undo(s.id); ui.toast(keep ? "Kept" : "Dropped"); } catch (e) { ui.toast((e as Error).message); }
    await refresh(qc, s.id);
  };
  if (error) return <div className="ws-page-in"><p className="note">The results page could not be built: {(error as Error).message}</p></div>;
  if (!rep) return <div className="ws-page-in"><p className="note">Building the results page…</p></div>;
  const measured = rep.data.tables.length > 0;

  return (
    <div className="ws-page-in rp-wrap"><div className="rp-grid">
      <nav className="rp-outline" aria-label="Sections">
        {SECTIONS.map(([id, label]) => <a key={id} href={`#rp-${id}`} className={reading === id ? "on" : ""} aria-current={reading === id ? "true" : undefined}
          onClick={(e) => { e.preventDefault(); document.getElementById(`rp-${id}`)?.scrollIntoView({ behavior: "smooth" }); }}>{label}</a>)}
      </nav>
      <article className="report">
        <header>
          {rep.draft && (
            <div className="rp-draft" role="status"><span className="n-pill orange">draft</span> The agent wrote this page's words; nothing is kept until you say.
              <span className="row-btns"><button className="btn sm ink" onClick={() => decide(true)}>Keep</button><button className="btn sm plain" onClick={() => decide(false)}>Undo</button></span></div>
          )}
          <span className="kick">Results</span>
          <h1>{rep.title || "No findings written yet"}</h1>
          <span className="byline">{s.title} · {s.snapshot.samples.length} files · numbers from saved tables</span>
        </header>

        {!!rep.key_numbers.length && (
          <div className="rstats">
            {rep.key_numbers.map((k, i) => (
              <div key={i} className="rs"><b><Bound seg={[k]} tip={tip} /></b><span>{k.label}</span></div>
            ))}
          </div>
        )}

        <section id="rp-summary">
          <h2>Summary</h2>
          {rep.summary.length
            ? <p><Bound seg={rep.summary} tip={tip} /></p>
            : <div className="empty-in"><p>{measured
                ? "The agent writes the title and summary from the kept tables. Every number in them points at a table cell."
                : "Nothing is measured yet. Run the plan's measure step, then ask for the summary."}</p>
                <button className="btn sm" onClick={() => sendInChat(measured ? "Write the results summary" : "Run the plan")}>{measured ? "Write the summary" : "Run the plan"}</button></div>}
        </section>

        <section id="rp-figures">
          <h2>Figures</h2>
          {picked && views.length > 0 && (
            <div className="rp-pick" role="status">Picked out in every figure: <b>{picked}</b><button onClick={clearSelection}>Clear</button></div>
          )}
          {views.length
            ? views.map((v, i) => <Figure key={v.spec.id + (v.draft ? "-d" : "")} v={v} n={i + 1} onDecide={decide} />)
            : <div className="empty-in"><p>No figures yet. Say what you want to compare, and the agent draws it from a saved table.</p>
                <button className="btn sm" onClick={() => sendInChat("Draw a figure that answers the experiment's question")}>Ask for a figure</button></div>}
          {!!views.length && <button className="btn sm plain ask-more" onClick={() => sendInChat("Draw another figure for the question, one that shows something the kept figures do not")}>Ask for another figure</button>}
        </section>

        <section id="rp-methods">
          <h2>Methods</h2>
          {rep.methods.map((m) => <p key={m.heading} className="rp-method"><b>{m.heading}.</b> {m.text}</p>)}
        </section>

        <section id="rp-gating">
          <h2>Gating strategy</h2>
          <GatingStrategy s={s} />
        </section>

        <section id="rp-data">
          <h2>Data and code</h2>
          <div className="n-tblwrap"><table className="n-tbl rp-files">
            <thead><tr><th>File</th><th className="n-r">Size</th><th>sha256</th></tr></thead>
            <tbody>{rep.data.files.map((f) => <tr key={f.name}><td>{f.name}</td><td className="n-r">{size(f.bytes)}</td><td><code title={f.sha256}>{f.sha256.slice(0, 16)}…</code></td></tr>)}</tbody>
          </table></div>
          <p className="rp-small">
            {rep.data.tables.length
              ? <>Tables: {rep.data.tables.map((t, i) => <span key={t.name}>{i ? ", " : ""}<code>{t.name}</code> ({t.rows} rows{t.step ? `, from step ${t.step}` : ""})</span>)}. </>
              : "No saved tables yet. "}
            {rep.data.workflow_version ? <>Workflow version <code>{rep.data.workflow_version}</code>. </> : "No workflow kept yet. "}
            Software: {rep.data.software.map((x) => `${x.name} ${x.version} (${x.licence})`).join(", ")}.
          </p>
          <h3>References</h3>
          <ol className="rp-refs">{rep.references.map((r) => <li key={r.url}><a href={r.url} target="_blank" rel="noreferrer noopener">{r.text}</a></li>)}</ol>
        </section>
      </article>
      <Tip state={tip.state} />
    </div></div>
  );
}
