// The conversation: the scientist asks in plain words, the agent answers, calls tools, and leaves
// one change waiting to be kept or undone. Files go in with the + or by dropping them here.
import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type ExperimentState, type Message, type UiEvent } from "../../api";
import { apply, refresh, useExperiment, useMessages } from "../../api/hooks";
import { Icon } from "../../components/icons";
import { LIVE0, LiveTurn, nextLive, type Live } from "../../components/LiveTurn";
import { markdown, ui, useUi } from "../../lib";
import { runLine } from "../pipeline/Pipeline";

const KIND = (name: string) => {
  const ext = (name.split(".").pop() || "").toLowerCase();
  return ({ fcs: "FCS", wsp: "FlowJo", csv: "Table", tsv: "Table", xlsx: "Table", pdf: "Doc", docx: "Doc", md: "Doc", txt: "Doc", tif: "Image", tiff: "Image", png: "Image", zip: "Zip" } as Record<string, string>)[ext] ?? "File";
};
const size = (n: number) => (n > 1048576 ? `${(n / 1048576).toFixed(1)} MB` : n > 1024 ? `${Math.round(n / 1024)} KB` : `${n} B`);

function suggestions(s?: ExperimentState): string[] {
  if (!s) return [];
  const d = s.draft;
  if (d) return [`Keep it`, `Undo that`, d.kind === "plan" ? "Why these steps?" : "Explain what this shows"];
  if (!s.plan.length) return ["Plan the analysis", "What is in these files?", "Show me the gates on the first sample"];
  const ran = new Set(s.runs.filter((r) => r.status === "done").map((r) => r.step));
  if (s.plan.some((p) => !ran.has(p.id))) return ["Run the plan", "Open the pipeline", "What does each step do?"];
  if (!s.views.length) return ["What did the run find?", "Draw the results as a chart", "Open the data lab on the T cells"];
  return ["What stands out?", "Draw it another way", "Open the data lab on the T cells"];
}

function Trace({ m }: { m: Message }) {
  const [open, setOpen] = useState(false);
  if (!m.trace?.length) return null;
  return (
    <div className={`act${open ? " open" : ""}`}>
      <button type="button" aria-expanded={open} onClick={() => setOpen(!open)}>
        <Icon.eye />Used {m.trace.length} tool{m.trace.length === 1 ? "" : "s"}<Icon.caret className="cv" />
      </button>
      {open && (
        <ul className="calls-list">
          {m.trace.map((t, i) => (
            <li key={i} className={t.ok ? "" : "bad"}><code>{t.tool}</code> {t.summary}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** What the agent kept or forgot in this reply, with Undo for a new memory. */
function MemoryChip({ e }: { e: Extract<UiEvent, { type: "memory" }> }) {
  const qc = useQueryClient();
  const [undone, setUndone] = useState(false);
  const undo = async () => {
    try { await api.forgetMemory(e.id); setUndone(true); qc.invalidateQueries({ queryKey: ["memory"] }); ui.toast("Forgotten."); }
    catch (err) { ui.toast((err as Error).message); }
  };
  return (
    <div className={`memo${undone ? " undone" : ""}`} role="status">
      <span className="memo-k">{e.action === "added" ? (undone ? "Forgotten" : "Remembered") : e.action === "already" ? "Already remembered" : "Forgot"}</span>
      <span className="memo-t">{e.text}</span>
      {e.action === "added" && !undone && <button className="btn sm plain" onClick={undo}>Undo</button>}
      {e.action === "added" && !undone && <button className="btn sm plain" onClick={() => ui.set({ memoryOpen: true })}>See all</button>}
    </div>
  );
}

/** A workflow run the agent started: its steps as dots that fill in as they finish, then one line on
 *  how it went. Opens the workflow canvas. */
function RunCard({ exp, id }: { exp: string; id: string }) {
  const live = useUi((u) => (u.run?.id === id ? u.run : null));
  const { data: rec } = useQuery({ queryKey: ["exp", exp, "execution", id], queryFn: () => api.execution(exp, id), enabled: !live || live.status !== "running" });
  const r = live?.status === "running" || !rec ? live : rec;
  if (!r) return null;
  const nodes = Object.entries(r.nodes);
  const running = r.status === "running";
  const finished = nodes.filter(([, n]) => ["done", "cached", "failed", "blocked"].includes(n.status)).length;
  return (
    <div className={`runcard ${r.status}`} role="status">
      <span className="rc-k">Workflow run</span>
      <span className="rc-dots">{nodes.map(([k, n]) => <i key={k} className={`d ${n.status}`} title={`${n.title ?? k}: ${n.status === "cached" ? "reused" : n.status}`} />)}</span>
      <span className="rc-t">{running ? `Running, ${finished} of ${nodes.length} steps finished` : runLine(r)}</span>
      <button className="btn sm plain" onClick={() => ui.open("flow")}>Open the workflow</button>
    </div>
  );
}

const host = (url: string) => { try { return new URL(url).hostname.replace(/^www\./, ""); } catch { return ""; } };

/** The pages a web search found, numbered as the answer cites them. */
function Sources({ m }: { m: Message }) {
  if (!m.sources?.length) return null;
  return (
    <ol className="sources" aria-label="Sources">
      {m.sources.map((s) => (
        <li key={s.n} value={s.n}>
          <a href={s.url} target="_blank" rel="noreferrer noopener">{s.title}</a>
          <span className="src-host">{host(s.url)}</span>
        </li>
      ))}
    </ol>
  );
}

function Bubble({ m, exp }: { m: Message; exp?: string }) {
  if (m.role === "user") {
    return (
      <div className={`u${m.attachments?.length ? " u-att" : ""}`}>
        {m.attachments?.map((a) => <span key={a} className="att sent"><span className="att-k">{KIND(a)}</span><span className="att-n">{a}</span></span>)}
        {m.text && <span className="u-t">{m.text}</span>}
      </div>
    );
  }
  const memos = (m.ui ?? []).filter((e): e is Extract<UiEvent, { type: "memory" }> => e.type === "memory");
  const runs = (m.ui ?? []).filter((e): e is Extract<UiEvent, { type: "execution" }> => e.type === "execution");
  return (
    <div className="a">
      <Trace m={m} />
      <div className="md" dangerouslySetInnerHTML={{ __html: markdown(m.text || "") }} />
      <Sources m={m} />
      {memos.map((e) => <MemoryChip key={e.id + e.action} e={e} />)}
      {exp && runs.map((e) => <RunCard key={e.id} exp={exp} id={e.id} />)}
      {m.error && <p className="small muted">Detail: {m.error}</p>}
    </div>
  );
}

export default function Chat({ exp }: { exp: string }) {
  const qc = useQueryClient();
  const { data: msgs = [] } = useMessages(exp);
  const { data: state } = useExperiment(exp);
  const [text, setText] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [pending, setPending] = useState<{ text: string; attachments: string[]; live: Live } | null>(null);
  const [over, setOver] = useState(false);
  const scroller = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => { scroller.current?.scrollTo({ top: 1e9, behavior: "smooth" }); }, [msgs.length, pending]);
  // a project started from the start page with a question: ask it straight away
  useEffect(() => {
    const first = ui.get().autoSend;
    if (first) { ui.set({ autoSend: null }); send(first); }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const focus = (e: Event) => { const t = (e as CustomEvent<string>).detail; if (t) setText(t); input.current?.focus(); };
    // a button whose ask is complete sends it straight away, as a suggestion does
    const go = (e: Event) => { const t = (e as CustomEvent<string>).detail; if (t) sendNow.current(t); };
    window.addEventListener("flower:ask", focus);
    window.addEventListener("flower:send", go);
    return () => { window.removeEventListener("flower:ask", focus); window.removeEventListener("flower:send", go); };
  }, []);

  const send = async (raw: string) => {
    const words = raw.trim();
    if ((!words && !files.length) || pending) return;
    const attach = files;
    setText(""); setFiles([]);
    setPending({ text: words, attachments: attach.map((f) => f.name), live: LIVE0 });
    try {
      let names: string[] = [];
      if (attach.length) names = (await api.upload(exp, attach)).saved.map((s) => s.name);
      // the turn streams: each step shows as it runs, and what it changes appears straight away
      let live = LIVE0;
      await api.chatStream(exp, words || "I added these files.", names, (e) => {
        live = nextLive(live, e);
        setPending((p) => p && { ...p, live });
        ui.set({ agentBusy: live.phase });
        if (e.type === "plan_partial") {
          // the plan builds itself out on the canvas as the model writes it
          if (!ui.get().planPreview) ui.set({ panelOpen: true, page: ui.get().page === "lab" ? "lab" : "flow" });
          ui.set({ planPreview: e.steps });
        }
        if (e.type === "ui") {
          apply(e.events);
          // everything but the conversation, which already holds this turn's question
          if (e.events.some((x) => x.type === "refresh")) qc.invalidateQueries({ queryKey: ["exp", exp], predicate: (q) => q.queryKey[2] !== "messages" });
        }
      });
    } catch (e) {
      ui.toast(`The agent could not answer: ${(e as Error).message}`);
    } finally {
      await refresh(qc, exp);
      ui.set({ agentBusy: null, planPreview: null });
      setPending(null);
    }
  };
  const sendNow = useRef(send);
  sendNow.current = send;

  const decide = async (keep: boolean) => {
    try { keep ? await api.keep(exp) : await api.undo(exp); ui.toast(keep ? "Kept" : "Dropped"); }
    catch (e) { ui.toast((e as Error).message); }
    await refresh(qc, exp);
  };

  const chips = useMemo(() => suggestions(state), [state]);
  const draft = state?.draft;

  return (
    <main className="ws-chat">
      <div className="scroll" id="chat-scroll" ref={scroller}>
        <div className="thread" id="thread">
          {!msgs.length && !pending && (
            <div className="a intro">
              <p><b>{state?.title ?? "Your experiment"}</b> is open: {state?.snapshot.samples.length ?? "…"} files
                {state?.snapshot.flowjo_workspace ? <>, with the FlowJo workspace <code>{state.snapshot.flowjo_workspace}</code></> : null}.</p>
              <p>Tell me what the experiment is for and I will plan the analysis. Everything I plan, run and draw appears on the right, and nothing changes until you keep it.</p>
            </div>
          )}
          {msgs.map((m) => <Bubble key={m.id} m={m} exp={exp} />)}
          {pending && <>
            <Bubble m={{ id: "p", t: "", role: "user", text: pending.text, attachments: pending.attachments }} />
            <LiveTurn live={pending.live} />
          </>}
        </div>
      </div>
      <div className="bottom"><div className="bottom-in">
        {draft && !pending && (
          <div className="draft-bar" role="status">
            <span className="db-k">Waiting for you</span>
            <span className="db-s">{draft.summary}</span>
            <button className="btn sm ink" onClick={() => decide(true)}>Keep</button>
            <button className="btn sm plain" onClick={() => decide(false)}>Undo</button>
          </div>
        )}
        {!pending && <div className="suggest" id="chat-sug">{chips.map((c) => <button key={c} onClick={() => send(c)}>{c}</button>)}</div>}
        <form
          className={`composer${over ? " over" : ""}`} id="chat-form" autoComplete="off"
          onSubmit={(e) => { e.preventDefault(); send(text); }}
          onDragOver={(e) => { if (e.dataTransfer.types.includes("Files")) { e.preventDefault(); setOver(true); } }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); setFiles([...files, ...Array.from(e.dataTransfer.files)]); }}
        >
          {!!files.length && (
            <div className="c-atts" id="chat-atts">
              {files.map((f, i) => (
                <span key={i} className="att"><span className="att-k">{KIND(f.name)}</span><span className="att-n" title={f.name}>{f.name}</span>
                  <span className="att-s">{size(f.size)}</span>
                  <button type="button" aria-label={`Remove ${f.name}`} onClick={() => setFiles(files.filter((_, j) => j !== i))}>×</button></span>
              ))}
            </div>
          )}
          <textarea
            id="chat-text" ref={input} rows={1} value={text} placeholder="Ask, or tell me what to change" aria-label="Message"
            onChange={(e) => { setText(e.target.value); e.target.style.height = "auto"; e.target.style.height = Math.min(160, e.target.scrollHeight) + "px"; }}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(text); } }}
          />
          <div className="c-row">
            <div className="c-l">
              <button type="button" className="c-plus" id="chat-plus" aria-label="Add files" title="Add files: data, a protocol, a FlowJo workspace" onClick={() => fileInput.current?.click()}><Icon.plus /></button>
              <input ref={fileInput} type="file" multiple hidden onChange={(e) => { setFiles([...files, ...Array.from(e.target.files ?? [])]); e.target.value = ""; }} />
            </div>
            <button className="send" id="chat-send" type="submit" disabled={(!text.trim() && !files.length) || !!pending} aria-label="Send"><Icon.send /></button>
          </div>
        </form>
        <p className="foot-note">The agent proposes; you keep or undo. Every number comes from a saved table.</p>
      </div></div>
    </main>
  );
}

/** Put words in the composer from anywhere on the screen, for an ask the scientist finishes ("Add a step that "). */
export const askInChat = (text = "") => window.dispatchEvent(new CustomEvent("flower:ask", { detail: text }));
/** Send a complete ask from anywhere on the screen, without a trip to the send button. */
export const sendInChat = (text: string) => window.dispatchEvent(new CustomEvent("flower:send", { detail: text }));
