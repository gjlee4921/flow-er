// The data lab's own assistant, in a pane beside the lab. It has its own conversation, apart from
// the main chat, and only the tools that move the lab and read numbers: it lays out plots, lines
// files up, fills the sheet and says what it sees. It never changes the analysis.
import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api, type Message, type TraceItem } from "../../api";
import { apply, useLabMessages } from "../../api/hooks";
import { Flower, Icon } from "../../components/icons";
import { LIVE0, LiveTurn, nextLive, type Live } from "../../components/LiveTurn";
import { ToolTrace, type ToolKind } from "../../components/ToolTrace";
import { markdown, shortSample, ui } from "../../lib";

const SUGGEST = [
  "Show forward against side scatter for every file",
  "Which file has the most dead cells?",
  "Median CD4 in CD4+ T cells, file by file",
  "Histogram of CD3 in live single cells",
];

/** A tool call as a short line of what happened on screen. */
const did = (t: TraceItem) => {
  if (!t.ok) return `Could not ${t.tool.replace(/_/g, " ")}: ${t.summary.replace(/^failed: /, "")}`;
  if (t.tool === "gate_stats") return `Read the numbers (${t.summary.split(" ")[0]} rows)`;
  if (t.tool === "describe_gates") return "Read the gate tree";
  return t.summary.replace(/\S+\.fcs\b/g, (f) => shortSample(f));
};

// what each lab tool does: looks something up, or changes what is on screen
const MOVES = new Set(["show_plot", "compare_files", "select_gate", "show_sheet", "set_gate_tool", "show_figure"]);
const READS = new Set(["gate_stats", "describe_gates", "fcs_keywords", "read_figure"]);
const kindOf = (tool: string): ToolKind | null => (MOVES.has(tool) ? "move" : READS.has(tool) ? "read" : null);

function Turn({ m }: { m: Message }) {
  if (m.role === "user") return <div className="u"><span className="u-t">{m.text}</span></div>;
  return (
    <div className="a">
      <ToolTrace trace={m.trace} line={did} kind={kindOf} />
      <div className="md" dangerouslySetInnerHTML={{ __html: markdown(m.text || "") }} />
    </div>
  );
}

export default function LabAgent({ exp, screen }: { exp: string; screen: Record<string, unknown> }) {
  const qc = useQueryClient();
  const { data: msgs = [] } = useLabMessages(exp);
  const [text, setText] = useState("");
  const [pending, setPending] = useState<{ text: string; live: Live } | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  useEffect(() => { scroller.current?.scrollTo({ top: 1e9, behavior: "smooth" }); }, [msgs.length, pending]);
  useEffect(() => { input.current?.focus(); }, []);

  const send = async (t: string) => {
    t = t.trim();
    if (!t || pending) return;
    setText("");
    setPending({ text: t, live: LIVE0 });
    try {
      // streamed: the lab moves on screen the moment each action runs
      await api.labChatStream(exp, t, screen, (e) => {
        setPending((p) => p && { ...p, live: nextLive(p.live, e) });
        if (e.type === "ui") apply(e.events);
      });
    } catch (e) {
      ui.toast((e as Error).message);
    } finally {
      await qc.invalidateQueries({ queryKey: ["exp", exp, "lab-messages"] });
      setPending(null);
    }
  };

  return (
    <aside className="lab-ask" aria-label="Lab assistant">
      <div className="la-h">
        <Flower className="sm" /><b>Lab assistant</b><span>looks and lays out; changes nothing</span>
        <button className="ib sm" aria-label="Close the lab assistant" onClick={() => ui.lab({ ask: false })}><Icon.close /></button>
      </div>
      <div className="la-scroll" ref={scroller}>
        <div className="la-thread">
          {!msgs.length && !pending && (
            <div className="la-intro">
              <p>Ask for any view of the events: a file, two parameters, a population, every file side by side, or the numbers in a sheet.</p>
              <div className="suggest">{SUGGEST.map((s) => <button key={s} onClick={() => send(s)}>{s}</button>)}</div>
            </div>
          )}
          {msgs.map((m) => <Turn key={m.id} m={m} />)}
          {pending && <>
            <div className="u"><span className="u-t">{pending.text}</span></div>
            <LiveTurn live={pending.live} />
          </>}
        </div>
      </div>
      <form className="la-comp" onSubmit={(e) => { e.preventDefault(); send(text); }}>
        <textarea id="lab-ask-text" ref={input} rows={2} value={text} placeholder="Show me…" aria-label="Ask the lab assistant"
          onChange={(e) => setText(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(text); } }} />
        <button className="send" id="lab-ask-send" type="submit" disabled={!text.trim() || !!pending} aria-label="Send"><Icon.send /></button>
      </form>
    </aside>
  );
}
