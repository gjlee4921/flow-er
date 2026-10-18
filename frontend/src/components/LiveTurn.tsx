// A turn while it runs: what the agent is doing now, and each step as it starts and finishes, so
// the wait shows the work (people wait more patiently for work they can see being done).
import type { TurnEvent } from "../api";
import { markdown } from "../lib";
import { Flower, Icon } from "./icons";

export type LiveStep = { i: number; label: string; done: boolean; ok: boolean; summary?: string };
export type Live = { phase: string; steps: LiveStep[]; text: string };
export const LIVE0: Live = { phase: "Reading your message", steps: [], text: "" };

/** The next state of a running turn after one of its events. */
export function nextLive(l: Live, e: TurnEvent): Live {
  // each round starts its own words; only the last round's are the answer
  if (e.type === "thinking") return { ...l, text: "", phase: e.round === 0 ? "Thinking" : "Thinking about what came back" };
  if (e.type === "text") return { ...l, phase: "Writing", text: l.text + e.delta };
  if (e.type === "plan_partial") return { ...l, phase: `Laying out the plan: ${e.steps.length} step${e.steps.length === 1 ? "" : "s"} so far` };
  if (e.type === "step") return { ...l, phase: e.label, steps: [...l.steps, { i: e.i, label: e.label, done: false, ok: false }] };
  if (e.type === "step_done") return { ...l, steps: l.steps.map((s) => (s.i === e.i ? { ...s, done: true, ok: e.ok, summary: e.summary } : s)) };
  // a long step saying what it is doing now (the script writer searching, reading, testing)
  if (e.type === "ui") {
    const p = [...e.events].reverse().find((u) => u.type === "progress");
    if (p && p.type === "progress") return { ...l, steps: l.steps.map((s, k) => (k === l.steps.length - 1 && !s.done ? { ...s, summary: p.text } : s)) };
  }
  return l;
}

export function LiveTurn({ live, compact = false }: { live: Live; compact?: boolean }) {
  return (
    <div className={`a live${compact ? " compact" : ""}`} aria-live="polite">
      <span className="thinking"><Flower /><span className="shimmer">{live.phase}…</span></span>
      {!!live.steps.length && (
        <ol className="live-steps">
          {live.steps.map((s) => (
            <li key={s.i} className={s.done ? (s.ok ? "done" : "bad") : "run"}>
              <span className="ls-ic">{s.done ? (s.ok ? <Icon.check /> : <Icon.close />) : <span className="ls-dot" />}</span>
              <span className="ls-l">{s.label}</span>
              {s.summary && s.summary !== "done" && !compact && <span className="ls-s">{s.summary}</span>}
            </li>
          ))}
        </ol>
      )}
      {live.text && <div className="md live-text" dangerouslySetInnerHTML={{ __html: markdown(live.text) }} />}
    </div>
  );
}
