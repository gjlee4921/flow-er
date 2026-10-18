// Shared helpers: formatting, the UI store, colours, markdown.
import type { Step } from "../api";
import { useSyncExternalStore } from "react";
import { marked } from "marked";

/** "today", "yesterday", "3 days ago", else the date. */
export function ago(iso?: string | null) {
  if (!iso) return "";
  const days = Math.floor((Date.now() - new Date(iso).getTime()) / 86_400_000);
  return days <= 0 ? "today" : days === 1 ? "yesterday" : days < 14 ? `${days} days ago` : iso.slice(0, 10);
}

/** Who is using flow-er on this machine (no accounts yet): their name draws their flower. */
export const ME = { name: "Will" };

export const fmt = (n: number, d = 0) =>
  Number(n).toLocaleString("en-GB", { minimumFractionDigits: d, maximumFractionDigits: d });
export const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));
export const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
export const esc = (s: unknown) =>
  String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]!));
/** A gate name as a scientist says it. FlowJo names a gate it made from two channels "X, Y subset";
 *  those become what they pick out ("Live cells" on a viability dye, "Time QC" on Time). A "(model)"
 *  ending is kept. The FlowJo name stays on hover wherever this is shown. */
export const gateLabel = (name: string) => {
  const base = name.replace(/ \((model(?: \d+)?)\)$/, ""), end = name.slice(base.length);
  const auto = /, .+ subset$/i.test(base);
  const out = /live[_/ ]?dead|viabil|zombie|7-?aad|fixable/i.test(base) ? "Live cells"
    : auto && /\btime\b/i.test(base) ? "Time QC"
    : base.replace(/^Comp-/, "");
  return out + end;
};
/** "101_DEN084Y5_15_E03_009_clean.fcs" -> "E03_009", "HC Specimen_001_C5_C05 Peptide.fcs" -> "HC C05 Peptide":
 *  enough to tell files apart, keeping the group when files from two groups share well names. */
export const shortSample = (id: string) => {
  const m = id.match(/([A-Za-z]\d{2}_\d{3})/);
  if (m) return m[1];
  const base = id.replace(/\.fcs$/i, "");
  const w = base.match(/^(\S+)\s+Specimen_\d+_\w*?([A-P]\d{2})(?:\s+(.*))?$/i);
  if (w) return [w[1], w[2], w[3]].filter(Boolean).join(" ");
  return base.length > 18 ? `…${base.slice(-17)}` : base;
};

// The agent writes Markdown. Raw HTML in it is shown as text, never run.
marked.use({ renderer: { html: (token: { text?: string; raw?: string }) => esc(token.text ?? token.raw ?? "") } });
// a table column of numbers lines up on the right; a column of words stays on the left and wraps
const NUMERIC = /^[\s\d.,%+\-−–()×≈<>~]*\d[\s\d.,%+\-−–()×≈<>~pP]*$/;
marked.use({
  walkTokens(token) {
    if (token.type !== "table") return;
    type Cell = { text: string; align: string | null };
    const t = token as unknown as { align: (string | null)[]; header: Cell[]; rows: Cell[][] };
    t.align.forEach((a, c) => {
      const al = a ?? (t.rows.length && t.rows.every((r) => !r[c]?.text.trim() || NUMERIC.test(r[c].text)) ? "right" : "left");
      for (const cell of [t.header[c], ...t.rows.map((r) => r[c])]) if (cell) cell.align = al;
    });
  },
});
export const markdown = (text: string) => marked.parse(text, { async: false, gfm: true, breaks: false }) as string;

// Colours for populations, in the order children are listed.
export const POP = ["#30d158", "#0a84ff", "#ff9f0a", "#bf5af2", "#ff375f", "#64d2ff", "#ac8e68", "#5e5ce6"];
// neighbours differ in hue, so two groups (ALS and HC, say) never read as one
export const SERIES = ["#0a84ff", "#ff9f0a", "#2fb7c9", "#bf5af2", "#30d158", "#ff375f", "#5e5ce6", "#8e8e93"];

// ---- a small store for what the screen shows ------------------------------------------------

export type DockState = "rest" | "peek" | "open";
export type Page = "none" | "flow" | "lab" | "results" | "measure" | "models" | "clean" | "data";
export type LabState = {
  sample?: string; parent: string | null; x?: string; y?: string;
  stage: "raw" | "processed" | "labels"; mode: "density" | "dots" | "hist" | "contour"; tool: "select" | "rect" | "poly" | "quadrant" | "range";
  sel: string | null; view: "plot" | "table" | "saved";
  /** the saved diagram open in Saved diagrams (none = the gallery); set from outside to land on one */
  fig?: string | null;
  /** every file side by side on the same axes and population */
  compare: boolean;
  /** watching the lab's models gate the files as a run does it (the data lab opens on this by itself) */
  live: boolean;
  /** the lab whose models are shown learning, round by round (demo), or null */
  rounds?: string | null;
  /** the sheet: which gates (none = all), which toolkit statistics as columns ("stat:parameter:arg"),
   *  a text filter, the sort, a column coloured by value and a column charted */
  sheetGates: string[]; sheetSpec: string[]; sheetFilter: string; sheetSort: { key: string; desc: boolean } | null;
  sheetColour: string | null; sheetChart: string | null;
  /** the lab assistant's pane */
  ask: boolean;
};
export type UiState = {
  exp: string | null; page: Page; panelOpen: boolean; full: boolean; railOpen: boolean;
  lab: LabState; prevPage: Page; toast: string | null;
  /** what the agent is doing right now, while a turn runs; null when it is not working */
  agentBusy: string | null;
  /** the plan as the model writes it, step by step, before it lands as a draft */
  planPreview: Step[] | null;
  /** a first message to send as soon as a new project opens (from the start page) */
  autoSend: string | null;
  /** the panel listing what flow-er remembers about the scientist */
  memoryOpen: boolean;
  /** the settings panel (opened from the profile button at the top of the sidebar) */
  settingsOpen: boolean;
  /** the start page's composer: folded into a flower, unfolded on hover, or open */
  dock: DockState;
  /** the workflow run being followed live: each step's status as it goes */
  run: { exp: string; id: string; status: string; nodes: Record<string, { status: string; summary?: string; title?: string; op?: string }> } | null;
};

const LAB0: LabState = {
  parent: null, stage: "processed", mode: "density", tool: "select", sel: null, view: "plot", fig: null,
  compare: false, sheetGates: [], sheetSpec: [], sheetFilter: "", sheetSort: null, sheetColour: null, sheetChart: null, ask: false, live: false, rounds: null,
};
const saved = (() => { try { return localStorage.getItem("flower.exp"); } catch { return null; } })();
// each project has its own link (…/#<project id>): opening or pasting one opens that project. A reload
// keeps to the project remembered last, as before (a script that sets it and reloads lands there)
const reloaded = (() => {
  try { return (performance.getEntriesByType("navigation")[0] as PerformanceNavigationTiming | undefined)?.type === "reload"; } catch { return false; }
})();
const linked = !reloaded && /^#\d{8}-[0-9a-f]{6}$/.test(location.hash) ? location.hash.slice(1) : null;
/** The project opened most recently, for the start page's Continue card. */
export const lastProject = () => { try { return localStorage.getItem("flower.last"); } catch { return null; } };
let state: UiState = {
  exp: linked ?? saved, page: "none", panelOpen: true, full: false, railOpen: true,
  lab: { ...LAB0 },
  prevPage: "flow", toast: null, agentBusy: null, planPreview: null, autoSend: null, memoryOpen: false, settingsOpen: false, dock: "rest", run: null,
};
const listeners = new Set<() => void>();
/** Where you are in a project: the page, the panel and the data lab. */
export type Place = Pick<UiState, "page" | "panelOpen" | "full" | "lab" | "prevPage">;
const where: Record<string, Place> = {};
let toastTimer = 0;

export const ui = {
  get: () => state,
  set(patch: Partial<UiState> | ((s: UiState) => Partial<UiState>)) {
    const p = typeof patch === "function" ? patch(state) : patch;
    state = { ...state, ...p };
    if ("exp" in p) {
      try {
        p.exp ? localStorage.setItem("flower.exp", p.exp) : localStorage.removeItem("flower.exp");
        if (p.exp) localStorage.setItem("flower.last", p.exp);
      } catch { /* private window */ }
      const want = p.exp ? `#${p.exp}` : "";
      if (location.hash !== want) history.replaceState(null, "", want || location.pathname + location.search);
    }
    listeners.forEach((l) => l());
  },
  /** Open a project (null for the start page). Each project keeps where you were in it (page, panel,
   *  data lab) for this session, so switching back returns you there; a new one starts fresh. */
  openProject(exp: string | null, leaving?: Place) {
    const s = ui.get();
    if (s.exp) where[s.exp] = leaving ?? { page: s.page, panelOpen: s.panelOpen, full: s.full, lab: s.lab, prevPage: s.prevPage };
    const back = exp ? where[exp] : undefined;
    ui.set(back ? { exp, ...back } : { exp, page: "none", full: false, lab: { ...LAB0 } });
  },
  lab(patch: Partial<LabState>) { ui.set((s) => ({ lab: { ...s.lab, ...patch } })); },
  /** Show a page in the panel. The lab takes the whole screen and remembers where it came from. */
  open(page: Page) {
    ui.set((s) => ({
      page, panelOpen: true,
      prevPage: page === "lab" ? (s.page === "lab" ? s.prevPage : s.page === "none" ? "flow" : s.page) : s.prevPage,
      full: page === "lab" ? true : s.page === "lab" ? false : s.full,
    }));
  },
  leaveLab() { const s = ui.get(); ui.set({ page: s.prevPage, full: false, lab: { ...s.lab, live: false, rounds: null } }); },
  toast(text: string) {
    ui.set({ toast: text });
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => ui.set({ toast: null }), 2800);
  },
};

export function useUi<T>(pick: (s: UiState) => T): T {
  return useSyncExternalStore((l) => { listeners.add(l); return () => listeners.delete(l); }, () => pick(state));
}

if (state.exp && location.hash !== `#${state.exp}`) history.replaceState(null, "", `#${state.exp}`);
if (state.exp) { try { localStorage.setItem("flower.last", state.exp); } catch { /* private window */ } }
window.addEventListener("hashchange", () => {
  const id = /^#\d{8}-[0-9a-f]{6}$/.test(location.hash) ? location.hash.slice(1) : null;
  if (id && id !== ui.get().exp) ui.openProject(id);
});
