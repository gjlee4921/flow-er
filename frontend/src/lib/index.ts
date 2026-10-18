// Shared helpers: formatting, the UI store, colours, markdown.
import type { Step } from "../api";
import { useSyncExternalStore } from "react";
import { marked } from "marked";

export const fmt = (n: number, d = 0) =>
  Number(n).toLocaleString("en-GB", { minimumFractionDigits: d, maximumFractionDigits: d });
export const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));
export const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
export const esc = (s: unknown) =>
  String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]!));
/** "101_DEN084Y5_15_E03_009_clean.fcs" -> "E03_009": enough to tell files apart. */
export const shortSample = (id: string) => {
  const m = id.match(/([A-Za-z]\d{2}_\d{3})/);
  return m ? m[1] : id.replace(/\.fcs$/i, "").slice(-14);
};

// The agent writes Markdown. Raw HTML in it is shown as text, never run.
marked.use({ renderer: { html: (token: { text?: string; raw?: string }) => esc(token.text ?? token.raw ?? "") } });
export const markdown = (text: string) => marked.parse(text, { async: false, gfm: true, breaks: false }) as string;

// Colours for populations, in the order children are listed.
export const POP = ["#30d158", "#0a84ff", "#ff9f0a", "#bf5af2", "#ff375f", "#64d2ff", "#ac8e68", "#5e5ce6"];
export const SERIES = ["#0a84ff", "#5e5ce6", "#2fb7c9", "#ff9f0a", "#30d158", "#bf5af2", "#ff375f", "#8e8e93"];

// ---- a small store for what the screen shows ------------------------------------------------

export type Page = "none" | "flow" | "lab" | "results" | "measure" | "models" | "clean" | "data";
export type LabState = {
  sample?: string; parent: string | null; x?: string; y?: string;
  stage: "raw" | "processed" | "labels"; mode: "density" | "dots" | "hist" | "contour"; tool: "select" | "rect" | "poly" | "quadrant" | "range";
  sel: string | null; view: "plot" | "table" | "saved";
  /** the saved diagram open in Saved diagrams (none = the gallery); set from outside to land on one */
  fig?: string | null;
  /** every file side by side on the same axes and population */
  compare: boolean;
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
  /** the workflow run being followed live: each step's status as it goes */
  run: { exp: string; id: string; status: string; nodes: Record<string, { status: string; summary?: string; title?: string; op?: string }> } | null;
};

const LAB0: LabState = {
  parent: null, stage: "processed", mode: "density", tool: "select", sel: null, view: "plot", fig: null,
  compare: false, sheetGates: [], sheetSpec: [], sheetFilter: "", sheetSort: null, sheetColour: null, sheetChart: null, ask: false,
};
const saved = (() => { try { return localStorage.getItem("flower.exp"); } catch { return null; } })();
let state: UiState = {
  exp: saved, page: "none", panelOpen: true, full: false, railOpen: true,
  lab: { ...LAB0 },
  prevPage: "flow", toast: null, agentBusy: null, planPreview: null, autoSend: null, memoryOpen: false, run: null,
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
      try { p.exp ? localStorage.setItem("flower.exp", p.exp) : localStorage.removeItem("flower.exp"); } catch { /* private window */ }
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
  leaveLab() { const s = ui.get(); ui.set({ page: s.prevPage, full: false }); },
  toast(text: string) {
    ui.set({ toast: text });
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => ui.set({ toast: null }), 2800);
  },
};

export function useUi<T>(pick: (s: UiState) => T): T {
  return useSyncExternalStore((l) => { listeners.add(l); return () => listeners.delete(l); }, () => pick(state));
}
