// One selection shared by every figure: clicking a sample, a gate or an organ in one figure picks it out
// in all of them (the report's linked highlight). It is a label, compared without case, so a file on a
// bar chart's axis and the same file as a colour in another figure match, as do "Lung" and "lung".
import { useSyncExternalStore } from "react";

let current: string | null = null;   // compared form
let shown: string | null = null;     // as it was clicked, for saying what is picked
const listeners = new Set<() => void>();

export const selKey = (label: string) => label.trim().toLowerCase();

export function setSelection(label: string | null) {
  const next = label == null ? null : selKey(label);
  current = current === next ? null : next; // clicking the picked one again lets it go
  shown = current ? label : null;
  listeners.forEach((f) => f());
}

export function clearSelection() {
  current = null; shown = null;
  listeners.forEach((f) => f());
}

/** The picked label (lower case), or null; re-renders when it changes. */
export function useSelection(): string | null {
  return useSyncExternalStore((f) => { listeners.add(f); return () => listeners.delete(f); }, () => current);
}

/** The picked label as it was clicked ("Lung", "E01_008"), for saying what is picked. */
export function useSelectionLabel(): string | null {
  return useSyncExternalStore((f) => { listeners.add(f); return () => listeners.delete(f); }, () => shown);
}
