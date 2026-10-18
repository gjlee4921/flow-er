// A thin handle on a panel's edge. Drag it to change the panel's width, nudge it with the arrow
// keys, double-click to put it back. Widths live in CSS variables and are kept between visits.
import type { KeyboardEvent, PointerEvent } from "react";

const KEY = "flower.sizes";
type Sizes = Record<string, number>;
const load = (): Sizes => { try { return JSON.parse(localStorage.getItem(KEY) || "{}"); } catch { return {}; } };
const save = (s: Sizes) => { try { localStorage.setItem(KEY, JSON.stringify(s)); } catch { /* private window */ } };
const root = document.documentElement;

// put back the widths from the last visit before anything is drawn
for (const [name, px] of Object.entries(load())) root.style.setProperty(name, `${px}px`);

type Props = {
  /** The CSS variable that holds the panel's width, e.g. "--railw". */
  cssVar: string;
  /** Which edge of the screen the panel sits against: a left panel grows as you drag right. */
  side: "left" | "right";
  /** The panel's width now, in pixels. */
  measure: () => number;
  min: number;
  max: () => number;
  label: string;
  className: string;
};

export function Resizer({ cssVar, side, measure, min, max, label, className }: Props) {
  const set = (px: number) => {
    const w = Math.round(Math.max(min, Math.min(max(), px)));
    root.style.setProperty(cssVar, `${w}px`);
    return w;
  };
  const keep = (w: number) => save({ ...load(), [cssVar]: w });

  const down = (e: PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    const handle = e.currentTarget, x0 = e.clientX, w0 = measure(), ws = document.getElementById("ws");
    let w = w0;
    handle.setPointerCapture(e.pointerId);
    handle.classList.add("on");
    ws?.classList.add("rsz-on");
    const move = (m: globalThis.PointerEvent) => { w = set(w0 + (side === "left" ? 1 : -1) * (m.clientX - x0)); };
    const up = () => {
      handle.removeEventListener("pointermove", move);
      handle.classList.remove("on");
      ws?.classList.remove("rsz-on");
      keep(w);
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", up, { once: true });
    handle.addEventListener("pointercancel", up, { once: true });
  };
  const key = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const grow = (e.key === "ArrowRight") === (side === "left");
    keep(set(measure() + (grow ? 16 : -16)));
  };
  const reset = () => {
    root.style.removeProperty(cssVar);
    const s = load();
    delete s[cssVar];
    save(s);
  };

  return (
    <div className={`rsz ${className}`} role="separator" aria-orientation="vertical" aria-label={label} tabIndex={0}
      title={`${label}: drag, or double-click to reset`} onPointerDown={down} onKeyDown={key} onDoubleClick={reset} />
  );
}

/** Width of the first element matching a selector, for `measure`. */
export const widthOf = (selector: string) => () => document.querySelector(selector)?.getBoundingClientRect().width ?? 0;
