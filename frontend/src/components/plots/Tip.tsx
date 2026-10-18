// One tooltip, positioned near the pointer. Content is HTML built from our own strings.
import { useCallback, useState } from "react";

type TipState = { html: string; x: number; y: number } | null;

export function useTip() {
  const [state, setState] = useState<TipState>(null);
  const show = useCallback((html: string, e: { clientX: number; clientY: number }) => setState({ html, x: e.clientX, y: e.clientY }), []);
  const hide = useCallback(() => setState(null), []);
  return { state, show, hide };
}

export function Tip({ state }: { state: TipState }) {
  if (!state) return null;
  const left = Math.min(window.innerWidth - 240, Math.max(8, state.x + 14));
  const top = Math.max(8, state.y - 12);
  return <div className="tip" style={{ left, top, position: "fixed" }} dangerouslySetInnerHTML={{ __html: state.html }} />;
}
