// The start page's composer, folded into a flower at the bottom of the screen. Hovering unfolds it
// into a bar; a click opens it fully and keeps it open; a click anywhere else (or Esc) folds it back.
// What was typed is kept while it is folded. "/" opens it from anywhere outside a text box.
import { useEffect, useRef } from "react";
import type { Dataset } from "../../api";
import { Bloom, Flower, Icon } from "../../components/icons";

export type DockState = "rest" | "peek" | "open";

type Props = {
  state: DockState; setState: (s: DockState) => void;
  text: string; setText: (t: string) => void;
  chosen: Dataset | null; setChosen: (d: Dataset | null) => void;
  datasets: Dataset[]; busy: boolean; onStart: () => void;
};

export default function Dock({ state, setState, text, setText, chosen, setChosen, datasets, busy, onStart }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const hovered = useRef(false);

  // open: a click outside folds it back, except on things that feed it (a dataset card)
  useEffect(() => {
    if (state !== "open") return;
    const away = (e: PointerEvent) => {
      const t = e.target as Element;
      if (!box.current?.contains(t) && !t.closest?.("[data-feeds-dock]")) setState("rest");
    };
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") setState("rest"); };
    document.addEventListener("pointerdown", away);
    window.addEventListener("keydown", esc);
    return () => { document.removeEventListener("pointerdown", away); window.removeEventListener("keydown", esc); };
  }, [state, setState]);
  useEffect(() => { if (state === "open") input.current?.focus(); }, [state, chosen]);
  useEffect(() => {
    const slash = (e: KeyboardEvent) => {
      if (e.key !== "/" || e.metaKey || e.ctrlKey || (e.target as Element)?.closest?.("textarea,input,select,[contenteditable]")) return;
      e.preventDefault();
      setState("open");
    };
    window.addEventListener("keydown", slash);
    return () => window.removeEventListener("keydown", slash);
  }, [setState]);

  const draft = text.trim();
  const hint = !chosen ? "Pick the data first" : draft ? "Start opens the project and plans the analysis" : "Start opens the project; add a sentence and it plans too";

  return (
    <div className={`dock ${state}`} ref={box}
      onMouseEnter={() => { hovered.current = true; if (state === "rest") setState("peek"); }}
      onMouseLeave={() => { hovered.current = false; if (state === "peek") setState("rest"); }}>
      {state !== "open" ? (
        <button className="dock-face" aria-label="Start an analysis (/)" aria-expanded={false}
          onClick={() => setState("open")}
          onFocus={() => { if (state === "rest") setState("peek"); }}
          onBlur={() => { if (state === "peek" && !hovered.current) setState("rest"); }}>
          <span className="dock-fl"><Bloom p={1} /></span>
          <span className="dock-ph">{draft || (chosen ? `What is ${chosen.label} for?` : "Ask flow-er, or start an analysis…")}</span>
          {chosen && <span className="dock-tag">{chosen.label}</span>}
          {draft && state === "rest" && <i className="dock-dot" aria-hidden="true" />}
        </button>
      ) : (
        <form className="dock-form" aria-label="Start an analysis" onSubmit={(e) => { e.preventDefault(); onStart(); }}>
          <div className="dock-top">
            <span className="dock-fl"><Bloom p={1} /></span>
            <div className="dock-data">
              {chosen ? (
                <span className="att st-data"><span className="att-k">Data</span><span className="att-n">{chosen.label}</span>
                  <span className="att-s">{chosen.files} files</span>
                  <button type="button" aria-label="Choose other data" onClick={() => setChosen(null)}>×</button></span>
              ) : datasets.length ? (
                <>
                  <span className="dock-k">Data</span>
                  {datasets.slice(0, 4).map((d) => (
                    <button key={d.name} type="button" className="dock-chip" onClick={() => setChosen(d)}>{d.label}</button>
                  ))}
                </>
              ) : (
                <span className="dock-k">Bring data on the dashboard first</span>
              )}
            </div>
            <button type="button" className="ib sm" aria-label="Fold away" onClick={() => setState("rest")}><Icon.close /></button>
          </div>
          <textarea id="start-text" ref={input} rows={2} value={text} onChange={(e) => setText(e.target.value)}
            placeholder="What is this experiment for? For example: how CD4 and CD8 T cells respond to stimulation"
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onStart(); } }} aria-label="What the experiment is for" />
          <div className="c-row">
            <span className="st-hint">{hint}</span>
            <button className="btn ink" id="start-go" type="submit" disabled={!chosen || busy}>
              {busy ? <Flower className="sm" /> : null}{draft ? "Start and plan" : "Start"}
            </button>
          </div>
        </form>
      )}
    </div>
  );
}
