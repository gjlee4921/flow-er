// The gating strategy as a figure: one panel per step down the gate tree, each showing the parent
// population on the axes its gates were drawn on, with the gates outlined and their % of parent. The
// panels are the data lab's own plot, read-only, so the figure and the lab always agree; clicking a
// gate opens the lab at the same file, population, axes and gate.
import { useEffect, useMemo, useState } from "react";
import { useQueries } from "@tanstack/react-query";
import { api, type ExperimentState, type GateNode } from "../../api";
import { useGates, useSamples } from "../../api/hooks";
import { LabPlot } from "../../components/plots";
import { shortSample, ui } from "../../lib";

type Panel = { parent: string | null; x: string; y: string; gates: GateNode[] };
const LETTERS = "abcdefghijklmnopqrstuvwxyz";

/** Gates drawn on two channels, grouped by parent and axes, in the tree's order. */
function panelsOf(gates: GateNode[]): { panels: Panel[]; skipped: string[] } {
  const panels: Panel[] = [], skipped: string[] = [];
  for (const g of gates) {
    if (g.dims.length !== 2) { skipped.push(g.name); continue; }
    const p = panels.find((q) => q.parent === g.parent && q.x === g.dims[0] && q.y === g.dims[1]);
    if (p) p.gates.push(g);
    else panels.push({ parent: g.parent, x: g.dims[0], y: g.dims[1], gates: [g] });
  }
  return { panels, skipped };
}

/** True once the element has been on screen: the panels load only when someone scrolls to them. */
function useSeen<T extends Element>() {
  // a callback ref, so the watch starts whenever the figure appears (it is not there on the first render)
  const [el, setEl] = useState<T | null>(null);
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    if (!el || seen) return;
    const io = new IntersectionObserver(([e]) => { if (e.isIntersecting) setSeen(true); }, { rootMargin: "200px" });
    io.observe(el);
    return () => io.disconnect();
  }, [el, seen]);
  return [setEl, seen] as const;
}

export function GatingStrategy({ s }: { s: ExperimentState }) {
  const { data: meta } = useSamples(s.id);
  const samples = meta?.samples ?? [];
  const channels = meta?.channels ?? [];
  const [picked, setPicked] = useState<string | null>(null);
  const sid = picked ?? samples[0]?.id;
  const { data: tree } = useGates(s.id, sid);
  const gates = tree?.gates ?? [];
  const { panels, skipped } = useMemo(() => panelsOf(gates), [gates]);
  const [ref, seen] = useSeen<HTMLElement>();
  const plots = useQueries({
    queries: panels.map((p) => ({
      queryKey: ["exp", s.id, "plot", sid, p.x, p.y, p.parent, "processed"],
      queryFn: () => api.plot(s.id, { sample: sid, x: p.x, y: p.y, parent: p.parent, stage: "processed" }),
      enabled: seen && !!sid, staleTime: 60_000,
    })),
  });
  const name = (id: string) => channels.find((c) => c.id === id)?.name ?? id;
  const gateName = (key: string | null) => (key ? gates.find((g) => g.key === key)?.name ?? key.split("/").pop() : "All events");
  const openInLab = (p: Panel, key: string | null) => {
    ui.lab({ view: "plot", sample: sid, parent: p.parent, x: p.x, y: p.y, sel: key, mode: "density", stage: "processed" });
    ui.open("lab");
  };

  if (!samples.length) return null;
  if (!panels.length) return <div className="empty-in"><p>No gates yet. Gates come from a FlowJo workspace, from drawing them in the data lab, or from the gating models.</p></div>;
  const sources = [...new Set(gates.map((g) => g.source))].map((x) => ({ flowjo: "the FlowJo workspace", user: "the scientist", model: "the gating models" })[x] ?? x);

  return (
    <figure className="rfig rp-gs" ref={ref}>
      <div className="rfig-h">
        <span><b>Figure S1</b> · Gating strategy, {shortSample(sid ?? "")}</span>
        {samples.length > 1 && (
          <select className="rp-gs-pick" value={sid} onChange={(e) => setPicked(e.target.value)} aria-label="File shown">
            {samples.map((x) => <option key={x.id} value={x.id}>{shortSample(x.id)}</option>)}
          </select>
        )}
      </div>
      <div className="rp-gs-grid">
        {panels.map((p, i) => (
          <div key={`${p.parent}|${p.x}|${p.y}`} className="rp-gs-panel">
            <div className="rp-gs-t">
              <b>{LETTERS[i] ?? i + 1}</b> {gateName(p.parent)}
              <span>{p.gates.map((g) => `${g.name} ${g.pct_parent.toFixed(1)}%`).join(" · ")}</span>
            </div>
            <LabPlot data={plots[i]?.data} gates={p.gates} x={p.x} y={p.y} xLabel={name(p.x)} yLabel={name(p.y)}
              mode="density" stage="processed" tool="select" selected={null} busy={seen && plots[i]?.isFetching}
              onSelect={(k) => openInLab(p, k)} onEnter={(k) => openInLab(p, k)} onDraw={() => {}} />
          </div>
        ))}
      </div>
      <figcaption><b>Figure S1.</b> Each panel shows the parent population of {shortSample(sid ?? "")} on the channels its
        gates were drawn on; gates are outlined with their % of parent. Gates from {sources.join(" and ")}. Click a gate to
        open it in the data lab.{skipped.length ? ` Gates on one channel (${skipped.join(", ")}) are listed in the methods.` : ""}</figcaption>
    </figure>
  );
}
