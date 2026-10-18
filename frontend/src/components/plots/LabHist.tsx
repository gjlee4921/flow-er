// One parameter of the plotted population as a histogram: the share of drawn events in each bin,
// smoothed a little, on the same display scale as the 2-D plot (0..1 after the channel's transform).
// Range gates on this parameter show as shaded bands; with the range tool, drag across the levels
// to keep and a new range gate is drawn.
import { useRef, useState } from "react";
import type { GateNode, PlotData } from "../../api";
import { gateLabel } from "../../lib";

const BINS = 96;
const W = 600, PAD = { l: 44, r: 12, t: 14, b: 36 };

type Props = {
  data?: PlotData; xLabel: string; height?: number;
  gates?: GateNode[]; x?: string; tool?: string; selected?: string | null;
  onSelect?: (key: string) => void;
  onDraw?: (vertices: number[][], kind: "range") => void;
};

export default function LabHist({ data, xLabel, height = 380, gates = [], x, tool, selected, onSelect, onDraw }: Props) {
  const H = height, pw = W - PAD.l - PAD.r, ph = H - PAD.t - PAD.b;
  const svg = useRef<SVGSVGElement>(null);
  const [drag, setDrag] = useState<number[] | null>(null);
  const at = (e: React.PointerEvent) => {
    const r = svg.current!.getBoundingClientRect();
    return Math.max(0, Math.min(1, ((e.clientX - r.left) * (W / r.width) - PAD.l) / pw));
  };
  const ranges = gates.filter((g) => g.dims.length === 1 && g.dims[0] === x && g.vertices?.length === 2);
  if (!data?.x.length) return <svg className="lab-hist" viewBox={`0 0 ${W} ${H}`} aria-label="No events to draw" />;
  const counts = new Array(BINS).fill(0);
  for (const v of data.x) counts[Math.max(0, Math.min(BINS - 1, Math.floor(v * BINS)))]++;
  const smooth = counts.map((c, i) => 0.25 * (counts[i - 1] ?? c) + 0.5 * c + 0.25 * (counts[i + 1] ?? c));
  const pct = smooth.map((c) => (c / data.x.length) * 100);
  const top = Math.max(...pct) * 1.12 || 1;
  const step = [0.5, 1, 2, 2.5, 5, 10, 20].find((s) => top / s <= 4) ?? 25;
  const ticks = Array.from({ length: Math.floor(top / step) + 1 }, (_, i) => i * step);
  const X = (u: number) => PAD.l + u * pw;
  const Xi = (i: number) => PAD.l + (i / BINS) * pw;
  const Y = (v: number) => PAD.t + ph - (v / top) * ph;
  const line = pct.map((v, i) => `${i ? "L" : "M"}${Xi(i + 0.5).toFixed(1)},${Y(v).toFixed(1)}`).join("");
  const area = `${line}L${Xi(BINS - 0.5).toFixed(1)},${Y(0)}L${Xi(0.5).toFixed(1)},${Y(0)}Z`;
  const drawing = tool === "range" && !!onDraw;
  return (
    <svg ref={svg} className={`lab-hist${drawing ? " drawing" : ""}`} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Histogram of ${xLabel}`}
      onPointerDown={(e) => { if (!drawing) return; const u = at(e); setDrag([u, u]); (e.target as Element).setPointerCapture?.(e.pointerId); }}
      onPointerMove={(e) => { if (drag) setDrag([drag[0], at(e)]); }}
      onPointerUp={() => {
        if (!drag) return;
        const [lo, hi] = [Math.min(...drag), Math.max(...drag)];
        setDrag(null);
        if (hi - lo > 0.01) onDraw!([[+lo.toFixed(4)], [+hi.toFixed(4)]], "range");
      }}>
      {ticks.map((t) => (
        <g key={t}>
          <line className="grid" x1={PAD.l} x2={W - PAD.r} y1={Y(t)} y2={Y(t)} />
          <text className="tick" x={PAD.l - 6} y={Y(t) + 4} textAnchor="end">{t}%</text>
        </g>
      ))}
      {/* grid and values on the channel's round values (decades, or 0, 50K, 100K...) */}
      {(data.x_ticks?.length ? data.x_ticks : [0.25, 0.5, 0.75].map((v) => ({ v, label: "" }))).map((t) => (
        <g key={t.v}>
          {t.v > 0.005 && t.v < 0.995 && <line className="grid" x1={X(t.v)} x2={X(t.v)} y1={PAD.t} y2={PAD.t + ph} />}
          {t.label && <text className="tick" x={X(t.v)} y={PAD.t + ph + 14} textAnchor="middle">{t.label}</text>}
        </g>
      ))}
      {ranges.map((g) => {
        const [lo, hi] = [Math.min(g.vertices[0][0], g.vertices[1][0]), Math.max(g.vertices[0][0], g.vertices[1][0])];
        const on = g.key === selected;
        return (
          <g key={g.key} className={`h-range${on ? " on" : ""}${g.draft ? " draft" : ""}`} onClick={() => onSelect?.(g.key)}>
            <rect x={X(lo)} y={PAD.t} width={X(hi) - X(lo)} height={ph} />
            <text x={(X(lo) + X(hi)) / 2} y={PAD.t + 13} textAnchor="middle">{gateLabel(g.name)} · {g.pct_parent.toFixed(1)}%</text>
          </g>
        );
      })}
      <path className="h-area" d={area} />
      <path className="h-line" d={line} />
      {drag && <rect className="h-drag" x={X(Math.min(...drag))} y={PAD.t} width={Math.abs(X(drag[1]) - X(drag[0]))} height={ph} />}
      <text className="axis-t" x={PAD.l + pw / 2} y={H - 8} textAnchor="middle">{xLabel}</text>
      <text className="axis-t" x={12} y={PAD.t + ph / 2} textAnchor="middle" transform={`rotate(-90 12 ${PAD.t + ph / 2})`}>% of events drawn</text>
    </svg>
  );
}
