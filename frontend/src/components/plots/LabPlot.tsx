// The data lab's plot: events of one population on two channels, the gates beneath it, and the
// tools to draw a new one. Everything is in display space (0..1 after each channel's transform),
// the same space FlowJo stores gate vertices in, so gates sit exactly where they were drawn.
import { useEffect, useRef, useState } from "react";
import type { GateNode, PlotData } from "../../api";
import { css, POP } from "../../lib";

type Props = {
  data?: PlotData;
  gates: GateNode[];          // the children of the plotted population, with counts
  x: string; y: string; xLabel: string; yLabel: string;
  mode: "density" | "dots" | "contour"; stage: string; tool: "select" | "rect" | "poly" | "quadrant" | "range";
  selected: string | null;
  onSelect: (key: string) => void;
  onEnter: (key: string) => void;
  onDraw: (vertices: number[][], kind: "rect" | "polygon") => void;
  /** a quadrant: where its four quarters meet */
  onQuadrant?: (centre: number[]) => void;
  busy?: boolean;
};

const PAD = { l: 44, r: 12, t: 12, b: 36 };
const DENS = [[214, 228, 246], [146, 184, 232], [74, 134, 214], [36, 92, 186], [122, 86, 170], [214, 120, 58], [228, 92, 40]];
export function densCol(u: number) {
  const t = Math.pow(Math.max(0, Math.min(1, u)), 0.55) * (DENS.length - 1);
  const i = Math.min(DENS.length - 2, Math.floor(t)), f = t - i;
  const c = DENS[i].map((v, k) => Math.round(v + (DENS[i + 1][k] - v) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

/** The events binned on a B × B grid and smoothed twice: what density and contour plots draw. */
function grid(d: { x: number[]; y: number[] }, B: number) {
  const bin = new Float32Array(B * B);
  for (let i = 0; i < d.x.length; i++) {
    const bx = Math.max(0, Math.min(B - 1, Math.floor(d.x[i] * B))), by = Math.max(0, Math.min(B - 1, Math.floor(d.y[i] * B)));
    bin[by * B + bx]++;
  }
  const blur = (src: Float32Array) => {
    const out = new Float32Array(B * B);
    for (let yy = 0; yy < B; yy++) for (let xx = 0; xx < B; xx++) {
      let t = 0, n = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const a = yy + dy, b = xx + dx;
        if (a < 0 || a >= B || b < 0 || b >= B) continue;
        const wt = dx && dy ? 0.6 : dx || dy ? 1 : 1.7; t += src[a * B + b] * wt; n += wt;
      }
      out[yy * B + xx] = t / n;
    }
    return out;
  };
  return blur(blur(bin));
}

/** Probability contours, as FlowJo draws them: each line encloses a share of the events (here the
 *  densest 50, 75, 90, 95 and 98%), traced through the grid by marching squares. */
function contours(sm: Float32Array, B: number): { level: number; segs: number[][] }[] {
  const sorted = Array.from(sm).sort((a, b) => b - a), total = sorted.reduce((a, v) => a + v, 0);
  const at = (share: number) => { let acc = 0; for (const v of sorted) { acc += v; if (acc >= share * total) return v; } return 0; };
  const f = (i: number, j: number) => sm[j * B + i];
  return [0.98, 0.95, 0.9, 0.75, 0.5].map((share) => {
    const t = at(share), segs: number[][] = [];
    for (let j = 0; j < B - 1; j++) for (let i = 0; i < B - 1; i++) {
      const a = f(i, j), b = f(i + 1, j), c = f(i + 1, j + 1), d = f(i, j + 1);
      const k = (a > t ? 1 : 0) | (b > t ? 2 : 0) | (c > t ? 4 : 0) | (d > t ? 8 : 0);
      if (k === 0 || k === 15) continue;
      const lerp = (p: number, q: number) => (t - p) / (q - p || 1e-9);
      const e = [[i + lerp(a, b), j], [i + 1, j + lerp(b, c)], [i + lerp(d, c), j + 1], [i, j + lerp(a, d)]]; // bottom, right, top, left
      const pairs: Record<number, number[][]> = { 1: [[3, 0]], 2: [[0, 1]], 3: [[3, 1]], 4: [[1, 2]], 5: [[3, 2], [0, 1]], 6: [[0, 2]], 7: [[3, 2]],
        8: [[2, 3]], 9: [[0, 2]], 10: [[0, 3], [1, 2]], 11: [[1, 2]], 12: [[1, 3]], 13: [[0, 1]], 14: [[0, 3]] };
      for (const [u, v] of pairs[k]) segs.push([(e[u][0] + 0.5) / B, (e[u][1] + 0.5) / B, (e[v][0] + 0.5) / B, (e[v][1] + 0.5) / B]);
    }
    return { level: share, segs };
  });
}

/** Vertices of a gate in this plot's x, y order, or null when it is drawn on other channels. */
function onAxes(g: { dims: string[]; vertices: number[][] }, x: string, y: string) {
  if (!g.vertices?.length || g.dims.length !== 2) return null;
  if (g.dims[0] === x && g.dims[1] === y) return g.vertices;
  if (g.dims[0] === y && g.dims[1] === x) return g.vertices.map(([a, b]) => [b, a]);
  return null;
}

function inPoly(px: number, py: number, poly: number[][]) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export default function LabPlot(p: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const box = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [draft, setDraft] = useState<number[][]>([]);
  const [hover, setHover] = useState<number[] | null>(null);
  const drag = useRef<number[] | null>(null);

  useEffect(() => {
    const el = box.current!;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  useEffect(() => { setDraft([]); setHover(null); }, [p.tool, p.x, p.y, p.data?.parent]);
  useEffect(() => {
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") { setDraft([]); drag.current = null; } };
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, []);

  // the plot keeps close to square so populations are not stretched
  let w = size.w, h = Math.max(220, size.h);
  if (w > h * 1.4) w = Math.round(h * 1.4);
  if (h > w * 1.15) h = Math.round(w * 1.15);
  const pw = w - PAD.l - PAD.r, ph = h - PAD.t - PAD.b;
  const X = (u: number) => PAD.l + Math.max(0, Math.min(1, u)) * pw;
  const Y = (u: number) => PAD.t + (1 - Math.max(0, Math.min(1, u))) * ph;
  const unit = (e: React.PointerEvent) => {
    const r = ref.current!.getBoundingClientRect();
    return [(e.clientX - r.left - PAD.l) / pw, 1 - (e.clientY - r.top - PAD.t) / ph];
  };

  useEffect(() => {
    const cv = ref.current;
    if (!cv || !w) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = h * dpr;
    const ctx = cv.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = css("--plot") || "#fff"; ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = css("--sep"); ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      ctx.beginPath(); ctx.moveTo(X(i / 4), PAD.t); ctx.lineTo(X(i / 4), PAD.t + ph);
      ctx.moveTo(PAD.l, Y(i / 4)); ctx.lineTo(PAD.l + pw, Y(i / 4)); ctx.stroke();
    }
    const d = p.data;
    if (d && d.x.length) {
      if (p.mode === "contour" && p.stage !== "labels") {
        // the events faintly behind, then the lines from the outermost (98%) to the densest (50%)
        ctx.fillStyle = "rgba(120,122,132,.16)";
        for (let i = 0; i < d.x.length; i++) ctx.fillRect(X(d.x[i]) - 0.8, Y(d.y[i]) - 0.8, 1.6, 1.6);
        const lines = contours(grid(d, 64), 64);
        lines.forEach(({ segs }, k) => {
          ctx.strokeStyle = densCol(0.25 + (k / (lines.length - 1)) * 0.75); ctx.lineWidth = 1.2 + k * 0.15;
          ctx.beginPath();
          for (const [x1, y1, x2, y2] of segs) { ctx.moveTo(X(x1), Y(y1)); ctx.lineTo(X(x2), Y(y2)); }
          ctx.stroke();
        });
      } else if (p.mode === "density" && p.stage !== "labels") {
        const B = 72;
        const sm = grid(d, B);
        const nz = Array.from(sm).filter((v) => v > 0.02).sort((a, b) => a - b);
        const mx = nz.length ? nz[Math.floor(nz.length * 0.99)] : 1;
        const cw = pw / B + 1, chh = ph / B + 1;
        for (let yy = 0; yy < B; yy++) for (let xx = 0; xx < B; xx++) {
          const v = sm[yy * B + xx]; if (v < 0.035) continue;
          ctx.fillStyle = densCol(v / mx); ctx.fillRect(X(xx / B), Y((yy + 1) / B), cw, chh);
        }
      } else {
        for (let i = 0; i < d.x.length; i++) {
          const lab = d.label[i];
          ctx.fillStyle = p.stage === "labels" && lab >= 0 ? POP[lab % POP.length] + "cc" : "rgba(120,122,132,.45)";
          ctx.fillRect(X(d.x[i]) - 1.2, Y(d.y[i]) - 1.2, 2.4, 2.4);
        }
      }
    }
    // gates below this population
    p.gates.forEach((g, i) => {
      const v = onAxes(g, p.x, p.y);
      if (!v) return;
      const on = g.key === p.selected;
      ctx.beginPath(); v.forEach(([a, b], k) => (k ? ctx.lineTo(X(a), Y(b)) : ctx.moveTo(X(a), Y(b)))); ctx.closePath();
      ctx.fillStyle = on ? "rgba(0,113,227,.08)" : "transparent"; ctx.fill();
      ctx.strokeStyle = g.draft ? css("--c-flag") : on ? css("--c-trt") : (p.stage === "labels" ? POP[i % POP.length] : css("--text-2"));
      ctx.lineWidth = on ? 2 : 1.3; ctx.setLineDash(g.draft ? [5, 4] : []); ctx.stroke(); ctx.setLineDash([]);
      const top = v.reduce((m, q) => (q[1] > m[1] ? q : m), v[0]);
      ctx.fillStyle = ctx.strokeStyle as string; ctx.font = `600 11px ${css("--font")}`; ctx.textAlign = "left";
      ctx.fillText(`${g.name} · ${g.pct_parent.toFixed(1)}%`, Math.min(X(top[0]), PAD.l + pw - 110), Math.max(PAD.t + 11, Y(top[1]) - 5));
    });
    // a quadrant being placed: a crosshair through the pointer
    if (p.tool === "quadrant" && hover) {
      ctx.strokeStyle = css("--c-trt"); ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(X(hover[0]), PAD.t); ctx.lineTo(X(hover[0]), PAD.t + ph); ctx.moveTo(PAD.l, Y(hover[1])); ctx.lineTo(PAD.l + pw, Y(hover[1])); ctx.stroke(); ctx.setLineDash([]);
    }
    // a gate being drawn
    const pts = hover && draft.length ? [...draft, hover] : draft;
    if (pts.length) {
      ctx.strokeStyle = css("--c-trt"); ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
      ctx.beginPath(); pts.forEach(([a, b], k) => (k ? ctx.lineTo(X(a), Y(b)) : ctx.moveTo(X(a), Y(b))));
      if (p.tool === "rect") ctx.closePath();
      ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.strokeStyle = css("--sep-2"); ctx.strokeRect(PAD.l + 0.5, PAD.t + 0.5, pw - 1, ph - 1);
    ctx.fillStyle = css("--text-3"); ctx.font = `11px ${css("--font")}`; ctx.textAlign = "center";
    ctx.fillText(p.xLabel, PAD.l + pw / 2, h - 10);
    ctx.save(); ctx.translate(14, PAD.t + ph / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(p.yLabel, 0, 0); ctx.restore();
  });

  const finish = (pts: number[][], kind: "rect" | "polygon") => {
    setDraft([]); setHover(null);
    const xs = pts.map((q) => q[0]), ys = pts.map((q) => q[1]);
    if (Math.max(...xs) - Math.min(...xs) < 0.02 || Math.max(...ys) - Math.min(...ys) < 0.02) return;
    p.onDraw(pts.map(([a, b]) => [+a.toFixed(4), +b.toFixed(4)]), kind);
  };

  return (
    <div className="lab-plot" ref={box}>
      <canvas
        ref={ref} id="lab-cv" style={{ width: w, height: h, left: Math.round((size.w - w) / 2), top: Math.round((Math.max(220, size.h) - h) / 2) }}
        aria-label={`Events on ${p.xLabel} against ${p.yLabel}`}
        onPointerDown={(e) => {
          const [a, b] = unit(e);
          if (a < 0 || a > 1 || b < 0 || b > 1) return;
          if (p.tool === "quadrant") { p.onQuadrant?.([+a.toFixed(4), +b.toFixed(4)]); setHover(null); return; }
          if (p.tool === "rect") { drag.current = [a, b]; (e.target as Element).setPointerCapture(e.pointerId); }
          else if (p.tool === "poly") {
            if (draft.length > 2 && Math.hypot(a - draft[0][0], b - draft[0][1]) < 0.035) return finish(draft, "polygon");
            setDraft([...draft, [a, b]]);
          } else {
            const hit = p.gates.find((g) => { const v = onAxes(g, p.x, p.y); return v && inPoly(a, b, v); });
            if (hit) p.onSelect(hit.key);
          }
        }}
        onPointerMove={(e) => {
          const [a, b] = unit(e);
          if (p.tool === "rect" && drag.current) { const [x0, y0] = drag.current; setDraft([[x0, y0], [a, y0], [a, b], [x0, b]]); }
          else if (p.tool === "poly" && draft.length) setHover([a, b]);
          else if (p.tool === "quadrant") setHover(a >= 0 && a <= 1 && b >= 0 && b <= 1 ? [a, b] : null);
        }}
        onPointerUp={(e) => {
          if (p.tool !== "rect" || !drag.current) return;
          // built from where the drag started and ended, not from the last drawn frame
          const [x0, y0] = drag.current, [a, b] = unit(e);
          drag.current = null;
          finish([[x0, y0], [a, y0], [a, b], [x0, b]].map(([u, v]) => [Math.max(0, Math.min(1, u)), Math.max(0, Math.min(1, v))]), "rect");
        }}
        onDoubleClick={(e) => {
          if (p.tool === "poly" && draft.length > 2) return finish(draft, "polygon");
          const [a, b] = unit(e as unknown as React.PointerEvent);
          const hit = p.gates.find((g) => { const v = onAxes(g, p.x, p.y); return v && inPoly(a, b, v); });
          if (hit) p.onEnter(hit.key);
        }}
      />
      {p.busy && <div className="lab-busy" />}
    </div>
  );
}
