// A step's density image, as the gate models see it (101 × 101, rows = y): a heat map on the
// lab's colour scale, log-scaled the way the models normalise it, with any regions found on it
// outlined and numbered.
import { useEffect, useRef } from "react";
import { css, POP } from "../../lib";
import { densCol } from "./LabPlot";

export function DensityImage({ grid, regions, size = 236, x, y }: { grid: number[][]; regions?: number[][]; size?: number; x?: string; y?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c || !grid.length) return;
    const ny = grid.length, nx = grid[0].length, dpr = window.devicePixelRatio || 1;
    c.width = size * dpr; c.height = size * dpr;
    const ctx = c.getContext("2d")!;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = css("--plot") || "#fff"; ctx.fillRect(0, 0, size, size);
    const cw = size / nx, ch = size / ny;
    const mx = Math.max(...grid.flat().map((v) => Math.log1p(v))) || 1;
    // row 0 is the lowest y, so it is drawn at the bottom
    for (let r = 0; r < ny; r++) for (let k = 0; k < nx; k++) {
      const v = grid[r][k];
      if (!v) continue;
      ctx.fillStyle = densCol(Math.log1p(v) / mx);
      ctx.fillRect(k * cw, size - (r + 1) * ch, cw + 0.5, ch + 0.5);
    }
    if (!regions?.length) return;
    // each region's edge: a line wherever a pixel's neighbour belongs to another region
    const lab = (r: number, k: number) => (r < 0 || k < 0 || r >= ny || k >= nx ? 0 : Math.round(regions[r][k]));
    ctx.lineWidth = 1.4;
    const sums: Record<number, [number, number, number]> = {};
    for (let r = 0; r < ny; r++) for (let k = 0; k < nx; k++) {
      const a = lab(r, k);
      if (!a) continue;
      const s = (sums[a] ??= [0, 0, 0]); s[0] += k; s[1] += r; s[2] += 1;
      ctx.strokeStyle = POP[(a - 1) % POP.length];
      const X = k * cw, Y = size - (r + 1) * ch;
      ctx.beginPath();
      if (lab(r, k + 1) !== a) { ctx.moveTo(X + cw, Y); ctx.lineTo(X + cw, Y + ch); }
      if (lab(r, k - 1) !== a) { ctx.moveTo(X, Y); ctx.lineTo(X, Y + ch); }
      if (lab(r + 1, k) !== a) { ctx.moveTo(X, Y); ctx.lineTo(X + cw, Y); }
      if (lab(r - 1, k) !== a) { ctx.moveTo(X, Y + ch); ctx.lineTo(X + cw, Y + ch); }
      ctx.stroke();
    }
    ctx.font = `600 11px ${css("--font")}`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (const [a, [sx, sy, n]] of Object.entries(sums)) {
      const X = (sx / n + 0.5) * cw, Y = size - (sy / n + 0.5) * ch;
      ctx.fillStyle = css("--raised") || "#fff"; ctx.beginPath(); ctx.arc(X, Y, 8, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = POP[(Number(a) - 1) % POP.length]; ctx.fillText(a, X, Y + 0.5);
    }
  }, [grid, regions, size]);
  return (
    <figure className="dimg">
      <canvas ref={ref} style={{ width: size, height: size }} aria-label={`Density image${regions ? " with regions" : ""}`} />
      {(x || y) && <figcaption><span>{x}</span><span className="dimg-y">{y}</span></figcaption>}
    </figure>
  );
}
