// A figure-kit view: each organ placed where it sits on a 2-D mouse, its circle sized by the value and,
// when the view compares two conditions, coloured by the fold change from the baseline (naive or
// control) to the other. The mouse is the first mockup's own drawing. Organs it cannot place are
// listed under the figure, never dropped silently. Hover shows every animal; click picks an organ out.
import { useMemo, useState } from "react";
import type { View } from "../../api";
import { esc, fmt } from "../../lib";
import { selKey, setSelection, useSelection } from "./selection";
import { Tip, useTip } from "./Tip";

const W = 640, H = 320;
// where each organ sits in the side view (head left), and the names people use for it
const ORGANS: { id: string; name: string; at: [number, number]; names: RegExp }[] = [
  { id: "brain", name: "Brain", at: [118, 160], names: /^brain|cns/ },
  { id: "thymus", name: "Thymus", at: [186, 182], names: /^thymus/ },
  { id: "lung", name: "Lung", at: [212, 156], names: /^lungs?$|^lung\b/ },
  { id: "heart", name: "Heart", at: [258, 178], names: /^heart/ },
  { id: "blood", name: "Blood", at: [236, 206], names: /^(peripheral )?blood|^pbmc/ },
  { id: "liver", name: "Liver", at: [296, 210], names: /^liver/ },
  { id: "spleen", name: "Spleen", at: [348, 176], names: /^spleen/ },
  { id: "kidney", name: "Kidney", at: [394, 164], names: /^kidneys?/ },
  { id: "gut", name: "Gut", at: [388, 222], names: /^(gut|intestine|small intestine|colon|ileum|siel|lamina propria)/ },
  { id: "mln", name: "Mesenteric node", at: [350, 236], names: /^(mesenteric|mln)/ },
  { id: "dln", name: "Draining node", at: [446, 232], names: /^(draining|dln|tdln|inguinal|lymph node|ln$)/ },
  { id: "marrow", name: "Bone marrow", at: [474, 252], names: /^(bone marrow|marrow|bm$)/ },
  { id: "tumour", name: "Tumour", at: [482, 166], names: /^(tumou?r|primary|flank)/ },
  { id: "skin", name: "Skin", at: [430, 118], names: /^skin|^ear skin/ },
];
const BASELINE = /naive|control|ctrl|baseline|untreated|healthy|young|wild|^wt|pre/i;
const place = (organ: string) => ORGANS.find((o) => o.names.test(organ.trim().toLowerCase()));
const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
const sd = (v: number[]) => (v.length > 1 ? Math.sqrt(v.reduce((a, b) => a + (b - mean(v)) ** 2, 0) / (v.length - 1)) : 0);
const nice = (v: number) => (Math.abs(v) >= 100 ? fmt(v) : fmt(v, Math.abs(v) >= 1 ? 1 : 3));
// fold change to colour: a clear rise, a small one, no real change, a fall; grey where there is
// nothing to compare with (no baseline value for that organ)
const NONE = "var(--text-3)";
const foldCol = (f: number | null, compared = true) =>
  !compared ? "var(--c-trt)" : f == null ? NONE : f >= 2 ? "var(--c-trt)" : f >= 1.25 ? "color-mix(in srgb, var(--c-trt) 55%, var(--c-mid))" : f > 0.8 ? "var(--c-mid)" : "var(--c-flag)";

export default function AnatomyMap({ view }: { view: View }) {
  const { spec, rows } = view;
  const tip = useTip();
  const hot = useSelection(); // shared: an organ picked in any figure is picked out here too

  const { groups, base, by, unplaced } = useMemo(() => {
    const groups = spec.color ? [...new Set(rows.map((r) => String(r[spec.color!])))] : ["_"];
    const base = groups.length > 1 ? (groups.find((g) => BASELINE.test(g)) ?? groups[0]) : null;
    const by = new Map<string, { organ: string; label: string; vals: Map<string, number[]> }>();
    const unplaced = new Set<string>();
    for (const r of rows) {
      const organ = String(r[spec.x ?? ""] ?? ""), o = place(organ), v = Number(r[spec.y!]);
      if (!o) { if (organ) unplaced.add(organ); continue; }
      if (!Number.isFinite(v)) continue;
      const e = by.get(o.id) ?? { organ: o.id, label: organ, vals: new Map() };
      const g = spec.color ? String(r[spec.color]) : "_";
      e.vals.set(g, [...(e.vals.get(g) ?? []), v]);
      by.set(o.id, e);
    }
    return { groups, base, by, unplaced: [...unplaced] };
  }, [rows, spec.x, spec.y, spec.color]);
  const others = groups.filter((g) => g !== base);
  const [shownPick, setShown] = useState<string | null>(null);
  const shown = shownPick && others.includes(shownPick) ? shownPick : others[0];

  if (!rows.length) return <p className="note">The table has no rows for this view yet.</p>;
  const items = ORGANS.filter((o) => by.get(o.id)?.vals.get(shown)?.length).map((o) => {
    const e = by.get(o.id)!, vals = e.vals.get(shown)!, m = mean(vals);
    const bv = base ? e.vals.get(base) : undefined, bm = bv?.length ? mean(bv) : null;
    return { o, label: e.label, vals, m, bm, bvals: bv ?? [], fold: bm ? m / bm : null };
  });
  const max = Math.max(1e-9, ...items.map((i) => i.m));
  const radius = (v: number) => 9 + 21 * Math.sqrt(v / max);
  const ylab = spec.y_label || spec.y || "value";

  // labels in two rows, above and below the animal, spread so they never overlap
  const lay = (row: typeof items, y: number) => {
    const out = new Map<string, number>();
    let last = -1e9;
    row.slice().sort((a, b) => a.o.at[0] - b.o.at[0]).forEach((i) => { const x = Math.max(i.o.at[0], last + 92); out.set(i.o.id, x); last = x; });
    const over = Math.max(0, last - (W - 56));
    for (const [k, x] of out) out.set(k, Math.max(56, x - over));
    return { y, at: out };
  };
  const top = lay(items.filter((i) => i.o.at[1] < 196), 46), bottom = lay(items.filter((i) => i.o.at[1] >= 196), 300);
  const tipFor = (i: (typeof items)[number]) => [
    `<b>${esc(i.label)}</b>`,
    `${esc(spec.color ? shown : "")}${spec.color ? ": " : ""}<b>${nice(i.m)}</b> ${esc(ylab)}${i.vals.length > 1 ? ` <span class=tip-m>± ${nice(sd(i.vals))} SD, n = ${i.vals.length}</span>` : ""}`,
    i.bm != null ? `${esc(base!)}: ${nice(i.bm)} · <b>${i.fold!.toFixed(1)}×</b>` : base ? `no ${esc(base)} value` : "",
    i.vals.length > 1 ? `<span class=tip-m>each animal: ${i.vals.map(nice).join(" · ")}</span>` : "",
  ].filter(Boolean).join("<br>");
  const blood = items.find((i) => i.o.id === "blood");
  const keys = (i: (typeof items)[number]) => [selKey(i.label), selKey(i.o.name), i.o.id];
  const has = hot != null && items.some((i) => keys(i).includes(hot));
  const dim = (i: (typeof items)[number]) => (!has ? "" : keys(i).includes(hot!) ? " on" : " dim");

  return (
    <div className="vchart">
      {others.length > 1 && (
        <div className="seg" role="group" aria-label="Condition shown">
          {others.map((g) => <button key={g} aria-pressed={g === shown} onClick={() => setShown(g)}>{g}</button>)}
        </div>
      )}
      <svg className="chart map2" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={spec.title}>
        <defs><radialGradient id="mp-body" cx=".45" cy=".4" r=".7"><stop offset="0" stopColor="var(--plot)" /><stop offset="1" stopColor="var(--fill)" /></radialGradient></defs>
        <path className="body" d="M66 196C66 176 94 154 130 144C164 114 236 102 336 106C436 110 516 142 526 198C534 238 492 262 424 266L222 266C172 266 142 252 130 236C102 234 74 218 66 196Z" />
        <ellipse className="body ear" cx="168" cy="124" rx="24" ry="28" /><circle className="eye" cx="110" cy="176" r="3.2" />
        <path className="tail" d="M524 210C566 214 600 196 612 166C620 146 618 128 606 116" />
        {[[214, 268], [298, 270], [420, 270], [470, 266]].map(([x, y]) => <ellipse key={x} className="body foot" cx={x} cy={y} rx="16" ry="6" />)}
        {blood && items.filter((i) => i.o.id !== "blood").map((i) => {
          const [hx, hy] = blood.o.at, [x, y] = i.o.at;
          return <path key={`r-${i.o.id}`} className={`route${dim(i)}`} d={`M${hx} ${hy}Q${(hx + x) / 2} ${Math.min(hy, y) - 30} ${x} ${y}`}
            stroke={foldCol(i.fold, !!base)} strokeWidth={(1 + 3 * Math.sqrt(i.m / max)).toFixed(1)} />;
        })}
        {items.map((i) => {
          const [x, y] = i.o.at, r = radius(i.m), up = i.o.at[1] < 196, row = up ? top : bottom, lx = row.at.get(i.o.id)!, col = foldCol(i.fold, !!base);
          return (
            <g key={i.o.id} className={`mp-org${dim(i)}`} tabIndex={0} role="button" aria-label={`${i.label}: ${nice(i.m)}`}
              onMouseMove={(e) => tip.show(tipFor(i), e)} onMouseLeave={tip.hide} onClick={() => setSelection(i.label)}>
              <line className="lead" x1={x} y1={up ? y - r : y + r} x2={lx} y2={up ? row.y + 20 : row.y - 26} />
              {i.fold != null && i.fold >= 2.5 && <circle className="halo" cx={x} cy={y} r={r + 7} fill={col} />}
              <circle className="node" cx={x} cy={y} r={r.toFixed(1)} fill={col} />
              <text className="mp-n" x={lx} y={up ? row.y : row.y - 12} textAnchor="middle">{i.label}</text>
              <text className="mp-v" x={lx} y={up ? row.y + 15 : row.y + 3} textAnchor="middle" fill={i.fold != null ? col : "var(--text-2)"}>
                {i.fold != null ? `${i.fold.toFixed(1)}× · ` : ""}{nice(i.m)}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="legend">
        {base ? <>
          <span><i style={{ background: "var(--c-trt)" }} />2× or more over {base}</span>
          <span><i style={{ background: "var(--c-mid)" }} />no real change</span>
          <span><i style={{ background: "var(--c-flag)" }} />lower than {base}</span>
          {items.some((i) => i.fold == null) && <span><i style={{ background: NONE }} />no {base} value</span>}
        </> : null}
        <span className="muted">circle size: {ylab}{spec.color ? ` (${shown})` : ""}</span>
      </div>
      {!!unplaced.length && <p className="note">Not placed on the mouse: {unplaced.join(", ")}.</p>}
      <Tip state={tip.state} />
    </div>
  );
}
