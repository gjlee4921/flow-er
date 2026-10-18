// Line icons and the flower, as small components. Paths are drawn on a 24-unit grid.

type P = { className?: string };
const I = ({ d, className, fill }: { d: string; className?: string; fill?: boolean }) => (
  <svg className={`i ${className ?? ""}`} viewBox="0 0 24 24" aria-hidden="true">
    <path d={d} {...(fill ? { fill: "currentColor", stroke: "none" } : {})} />
  </svg>
);

export const Icon = {
  plus: (p: P) => <I {...p} d="M12 5v14M5 12h14" />,
  send: (p: P) => <I {...p} d="M12 19V6M6 11l6-6 6 6" />,
  close: (p: P) => <I {...p} d="M6 6l12 12M18 6L6 18" />,
  expand: (p: P) => <I {...p} d="M14 4h6v6M10 20H4v-6M20 4l-7 7M4 20l7-7" />,
  shrink: (p: P) => <I {...p} d="M10 4v6H4M14 20v-6h6M4 10l6-6M20 14l-6 6" />,
  back: (p: P) => <I {...p} d="M15 5l-7 7 7 7" />,
  caret: (p: P) => <I {...p} d="M9 6l6 6-6 6" />,
  play: (p: P) => <I {...p} fill d="M8 5.5v13l10-6.5z" />,
  sidebar: (p: P) => <I {...p} d="M5.5 4.5h13a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2v-11a2 2 0 0 1 2-2zM9.5 4.5v15" />,
  pipeline: (p: P) => <I {...p} d="M6 9.3a2.3 2.3 0 1 0 0-4.6 2.3 2.3 0 0 0 0 4.6zM6 19.3a2.3 2.3 0 1 0 0-4.6 2.3 2.3 0 0 0 0 4.6zM18 14.3a2.3 2.3 0 1 0 0-4.6 2.3 2.3 0 0 0 0 4.6zM8.3 7.9l7.4 2.9M8.3 16.1l7.4-2.9" />,
  lab: (p: P) => <I {...p} d="M4.5 19.5h15M7 4.5h10M9.5 4.5v6L5.2 18a1.4 1.4 0 0 0 1.2 2.1h11.2a1.4 1.4 0 0 0 1.2-2.1l-4.3-7.5v-6" />,
  file: (p: P) => <I {...p} d="M7 3.5h6.5L18 8v12.5H7zM13.5 3.5V8H18" />,
  eye: (p: P) => <I {...p} d="M2.5 12s3.5-6.5 9.5-6.5 9.5 6.5 9.5 6.5-3.5 6.5-9.5 6.5S2.5 12 2.5 12zM12 14.6a2.6 2.6 0 1 0 0-5.2 2.6 2.6 0 0 0 0 5.2z" />,
  axis: (p: P) => <I {...p} d="M5 4v15h15M9 15l3-5 3 3 4-7" />,
  table: (p: P) => <I {...p} d="M5.5 5h13a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2zM3.5 10h17M9 10v9" />,
  hash: (p: P) => <I {...p} d="M5 9h14M5 15h14M10 4l-2 16M16 4l-2 16" />,
  tag: (p: P) => <I {...p} d="M4 11V5a1 1 0 0 1 1-1h6l9 9-7 7zM8.5 9.8a1.3 1.3 0 1 0 0-2.6 1.3 1.3 0 0 0 0 2.6z" />,
  raw: (p: P) => <I {...p} d="M7 16.3a1.3 1.3 0 1 0 0-2.6 1.3 1.3 0 0 0 0 2.6zM11 10.3a1.3 1.3 0 1 0 0-2.6 1.3 1.3 0 0 0 0 2.6zM15 15.3a1.3 1.3 0 1 0 0-2.6 1.3 1.3 0 0 0 0 2.6zM17.5 8.8a1.3 1.3 0 1 0 0-2.6 1.3 1.3 0 0 0 0 2.6z" />,
  sliders: (p: P) => <I {...p} d="M4 7h10M18 7h2M4 17h4M12 17h8M16 9a2 2 0 1 0 0-4 2 2 0 0 0 0 4zM10 19a2 2 0 1 0 0-4 2 2 0 0 0 0 4z" />,
  sparkle: (p: P) => <I {...p} d="M12 3.5l1.9 4.6 4.6 1.9-4.6 1.9L12 16.5l-1.9-4.6L5.5 10l4.6-1.9z" />,
  density: (p: P) => <I {...p} d="M7 4h10a3 3 0 0 1 3 3v10a3 3 0 0 1-3 3H7a3 3 0 0 1-3-3V7a3 3 0 0 1 3-3zM13 14.2a3.2 3.2 0 1 0 0-6.4 3.2 3.2 0 0 0 0 6.4z" />,
  dots: (p: P) => <I {...p} d="M7 9a1 1 0 1 0 0-2 1 1 0 0 0 0 2zM12 13a1 1 0 1 0 0-2 1 1 0 0 0 0 2zM16 10a1 1 0 1 0 0-2 1 1 0 0 0 0 2zM9 17a1 1 0 1 0 0-2 1 1 0 0 0 0 2zM15 17a1 1 0 1 0 0-2 1 1 0 0 0 0 2z" />,
  swap: (p: P) => <I {...p} d="M7 4v14M4 15l3 3 3-3M17 20V6M14 9l3-3 3 3" />,
  pointer: (p: P) => <I {...p} d="M6 4l12 7-5.2 1.4L16 19l-2.4 1.2-3.2-6.5L6 18z" />,
  rect: (p: P) => <I {...p} d="M6 6h12v12H6z" />,
  poly: (p: P) => <I {...p} d="M6 8l7-4 6 6-3 9-9-2z" />,
  check: (p: P) => <I {...p} d="M5 12.5l4.5 4.5L19 7" />,
  hist: (p: P) => <I {...p} d="M4 20h16M6 20v-5M10 20V8M14 20v-9M18 20v-4" />,
  copy: (p: P) => <I {...p} d="M9 9h10v11H9zM5 15V4h10" />,
  upload: (p: P) => <I {...p} d="M12 16V4M7 9l5-5 5 5M4.5 15v3.5a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V15" />,
  folder: (p: P) => <I {...p} d="M3.5 7.5a2 2 0 0 1 2-2h4l2 2.5h7a2 2 0 0 1 2 2v7.5a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z" />,
  search: (p: P) => <I {...p} d="M10.5 17a6.5 6.5 0 1 0 0-13 6.5 6.5 0 0 0 0 13zM15.5 15.5L20 20" />,
  gallery: (p: P) => <I {...p} d="M5 5h6v6H5zM13 5h6v6h-6zM5 13h6v6H5zM13 13h6v6h-6z" />,
};

// ---- the flower: six petals that open, drop one by one, and open again --------------------

const PETALS = [[-7, -95], [6, 80], [-4, -50], [8, 120], [-9, -130], [5, 65]];
const petal = (i: number) => <ellipse cx="20" cy="10.6" rx="4.7" ry="7.4" transform={`rotate(${i * 60} 20 18)`} />;

export function Flower({ className = "" }: { className?: string }) {
  return (
    <svg className={`fl ${className}`} viewBox="5 3 30 30" aria-hidden="true">
      <g className="pts">
        {PETALS.map(([dx, r], i) => (
          <g key={i} className={`pt${i % 2 ? " alt" : ""}`} style={{ ["--i" as any]: i, ["--dx" as any]: `${dx}px`, ["--r" as any]: `${r}deg` }}>{petal(i)}</g>
        ))}
      </g>
      <circle className="co" cx="20" cy="18" r="3.4" />
    </svg>
  );
}

/** A still flower whose petals open with progress p in 0..1. */
export function Bloom({ p, className = "" }: { p: number; className?: string }) {
  return (
    <svg className={`fl still ${className}`} viewBox="5 3 30 30" aria-hidden="true">
      <g className="pts">
        {PETALS.map((_, i) => {
          const k = Math.max(0, Math.min(1, p * 6 - i));
          return k ? <g key={i} className={`pt${i % 2 ? " alt" : ""}`} transform={`translate(20 18) scale(${k.toFixed(3)}) translate(-20 -18)`}>{petal(i)}</g> : null;
        })}
      </g>
      {p > 0.12 && <circle className="co" cx="20" cy="18" r="3.4" />}
    </svg>
  );
}

// ---- a flower of one's own, instead of initials -------------------------------------------

/** [petal, second petal, core] shades from the flower palette */
const OWN = [["#d77f9c", "#f2c4d3", "#9e3f62"], ["#c0577c", "#ecb3c6", "#7d2f4b"], ["#d77f9c", "#b9d3bf", "#9e3f62"],
  ["#b56d8c", "#f3d4c0", "#7d3b57"], ["#e39aae", "#cfc3e6", "#a9506f"], ["#c86a86", "#f1dfa6", "#8a3a56"]];

/** A person's own flower, drawn from their name: 5-8 petals, round or pointed, two shades from the
 *  palette and a turn. The same name always gives the same flower; different names rarely match. */
export function OwnFlower({ name, className = "" }: { name: string; className?: string }) {
  let h = 2166136261;
  for (const c of name) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619) >>> 0; }
  const n = 5 + (h % 4);
  const [a, b, core] = OWN[(h >>> 3) % OWN.length];
  const pointed = (h >>> 7) % 2 === 1;
  const turn = (h >>> 9) % 60;
  return (
    <svg className={`own ${className}`} viewBox="2 0 36 36" aria-hidden="true">
      {Array.from({ length: n }, (_, i) => {
        const t = `rotate(${turn + (i * 360) / n} 20 18)`;
        const fill = i % 2 ? b : a;
        return pointed
          ? <path key={i} d="M20 18C16.2 13.4 16.6 6.6 20 3.4C23.4 6.6 23.8 13.4 20 18Z" fill={fill} transform={t} />
          : <ellipse key={i} cx="20" cy="10.6" rx={n > 6 ? 4 : 4.8} ry="7" fill={fill} opacity=".92" transform={t} />;
      })}
      <circle cx="20" cy="18" r="3.3" fill={core} />
    </svg>
  );
}

/** Who is signed in, drawn as their own flower. */
export function Me({ name }: { name: string }) {
  return <span className="me-av" aria-hidden="true"><OwnFlower name={name} /></span>;
}

export function Mark({ small = false }: { small?: boolean }) {
  return <span className={`glyph brand${small ? " sm" : ""}`} aria-hidden="true"><Bloom p={1} className="head" /></span>;
}

export function Sprout({ up }: { up: boolean }) {
  return (
    <svg className={`sprout${up ? " up" : ""}`} viewBox="10 14 40 38" aria-hidden="true">
      <path className="soil" d="M8 46h44" />
      <ellipse className="seed" cx="30" cy={up ? 44 : 42.4} rx={up ? 5 : 5.4} ry={up ? 3.1 : 3.5} transform={`rotate(-16 30 ${up ? 44 : 42.4})`} />
      {up && <>
        <path className="stem" pathLength={1} d="M30 43C30 36 28.8 30 30 21" />
        <path className="lfl" d="M29.6 31C23.6 31 20.4 27.2 21 24.2C26 24.2 29.3 27.3 29.6 31Z" />
        <path className="lfr" d="M30.2 26.6C35.8 26.2 39.2 22.4 38.8 19.4C33.8 19.6 30.6 22.8 30.2 26.6Z" />
      </>}
    </svg>
  );
}
