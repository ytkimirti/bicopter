import { type Drone } from "./sim/drone";
import { SPEC } from "./sim/physics";

export interface View {
  w: number;
  h: number;
  dpr: number;
  ppm: number; // pixels per meter
  cx: number;
  cy: number;
  uw: number; // usable width left of the panel
}

/** insetRight: pixels covered by the side panel; the arena is centered in the rest. */
export function makeView(w: number, h: number, dpr: number, insetRight = 0): View {
  const uw = w - insetRight;
  const ppm = Math.min(uw / (2 * (SPEC.arena.half_w + 1)), h / (2 * (SPEC.arena.half_h + 1)));
  return { w, h, dpr, ppm, cx: uw / 2, cy: h / 2, uw };
}

export const toScreen = (v: View, x: number, y: number): [number, number] => [v.cx + x * v.ppm, v.cy - y * v.ppm];
export const toWorld = (v: View, sx: number, sy: number): [number, number] => [(sx - v.cx) / v.ppm, (v.cy - sy) / v.ppm];

/** Half extents in meters of the area not covered by the panel. */
export const extents = (v: View): [number, number] => [v.uw / 2 / v.ppm, v.h / 2 / v.ppm];

export interface Pose {
  x: number;
  y: number;
  th: number;
  fl: number;
  fr: number;
}

export class Trail {
  xs: Float32Array;
  ys: Float32Array;
  sp: Float32Array;
  n = 0;
  i = 0;
  constructor(readonly cap = 100) {
    this.xs = new Float32Array(cap);
    this.ys = new Float32Array(cap);
    this.sp = new Float32Array(cap);
  }
  push(x: number, y: number, speed: number) {
    this.xs[this.i] = x;
    this.ys[this.i] = y;
    this.sp[this.i] = speed;
    this.i = (this.i + 1) % this.cap;
    this.n = Math.min(this.n + 1, this.cap);
  }
  clear() {
    this.n = 0;
  }
}

export function drawBackground(ctx: CanvasRenderingContext2D, v: View, t: number, windX: number, particles: Float32Array) {
  const g = ctx.createRadialGradient(v.cx, v.cy * 0.8, 0, v.cx, v.cy, Math.max(v.w, v.h) * 0.75);
  g.addColorStop(0, "#111a2b");
  g.addColorStop(1, "#06080e");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, v.w, v.h);

  const [ex, ey] = extents(v);
  ctx.lineWidth = 1;
  for (let x = Math.ceil(-ex); x <= ex; x++) {
    const [sx] = toScreen(v, x, 0);
    ctx.strokeStyle = x === 0 ? "rgba(120,150,200,0.16)" : x % 5 === 0 ? "rgba(120,150,200,0.10)" : "rgba(120,150,200,0.045)";
    ctx.beginPath();
    ctx.moveTo(Math.round(sx) + 0.5, 0);
    ctx.lineTo(Math.round(sx) + 0.5, v.h);
    ctx.stroke();
  }
  for (let y = Math.ceil(-ey); y <= ey; y++) {
    const [, sy] = toScreen(v, 0, y);
    ctx.strokeStyle = y === 0 ? "rgba(120,150,200,0.16)" : y % 5 === 0 ? "rgba(120,150,200,0.10)" : "rgba(120,150,200,0.045)";
    ctx.beginPath();
    ctx.moveTo(0, Math.round(sy) + 0.5);
    ctx.lineTo(v.w, Math.round(sy) + 0.5);
    ctx.stroke();
  }
  // 1 m scale bar.
  ctx.fillStyle = "rgba(160,180,220,0.35)";
  ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.fillRect(v.uw - 16 - v.ppm, v.h - 18, v.ppm, 2);
  ctx.fillText("1 m", v.uw - 16 - v.ppm, v.h - 24);

  // Wind streaks.
  if (Math.abs(windX) > 0.05) {
    ctx.strokeStyle = `rgba(170,200,255,${Math.min(0.22, Math.abs(windX) * 0.04)})`;
    ctx.lineWidth = 1;
    const len = windX * 10;
    ctx.beginPath();
    for (let i = 0; i < particles.length; i += 2) {
      const x = ((particles[i] + t * windX * 0.12) % 1 + 1) % 1;
      const sx = x * v.w;
      const sy = particles[i + 1] * v.h;
      ctx.moveTo(sx, sy);
      ctx.lineTo(sx - len, sy);
    }
    ctx.stroke();
  }
}

export function drawTarget(ctx: CanvasRenderingContext2D, v: View, x: number, y: number, t: number, color: string, auto: boolean) {
  const [sx, sy] = toScreen(v, x, y);
  const r = 0.28 * v.ppm;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.9;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(sx, sy, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.globalAlpha = 0.35 + 0.25 * Math.sin(t * 4);
  ctx.beginPath();
  ctx.arc(sx, sy, r * (1.5 + 0.2 * Math.sin(t * 4)), 0, Math.PI * 2);
  ctx.stroke();
  ctx.globalAlpha = 0.9;
  const c = r * 0.55;
  ctx.beginPath();
  ctx.moveTo(sx - r - c, sy);
  ctx.lineTo(sx - r + c * 0.6, sy);
  ctx.moveTo(sx + r - c * 0.6, sy);
  ctx.lineTo(sx + r + c, sy);
  ctx.moveTo(sx, sy - r - c);
  ctx.lineTo(sx, sy - r + c * 0.6);
  ctx.moveTo(sx, sy + r - c * 0.6);
  ctx.lineTo(sx, sy + r + c);
  ctx.stroke();
  if (auto) {
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.6;
    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillText("AUTO", sx + r + 6, sy - r - 4);
  }
  ctx.restore();
}

export function drawTrail(ctx: CanvasRenderingContext2D, v: View, tr: Trail, rgb: string) {
  if (tr.n < 2) return;
  ctx.save();
  ctx.lineCap = "round";
  const start = (tr.i - tr.n + tr.cap) % tr.cap;
  let [px, py] = toScreen(v, tr.xs[start], tr.ys[start]);
  for (let k = 1; k < tr.n; k++) {
    const j = (start + k) % tr.cap;
    const [sx, sy] = toScreen(v, tr.xs[j], tr.ys[j]);
    const a = k / tr.n;
    ctx.strokeStyle = `rgba(${rgb},${a * a * 0.55})`;
    ctx.lineWidth = 1 + a * 2.5 + Math.min(tr.sp[j] * 0.06, 2);
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(sx, sy);
    ctx.stroke();
    px = sx;
    py = sy;
  }
  ctx.restore();
}

export function drawPlan(ctx: CanvasRenderingContext2D, v: View, xs: Float32Array, ys: Float32Array, n: number, rgb: string) {
  ctx.save();
  for (let k = 2; k < n; k += 2) {
    const [sx, sy] = toScreen(v, xs[k], ys[k]);
    ctx.fillStyle = `rgba(${rgb},${0.55 * (1 - k / n)})`;
    ctx.beginPath();
    ctx.arc(sx, sy, 1.6, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

/** Draws a drone at a pose. Geometry in meters, matches SPEC.arm. */
export function drawDrone(
  ctx: CanvasRenderingContext2D,
  v: View,
  p: Pose,
  rgb: string,
  t: number,
  opts: { alpha?: number; glow?: boolean; label?: string } = {},
) {
  const alpha = opts.alpha ?? 1;
  const [sx, sy] = toScreen(v, p.x, p.y);
  const s = v.ppm;
  const arm = SPEC.arm;
  const T = SPEC.max_thrust;
  ctx.save();
  ctx.translate(sx, sy);
  ctx.rotate(-p.th);
  ctx.scale(s, s);
  ctx.globalAlpha = alpha;

  // Thrust plumes (drawn in body frame, pointing body-down).
  ctx.globalCompositeOperation = "lighter";
  for (const [mx, f] of [
    [-arm, p.fl],
    [arm, p.fr],
  ] as const) {
    const k = Math.max(0, Math.min(1.2, f / T));
    const len = 0.06 + 0.55 * k;
    const flick = 1 + 0.12 * Math.sin(t * 60 + mx * 40);
    const g = ctx.createLinearGradient(0, 0.02, 0, 0.02 + len * flick);
    g.addColorStop(0, `rgba(255,240,200,${0.9 * k})`);
    g.addColorStop(0.25, `rgba(255,170,70,${0.65 * k})`);
    g.addColorStop(1, "rgba(255,80,30,0)");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.moveTo(mx - 0.035, 0.0);
    ctx.quadraticCurveTo(mx, 0.02 + len * flick * 1.2, mx + 0.035, 0.0);
    ctx.closePath();
    // Canvas y grows downward; body-down is +y here because the frame is not flipped.
    ctx.fill();
  }
  ctx.globalCompositeOperation = "source-over";

  if (opts.glow) {
    ctx.shadowColor = `rgba(${rgb},0.9)`;
    ctx.shadowBlur = 0.35 * s;
  }
  // Body.
  ctx.fillStyle = "#d9e2f2";
  roundRect(ctx, -0.17, -0.045, 0.34, 0.07, 0.025);
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.fillStyle = `rgb(${rgb})`;
  roundRect(ctx, -0.07, -0.03, 0.14, 0.04, 0.012);
  ctx.fill();
  // Arms and motor pods.
  ctx.fillStyle = "#8b97ad";
  ctx.fillRect(-arm, -0.02, 2 * arm, 0.018);
  for (const mx of [-arm, arm]) {
    ctx.fillStyle = "#2a3346";
    roundRect(ctx, mx - 0.032, -0.06, 0.064, 0.07, 0.012);
    ctx.fill();
    // Spinning prop: a thin bar whose width oscillates.
    const w = 0.1 * Math.abs(Math.cos(t * 50 + mx * 13)) + 0.02;
    ctx.fillStyle = "rgba(220,230,255,0.55)";
    ctx.fillRect(mx - w, -0.078, 2 * w, 0.012);
    ctx.fillStyle = "rgba(220,230,255,0.12)";
    ctx.fillRect(mx - 0.12, -0.08, 0.24, 0.016);
  }
  ctx.restore();

  if (opts.label) {
    ctx.save();
    ctx.globalAlpha = 0.75 * alpha;
    ctx.fillStyle = `rgb(${rgb})`;
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign = "center";
    ctx.fillText(opts.label, sx, sy - 0.45 * s);
    ctx.restore();
  }
}

export function drawOffscreen(ctx: CanvasRenderingContext2D, v: View, x: number, y: number, rgb: string) {
  const [sx, sy] = toScreen(v, x, y);
  const m = 18;
  if (sx >= 0 && sx <= v.w && sy >= 0 && sy <= v.h) return;
  const ax = Math.max(m, Math.min(v.w - m, sx));
  const ay = Math.max(m, Math.min(v.h - m, sy));
  const ang = Math.atan2(sy - ay, sx - ax);
  ctx.save();
  ctx.translate(ax, ay);
  ctx.rotate(ang);
  ctx.fillStyle = `rgba(${rgb},0.9)`;
  ctx.beginPath();
  ctx.moveTo(10, 0);
  ctx.lineTo(-6, -7);
  ctx.lineTo(-6, 7);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/** Small network activity view: inputs, hidden layers, outputs as grids of colored cells. */
export function drawNet(ctx: CanvasRenderingContext2D, w: number, h: number, obs: Float32Array, acts: Float32Array[], out: Float32Array) {
  ctx.clearRect(0, 0, w, h);
  const cols = [obs, ...acts.slice(0, -1), out];
  const maxRows = 16;
  const grids = cols.map((c) => Math.ceil(c.length / maxRows));
  const cell = Math.min((h - 8) / maxRows, (w - 8 - 14 * (cols.length - 1)) / grids.reduce((a, b) => a + b, 0));
  let x = (w - (grids.reduce((a, b) => a + b, 0) * cell + 14 * (cols.length - 1))) / 2;
  cols.forEach((col, ci) => {
    const per = grids[ci];
    const rows = Math.ceil(col.length / per);
    const y0 = (h - rows * cell) / 2;
    for (let i = 0; i < col.length; i++) {
      const v = Math.max(-1, Math.min(1, col[i]));
      ctx.fillStyle = v >= 0 ? `rgba(94,240,255,${0.1 + 0.9 * v})` : `rgba(255,120,80,${0.1 - 0.9 * v})`;
      ctx.fillRect(x + (i % per) * cell, y0 + Math.floor(i / per) * cell, cell - 1, cell - 1);
    }
    x += per * cell + 14;
  });
}

export function drawChart(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  series: { gen: number; val: number }[],
  refs: { label: string; value: number; color: string }[],
) {
  ctx.clearRect(0, 0, w, h);
  const vals = series.map((s) => s.val).concat(refs.map((r) => r.value));
  const lo = Math.min(0, ...vals);
  const hi = Math.max(0.5, ...vals) * 1.05;
  const maxGen = Math.max(50, series.length ? series[series.length - 1].gen : 0);
  const X = (g: number) => 6 + (g / maxGen) * (w - 12);
  const Y = (v: number) => h - 14 - ((v - lo) / (hi - lo)) * (h - 24);
  ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
  for (const r of refs) {
    ctx.strokeStyle = r.color;
    ctx.globalAlpha = 0.7;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(6, Y(r.value));
    ctx.lineTo(w - 6, Y(r.value));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = r.color;
    ctx.fillText(r.label, 8, Y(r.value) - 3);
  }
  ctx.globalAlpha = 1;
  ctx.strokeStyle = "#8cff6b";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  series.forEach((s, i) => (i ? ctx.lineTo(X(s.gen), Y(s.val)) : ctx.moveTo(X(s.gen), Y(s.val))));
  ctx.stroke();
  ctx.fillStyle = "rgba(160,180,220,0.6)";
  ctx.fillText(`gen ${maxGen}`, w - 52, h - 2);
}

export type { Drone };
