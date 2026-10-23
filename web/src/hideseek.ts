import { drawBackground, drawDrone, drawTrail, makeView, type Pose, toScreen, Trail, type View } from "./render";
import { SPEC, wrapAngle } from "./sim/physics";
import "./style.css";

const CYAN = "94,240,255";
const ORANGE = "255,174,66";
const RED = "255,92,92";
const GOLD = "255,214,94";
const TEAM_RGB = [CYAN, ORANGE];
/** Drawn at the size of the collision box (half width 0.25 m), unlike free flight's larger-than-life drones. */
const DRONE_DRAW = 1.08;

// ---------- data ----------

interface Episode {
  walls: number[][]; // cx, cy, hx, hy
  room: number[]; // interior x0, x1, y0, y1
  doors: number[];
  frames: number;
  data: string; // base64 little-endian int16, frames x channels
  hider_reward: number;
}

interface Replay {
  fps: number;
  decision_hz: number;
  prep_frames: number;
  channels: string[];
  box_half: number[][];
  drone_half: number[];
  episodes: Episode[];
}

interface Checkpoint {
  id: string;
  file: string;
  decisions: number;
  episodes_seen: number;
  hours: number;
  metrics: Record<string, number>;
}

interface Stage {
  id: string;
  label: string;
  metric: string;
  decisions: number;
  hours: number;
}

interface Manifest {
  checkpoints: Checkpoint[];
  stages: Stage[];
}

type Curves = Record<string, number[]>;

/** One decoded match: per-frame channels as floats, plus the static layout. */
class Match {
  readonly ch: Float32Array;
  readonly n: number;
  readonly w: number;
  constructor(
    readonly ep: Episode,
    readonly rep: Replay,
  ) {
    const bin = atob(ep.data);
    const view = new DataView(new ArrayBuffer(bin.length));
    for (let i = 0; i < bin.length; i++) view.setUint8(i, bin.charCodeAt(i));
    this.w = rep.channels.length;
    this.n = ep.frames;
    this.ch = new Float32Array(this.n * this.w);
    // The first 28 channels (drone poses and thrust, box positions) are scaled by 1000; the rest are raw indices.
    for (let i = 0; i < this.ch.length; i++) {
      const v = view.getInt16(2 * i, true);
      this.ch[i] = i % this.w < 28 ? v / 1000 : v;
    }
  }
  get(f: number, c: number) {
    return this.ch[Math.min(this.n - 1, Math.max(0, f)) * this.w + c];
  }
  drone(f: number, i: number): Pose {
    const T = SPEC.max_thrust;
    return { x: this.get(f, 5 * i), y: this.get(f, 5 * i + 1), th: this.get(f, 5 * i + 2), fl: this.get(f, 5 * i + 3) * T, fr: this.get(f, 5 * i + 4) * T };
  }
  box(f: number, b: number): [number, number] {
    return [this.get(f, 20 + 2 * b), this.get(f, 21 + 2 * b)];
  }
  locked(f: number, b: number) {
    return this.get(f, 28 + b);
  }
  held(f: number, b: number) {
    return this.get(f, 32 + b);
  }
  seen(f: number, h: number) {
    return this.get(f, 36 + h) > 0;
  }
}

// ---------- geometry ----------

/** Same slab test as game.segment_blocked: does the segment cross any box (cx, cy, hx, hy)? */
function blocked(x0: number, y0: number, x1: number, y1: number, boxes: number[][]) {
  const dx = x1 - x0;
  const dy = y1 - y0;
  const ix = 1 / (Math.abs(dx) < 1e-6 ? 1e-6 : dx);
  const iy = 1 / (Math.abs(dy) < 1e-6 ? 1e-6 : dy);
  for (const [cx, cy, hx, hy] of boxes) {
    const ax = (cx - hx - x0) * ix;
    const bx = (cx + hx - x0) * ix;
    const ay = (cy - hy - y0) * iy;
    const by = (cy + hy - y0) * iy;
    const tmin = Math.max(Math.min(ax, bx), Math.min(ay, by));
    const tmax = Math.min(Math.max(ax, bx), Math.max(ay, by));
    if (tmax >= tmin && tmax > 0 && tmin < 1) return true;
  }
  return false;
}

// ---------- state ----------

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>("c");
const ctx = canvas.getContext("2d")!;
const curveCanvas = $<HTMLCanvasElement>("curves");
const curveCtx = curveCanvas.getContext("2d")!;
const panelInset = () => (innerWidth > 720 ? 300 : 0);
let view: View = makeView(innerWidth, innerHeight, devicePixelRatio || 1, panelInset());

let manifest: Manifest = { checkpoints: [], stages: [] };
let curves: Curves = {};
const replays = new Map<string, Promise<Replay>>();
let ckpt = 0;
let epIdx = 0;
let match: Match | null = null;
let replay: Replay | null = null;
let frameT = 0; // fractional frame index
let playing = true;
let speed = 1;
const opts = { sight: true, trail: true, auto: true };
const trails = Array.from({ length: 4 }, () => new Trail(40));
let lastTrailFrame = -1;
const particles = new Float32Array(0);

function loadReplay(c: Checkpoint): Promise<Replay> {
  let p = replays.get(c.id);
  if (!p) {
    p = fetch(`/hideseek/${c.file}`).then((r) => r.json());
    replays.set(c.id, p);
  }
  return p;
}

async function selectCheckpoint(i: number, keepEpisode = true) {
  ckpt = Math.max(0, Math.min(manifest.checkpoints.length - 1, i));
  $<HTMLInputElement>("ckpt").value = String(ckpt);
  const c = manifest.checkpoints[ckpt];
  syncCheckpointInfo();
  const rep = await loadReplay(c);
  if (manifest.checkpoints[ckpt] !== c) return; // a newer selection won
  replay = rep;
  if (!keepEpisode || epIdx >= rep.episodes.length) epIdx = 0;
  buildEpisodeButtons();
  startEpisode(epIdx);
  // Warm the neighbours so stepping through the timeline is instant.
  for (const j of [ckpt - 1, ckpt + 1]) if (manifest.checkpoints[j]) void loadReplay(manifest.checkpoints[j]);
}

function startEpisode(i: number) {
  if (!replay) return;
  epIdx = i;
  match = new Match(replay.episodes[i], replay);
  frameT = 0;
  lastTrailFrame = -1;
  for (const t of trails) t.clear();
  const scrub = $<HTMLInputElement>("scrub");
  scrub.max = String(match.n - 1);
  document.querySelectorAll<HTMLButtonElement>("#episodes button").forEach((b, j) => b.classList.toggle("on", j === i));
}

function buildEpisodeButtons() {
  const box = $("episodes");
  if (!replay) return;
  if (box.childElementCount !== replay.episodes.length) {
    box.innerHTML = "";
    replay.episodes.forEach((_, j) => {
      const b = document.createElement("button");
      b.textContent = String(j + 1);
      b.addEventListener("click", () => startEpisode(j));
      box.appendChild(b);
    });
  }
}

const fmtDecisions = (d: number) => (d >= 1e9 ? `${(d / 1e9).toFixed(2)}B` : `${Math.round(d / 1e6)}M`);

function syncCheckpointInfo() {
  const c = manifest.checkpoints[ckpt];
  if (!c) return;
  const m = c.metrics;
  $("ckpt-when").textContent = `${c.hours.toFixed(1)} h`;
  $("ckpt-info").innerHTML =
    `After <b>${fmtDecisions(c.decisions)}</b> decisions per drone (${(c.episodes_seen / 1e6).toFixed(2)}M games). ` +
    `Hiders stayed hidden <b>${Math.round(100 * m.hidden)}%</b> of the seek phase; ` +
    `${Math.round(100 * m.in_room)}% were in the room when the seekers woke up.`;
  document.querySelectorAll<HTMLButtonElement>(".stages button").forEach((b) => {
    b.classList.toggle("reached", Number(b.dataset.d) <= c.decisions);
  });
}

// ---------- training curves ----------

const SERIES: { key: string; label: string; rgb: string; scale?: number }[] = [
  { key: "hidden", label: "hidden", rgb: CYAN },
  { key: "in_room", label: "in room", rgb: "220,230,255" },
  { key: "sealed_gain", label: "doors sealed by hiders", rgb: "140,255,107" },
  { key: "locked_prep", label: "boxes locked", rgb: "190,140,255", scale: 0.25 },
  { key: "key_unlocks", label: "key unlocks", rgb: GOLD },
];

function drawCurves() {
  const w = curveCanvas.width;
  const h = curveCanvas.height;
  const g = curveCtx;
  g.clearRect(0, 0, w, h);
  const xs = curves.decisions;
  if (!xs || xs.length < 2) {
    g.fillStyle = "rgba(160,180,220,0.5)";
    g.font = "11px ui-monospace, monospace";
    g.fillText("training curves appear after the first checkpoint", 8, h / 2);
    return;
  }
  const maxX = xs[xs.length - 1];
  const X = (d: number) => 6 + (d / maxX) * (w - 12);
  const Y = (v: number) => h - 12 - Math.max(0, Math.min(1, v)) * (h - 22);
  g.strokeStyle = "rgba(140,165,210,0.12)";
  g.lineWidth = 1;
  for (const v of [0, 0.5, 1]) {
    g.beginPath();
    g.moveTo(6, Y(v) + 0.5);
    g.lineTo(w - 6, Y(v) + 0.5);
    g.stroke();
  }
  for (const s of SERIES) {
    const ys = curves[s.key];
    if (!ys) continue;
    g.strokeStyle = `rgb(${s.rgb})`;
    g.globalAlpha = 0.9;
    g.lineWidth = 1.4;
    g.beginPath();
    ys.forEach((v, i) => {
      const px = X(xs[i]);
      const py = Y(v * (s.scale ?? 1));
      if (i) g.lineTo(px, py);
      else g.moveTo(px, py);
    });
    g.stroke();
  }
  g.globalAlpha = 1;
  // Stage onsets and the selected checkpoint.
  g.font = "9px ui-monospace, monospace";
  manifest.stages.forEach((st, i) => {
    const px = X(st.decisions);
    g.fillStyle = "rgba(255,255,255,0.75)";
    g.beginPath();
    g.arc(px, 6, 5.5, 0, Math.PI * 2);
    g.fill();
    g.fillStyle = "#06080e";
    g.textAlign = "center";
    g.fillText(String(i + 1), px, 9);
  });
  g.textAlign = "left";
  const c = manifest.checkpoints[ckpt];
  if (c) {
    const px = X(c.decisions);
    g.strokeStyle = "rgba(255,255,255,0.6)";
    g.setLineDash([3, 3]);
    g.beginPath();
    g.moveTo(px + 0.5, 12);
    g.lineTo(px + 0.5, h - 10);
    g.stroke();
    g.setLineDash([]);
  }
  g.fillStyle = "rgba(160,180,220,0.6)";
  g.fillText(`${fmtDecisions(maxX)} decisions`, w - 92, h - 1);
}

function buildLegend() {
  $("curve-legend").innerHTML = SERIES.map(
    (s) => `<span><i style="background:rgb(${s.rgb})"></i>${s.label}${s.scale ? " ÷4" : ""}</span>`,
  ).join("");
  const box = $("stages");
  box.innerHTML = "";
  if (!manifest.stages.length) {
    box.innerHTML = `<p class="sub">No strategy has taken hold yet. Stages show up here as the metrics cross their thresholds.</p>`;
    return;
  }
  manifest.stages.forEach((st, i) => {
    const b = document.createElement("button");
    b.dataset.d = String(st.decisions);
    b.innerHTML = `<em>${i + 1}</em><span>${st.label}</span><small>${st.hours.toFixed(1)} h</small>`;
    // Jump to the first checkpoint at or after the onset.
    b.addEventListener("click", () => {
      const j = manifest.checkpoints.findIndex((c) => c.decisions >= st.decisions);
      void selectCheckpoint(j < 0 ? manifest.checkpoints.length - 1 : j);
    });
    box.appendChild(b);
  });
}

// ---------- drawing ----------

function rect(v: View, cx: number, cy: number, hx: number, hy: number): [number, number, number, number] {
  const [x0, y0] = toScreen(v, cx - hx, cy + hy);
  return [x0, y0, 2 * hx * v.ppm, 2 * hy * v.ppm];
}

function drawWalls(m: Match) {
  const { walls, room } = m.ep;
  // Room interior, so "inside" reads at a glance.
  ctx.fillStyle = "rgba(94,240,255,0.035)";
  const [rx, ry, rw, rh] = rect(view, (room[0] + room[1]) / 2, (room[2] + room[3]) / 2, (room[1] - room[0]) / 2, (room[3] - room[2]) / 2);
  ctx.fillRect(rx, ry, rw, rh);
  // The first four walls are the arena's floor, ceiling and sides: draw them as its border, shading outside.
  const [ax, ay, aw, ah] = rect(view, 0, 0, SPEC.arena.half_w, SPEC.arena.half_h);
  ctx.fillStyle = "rgba(3,5,10,0.55)";
  ctx.fillRect(0, 0, view.w, ay);
  ctx.fillRect(0, ay + ah, view.w, view.h - ay - ah);
  ctx.fillRect(0, ay, ax, ah);
  ctx.fillRect(ax + aw, ay, view.w - ax - aw, ah);
  ctx.strokeStyle = "rgba(150,175,225,0.45)";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(ax, ay, aw, ah);
  for (const [cx, cy, hx, hy] of walls.slice(4)) {
    if (hx <= 0 || hy <= 0) continue;
    const [x, y, w, h] = rect(view, cx, cy, hx, hy);
    ctx.fillStyle = "#162036";
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = "rgba(150,175,225,0.4)";
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
  }
}

function drawPadlock(sx: number, sy: number, r: number, rgb: string) {
  ctx.save();
  ctx.strokeStyle = `rgb(${rgb})`;
  ctx.fillStyle = `rgb(${rgb})`;
  ctx.lineWidth = Math.max(1.5, r * 0.28);
  ctx.beginPath();
  ctx.arc(sx, sy - r * 0.35, r * 0.5, Math.PI, 0);
  ctx.stroke();
  ctx.fillRect(sx - r * 0.75, sy - r * 0.35, r * 1.5, r * 1.1);
  ctx.restore();
}

function drawBoxes(m: Match, f: number, a: number) {
  const half = m.rep.box_half;
  for (let b = 0; b < half.length; b++) {
    const [x0, y0] = m.box(f, b);
    const [x1, y1] = m.box(f + 1, b);
    const x = x0 + (x1 - x0) * a;
    const y = y0 + (y1 - y0) * a;
    const [hx, hy] = half[b];
    const [sx, sy, w, h] = rect(view, x, y, hx, hy);
    const isKey = b === half.length - 1;
    const lock = m.locked(f, b);
    ctx.fillStyle = isKey ? `rgba(${GOLD},0.85)` : "rgba(170,185,215,0.18)";
    ctx.fillRect(sx, sy, w, h);
    ctx.lineWidth = lock >= 0 ? 2.5 : 1.2;
    ctx.strokeStyle = lock >= 0 ? `rgb(${TEAM_RGB[lock]})` : isKey ? `rgb(${GOLD})` : "rgba(205,218,245,0.6)";
    ctx.strokeRect(sx + 0.5, sy + 0.5, w - 1, h - 1);
    if (isKey) {
      // A key glyph: ring and bit.
      const r = Math.min(w, h) * 0.22;
      const cx = sx + w * 0.38;
      const cy = sy + h / 2;
      ctx.strokeStyle = "#3a2c05";
      ctx.lineWidth = Math.max(1.2, r * 0.45);
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.moveTo(cx + r, cy);
      ctx.lineTo(sx + w * 0.82, cy);
      ctx.moveTo(sx + w * 0.72, cy);
      ctx.lineTo(sx + w * 0.72, cy + r);
      ctx.stroke();
    }
    if (lock >= 0) drawPadlock(sx + w / 2, isKey ? sy - 0.18 * view.ppm : sy + h / 2, Math.max(4, 0.1 * view.ppm), TEAM_RGB[lock]);
  }
}

function drawSight(m: Match, f: number, poses: Pose[]) {
  if (f < m.rep.prep_frames) return;
  const boxes = m.rep.box_half.map(([hx, hy], b) => [...m.box(f, b), hx, hy]);
  for (let h = 0; h < 2; h++) {
    if (!m.seen(f, h)) continue;
    const hp = poses[h];
    for (let s = 2; s < 4; s++) {
      const sp = poses[s];
      // A drone sees past the box it carries.
      const occ = m.ep.walls.concat(boxes.filter((_, b) => m.held(f, b) !== s));
      const visible = [-0.2, 0, 0.2].some((dx) => !blocked(sp.x, sp.y, hp.x + dx, hp.y, occ));
      if (!visible) continue;
      const [ax, ay] = toScreen(view, sp.x, sp.y);
      const [bx, by] = toScreen(view, hp.x, hp.y);
      const grad = ctx.createLinearGradient(ax, ay, bx, by);
      grad.addColorStop(0, `rgba(${ORANGE},0.15)`);
      grad.addColorStop(1, `rgba(${RED},0.75)`);
      ctx.strokeStyle = grad;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 5]);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
}

function lerpPose(p: Pose, q: Pose, a: number): Pose {
  return { x: p.x + (q.x - p.x) * a, y: p.y + (q.y - p.y) * a, th: p.th + wrapAngle(q.th - p.th) * a, fl: q.fl, fr: q.fr };
}

function drawMatch(t: number) {
  const m = match;
  if (!m) return;
  const f = Math.floor(frameT);
  const a = frameT - f;
  const poses = [0, 1, 2, 3].map((i) => lerpPose(m.drone(f, i), m.drone(f + 1, i), a));
  const prep = f < m.rep.prep_frames;

  drawWalls(m);
  if (opts.trail) {
    if (f !== lastTrailFrame) {
      lastTrailFrame = f;
      poses.forEach((p, i) => trails[i].push(p.x, p.y, 0));
    }
    trails.forEach((tr, i) => drawTrail(ctx, view, tr, i < 2 ? CYAN : ORANGE));
  }
  if (opts.sight) drawSight(m, f, poses);
  // Tethers to carried boxes.
  for (let b = 0; b < m.rep.box_half.length; b++) {
    const holder = m.held(f, b);
    if (holder < 0) continue;
    const p = poses[holder];
    const [x0, y0] = toScreen(view, p.x, p.y);
    const [bx, by] = m.box(f, b);
    const [x1, y1] = toScreen(view, bx, by + m.rep.box_half[b][1]);
    ctx.strokeStyle = `rgba(${TEAM_RGB[holder < 2 ? 0 : 1]},0.7)`;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
  }
  drawBoxes(m, f, a);
  poses.forEach((p, i) => {
    const seeker = i >= 2;
    const seen = !seeker && !prep && m.seen(f, i);
    if (seen) {
      const [sx, sy] = toScreen(view, p.x, p.y);
      ctx.strokeStyle = `rgba(${RED},${0.55 + 0.35 * Math.sin(t * 12)})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(sx, sy, 0.42 * view.ppm, 0, Math.PI * 2);
      ctx.stroke();
    }
    drawDrone(ctx, view, p, seeker ? ORANGE : CYAN, t, {
      alpha: seeker && prep ? 0.45 : 1,
      glow: true,
      scale: DRONE_DRAW,
      label: seeker ? (prep ? "zzz" : `S${i - 1}`) : `H${i + 1}`,
    });
  });
}

function updateStats() {
  const m = match;
  if (!m) return;
  const f = Math.floor(frameT);
  const fps = m.rep.fps;
  const prep = f < m.rep.prep_frames;
  let hiddenFrames = 0;
  let seekFrames = 0;
  for (let i = m.rep.prep_frames; i <= f; i++) {
    seekFrames++;
    if (!m.seen(i, 0) && !m.seen(i, 1)) hiddenFrames++;
  }
  const left = prep ? (m.rep.prep_frames - f) / fps : (m.n - f) / fps;
  const nowSeen = !prep && (m.seen(f, 0) || m.seen(f, 1));
  const c = manifest.checkpoints[ckpt];
  $("stats").innerHTML =
    (c ? `checkpoint <b>${fmtDecisions(c.decisions).padStart(6)}</b>  ${c.hours.toFixed(1)} h\n` : "") +
    `phase      <b>${(prep ? "prep" : "seek").padStart(6)}</b>  ${left.toFixed(1)} s left\n` +
    `hiders     <b>${(prep ? "-" : nowSeen ? "SEEN" : "hidden").padStart(6)}</b>\n` +
    `hidden     <b>${(seekFrames ? `${Math.round((100 * hiddenFrames) / seekFrames)}%` : "-").padStart(6)}</b>  of the seek phase`;
}

// ---------- loop ----------

let last = performance.now();
let statT = 0;
let simT = 0;

function frame(now: number) {
  const real = Math.min(0.1, (now - last) / 1000);
  last = now;
  const m = match;
  if (m && playing) {
    simT += real * speed;
    frameT += real * speed * m.rep.fps;
    if (frameT >= m.n - 1) {
      if (opts.auto && replay) startEpisode((epIdx + 1) % replay.episodes.length);
      else frameT = 0;
    }
    $<HTMLInputElement>("scrub").value = String(Math.floor(frameT));
  }
  ctx.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
  drawBackground(ctx, view, simT, 0, particles);
  drawMatch(simT);
  if (now - statT > 100) {
    statT = now;
    updateStats();
  }
  requestAnimationFrame(frame);
}

function resize() {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.round(innerWidth * dpr);
  canvas.height = Math.round(innerHeight * dpr);
  view = makeView(innerWidth, innerHeight, dpr, panelInset());
}

function initUI() {
  const ck = $<HTMLInputElement>("ckpt");
  ck.max = String(Math.max(0, manifest.checkpoints.length - 1));
  ck.addEventListener("input", () => void selectCheckpoint(Number(ck.value)).then(drawCurves));
  const scrub = $<HTMLInputElement>("scrub");
  scrub.addEventListener("input", () => {
    frameT = Number(scrub.value);
    for (const t of trails) t.clear();
  });
  document.querySelectorAll<HTMLButtonElement>("#speed button").forEach((b) =>
    b.addEventListener("click", () => {
      speed = Number(b.dataset.s);
      document.querySelectorAll("#speed button").forEach((x) => x.classList.toggle("on", x === b));
    }),
  );
  const play = $<HTMLButtonElement>("play");
  const syncPlay = () => (play.textContent = playing ? "Pause" : "Play");
  play.addEventListener("click", () => {
    playing = !playing;
    syncPlay();
  });
  $("restart").addEventListener("click", () => startEpisode(epIdx));
  for (const [id, key] of [
    ["v-sight", "sight"],
    ["v-trail", "trail"],
    ["v-auto", "auto"],
  ] as const) {
    const el = $<HTMLInputElement>(id);
    el.checked = opts[key];
    el.addEventListener("change", () => (opts[key] = el.checked));
  }
  $("panel-toggle").addEventListener("click", () => $("panel").classList.toggle("open"));
  window.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLInputElement && e.target.type !== "checkbox" && e.target.type !== "range") return;
    if (e.key === " ") {
      playing = !playing;
      syncPlay();
      e.preventDefault();
    } else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      void selectCheckpoint(ckpt + (e.key === "ArrowLeft" ? -1 : 1)).then(drawCurves);
      e.preventDefault();
    } else if (e.key.toLowerCase() === "r") startEpisode(epIdx);
  });
  canvas.addEventListener("click", () => {
    playing = !playing;
    syncPlay();
  });
  window.addEventListener("resize", resize);
}

async function boot() {
  resize();
  try {
    [manifest, curves] = await Promise.all([
      fetch("/hideseek/manifest.json").then((r) => r.json()),
      fetch("/hideseek/curves.json").then((r) => r.json()),
    ]);
  } catch {
    $("ckpt-info").textContent = "No training run published yet.";
  }
  initUI();
  buildLegend();
  if (manifest.checkpoints.length) await selectCheckpoint(manifest.checkpoints.length - 1, false);
  drawCurves();
  requestAnimationFrame(frame);
  Object.assign(window, { hideseek: { get match() { return match; }, get frame() { return frameT; }, manifest } });
}

void boot();
