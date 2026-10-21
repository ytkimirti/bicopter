import { ES_SIZES } from "./evolve/es";
import { EvoPool } from "./evolve/pool";
import {
  drawBackground,
  drawChart,
  drawDrone,
  drawNet,
  drawOffscreen,
  drawPlan,
  drawTarget,
  drawTrail,
  extents,
  makeView,
  type Pose,
  toWorld,
  Trail,
  type View,
} from "./render";
import { Drone } from "./sim/drone";
import { DT, SPEC, wrapAngle } from "./sim/physics";
import { type Brain, type BrainFile, MLP, PID } from "./sim/policy";
import "./style.css";

const CYAN = "94,240,255";
const ORANGE = "255,174,66";
const GREEN = "140,255,107";

// ---------- brains ----------

interface ManifestEntry {
  id: string;
  label: string;
  file: string;
  steps: number;
  info: string;
}
interface Manifest {
  default: string;
  brains: ManifestEntry[];
}

const brainFiles = new Map<string, BrainFile>();
const brains = new Map<string, Brain>();
const brainInfo = new Map<string, string>();
let manifest: Manifest = { default: "pd", brains: [] };
const pd = new PID();
brains.set("pd", pd);
brainInfo.set("pd", "Hand-tuned cascaded PD controller, the classic way. It assumes nominal mass and no wind.");

async function loadBrain(id: string): Promise<Brain> {
  const hit = brains.get(id);
  if (hit) return hit;
  const e = manifest.brains.find((b) => b.id === id);
  if (!e) return pd;
  const f: BrainFile = await (await fetch(`/brains/${e.file}`)).json();
  brainFiles.set(id, f);
  const b = MLP.fromFile(e.label, f);
  brains.set(id, b);
  return b;
}

// ---------- state ----------

interface Flyer {
  drone: Drone;
  trail: Trail;
  rgb: string;
  label?: string;
  prev: Pose;
  ox: number;
  oy: number;
  unwrapped: number;
  lastTh: number;
  alpha: number;
  team: "nn" | "pd";
}

const settings = {
  mode: "fly" as "fly" | "evolve",
  brainId: "pd",
  swarm: 1,
  race: false,
  mass: 1,
  thrust: 1,
  imbalance: 0,
  wind: 0,
  gusts: false,
  lag: SPEC.motor_tau,
  latency: 0,
  plan: true,
  trail: true,
  slow: false,
  net: true,
};

const canvas = document.getElementById("c") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;
const coarse = matchMedia("(pointer: coarse)").matches;
const panelInset = () => (innerWidth > 720 ? 300 : 0);
let view: View = makeView(innerWidth, innerHeight, devicePixelRatio || 1, panelInset());
const windParticles = Float32Array.from({ length: 240 }, () => Math.random());

let flyers: Flyer[] = [];
let evolvers: Flyer[] = [];
const target = { x: 0, y: 0, vx: 0, vy: 0, px: 0, py: 0, auto: true };
const pointer = { x: 0, y: 0, lastMove: -1e9, down: false, touch: false };
const gust = { x: 0, y: 0 };
let simTime = 0;
let grab: { f: Flyer; ox: number; oy: number } | null = null;

const pose = (d: Drone): Pose => ({ x: d.phys.x, y: d.phys.y, th: d.phys.th, fl: d.phys.fl, fr: d.phys.fr });

function makeFlyer(brain: Brain, x: number, y: number, rgb: string, label?: string, alpha = 1): Flyer {
  const d = new Drone(brain, x, y);
  return { drone: d, trail: new Trail(), rgb, label, prev: pose(d), ox: 0, oy: 0, unwrapped: 0, lastTh: 0, alpha, team: "nn" };
}

/** Hexagonal formation offsets so swarm members do not sit on top of each other. */
function formation(n: number): [number, number][] {
  const pts: [number, number][] = [[0, 0]];
  const sp = 1.15;
  for (let ring = 1; pts.length < n; ring++) {
    for (let k = 0; k < 6 * ring && pts.length < n; k++) {
      const a = (k / (6 * ring)) * Math.PI * 2;
      pts.push([Math.cos(a) * ring * sp, Math.sin(a) * ring * sp * 0.8]);
    }
  }
  return pts;
}

async function rebuildFlyers() {
  const id = settings.brainId;
  const brain = await loadBrain(id);
  // A newer selection may have finished first while this one was still fetching.
  if (id !== settings.brainId) return;
  const old = flyers.filter((f) => f.team === "nn");
  const offs = formation(settings.swarm);
  // Racing teams fly side by side, each around its own copy of the target.
  const shift = settings.race ? (settings.swarm === 1 ? 0.8 : 1.6 + Math.sqrt(settings.swarm) * 0.6) : 0;
  flyers = offs.map(([ox, oy], i) => {
    const f = makeFlyer(brain, ox - shift, oy, CYAN);
    if (old[i]) adopt(f, old[i]);
    f.ox = ox - shift;
    f.oy = oy;
    return f;
  });
  if (settings.race) {
    flyers[0].label = brain.name;
    flyers.push(
      ...offs.map(([ox, oy], i) => {
        const f = makeFlyer(pd, ox + shift, oy, ORANGE, i === 0 ? "classic PD" : undefined);
        f.team = "pd";
        f.ox = ox + shift;
        f.oy = oy;
        return f;
      }),
    );
  }
  applyPhysics();
}

/** Continue an existing drone's flight in a new flyer. */
function adopt(f: Flyer, from: Flyer) {
  Object.assign(f.drone.phys, from.drone.phys);
  f.drone.past.set(from.drone.past);
  f.prev = from.prev;
  f.lastTh = from.lastTh;
  f.unwrapped = from.unwrapped;
}

function applyPhysics() {
  for (const f of [...flyers, ...evolvers]) {
    const p = f.drone.params;
    p.mass = SPEC.mass * settings.mass;
    p.inertia = SPEC.inertia * settings.mass;
    p.tmaxL = SPEC.max_thrust * settings.thrust * (1 - settings.imbalance);
    p.tmaxR = SPEC.max_thrust * settings.thrust * (1 + settings.imbalance);
    p.tau = settings.lag;
    p.latency = settings.latency;
  }
}

// ---------- autopilot ----------

type Leg = { kind: "jump"; x: number; y: number; dur: number } | { kind: "orbit" | "eight"; dur: number };
const TOUR: Leg[] = [
  { kind: "jump", x: -0.7, y: 0.6, dur: 2.2 },
  { kind: "jump", x: 0.7, y: -0.7, dur: 2.2 },
  { kind: "jump", x: 0.6, y: 0.7, dur: 2 },
  { kind: "jump", x: 0.6, y: -0.75, dur: 2.2 },
  { kind: "orbit", dur: 6 },
  { kind: "jump", x: -0.75, y: -0.6, dur: 2 },
  { kind: "jump", x: -0.75, y: 0.75, dur: 2 },
  { kind: "jump", x: 0.75, y: 0.0, dur: 2 },
  { kind: "eight", dur: 7 },
  { kind: "jump", x: 0, y: 0.8, dur: 2 },
  { kind: "jump", x: 0, y: -0.8, dur: 2.4 },
];
const tour = { i: 0, t: 0 };

function autopilotActive() {
  return simTime - pointer.lastMove > 6 && !pointer.down;
}

function updateTarget() {
  const [ex, ey] = extents(view);
  const ax = Math.min(ex - 1.2, SPEC.arena.half_w);
  const ay = Math.min(ey - 1.2, SPEC.arena.half_h);
  let x: number;
  let y: number;
  let vx = 0;
  let vy = 0;
  if (autopilotActive()) {
    tour.t += DT;
    let leg = TOUR[tour.i];
    if (tour.t > leg.dur) {
      tour.t = 0;
      tour.i = (tour.i + 1) % TOUR.length;
      leg = TOUR[tour.i];
    }
    if (leg.kind === "jump") {
      x = leg.x * ax;
      y = leg.y * ay;
    } else {
      const w = leg.kind === "orbit" ? 1.6 : 1.1;
      const a = tour.t * w;
      const r = Math.min(ax, ay) * 0.6;
      if (leg.kind === "orbit") {
        x = r * Math.cos(a);
        y = r * Math.sin(a);
        vx = -r * w * Math.sin(a);
        vy = r * w * Math.cos(a);
      } else {
        x = 1.6 * r * Math.sin(a);
        y = r * Math.sin(2 * a) * 0.6;
        vx = 1.6 * r * w * Math.cos(a);
        vy = 1.2 * r * w * Math.cos(2 * a);
      }
    }
    target.vx = vx;
    target.vy = vy;
    target.auto = true;
  } else {
    x = pointer.x;
    y = pointer.y;
    const ix = (x - target.px) / DT;
    const iy = (y - target.py) / DT;
    if (target.auto || Math.hypot(ix, iy) > 30) {
      // A teleport (tap elsewhere, autopilot handoff) is a jump target in training terms: zero velocity.
      target.vx = target.vy = 0;
    } else {
      // Smoothed pointer velocity, the way a target moving at constant speed looks in training.
      const k = 0.25;
      target.vx += k * (ix - target.vx);
      target.vy += k * (iy - target.vy);
      const sp = Math.hypot(target.vx, target.vy);
      if (sp > 20) {
        target.vx *= 20 / sp;
        target.vy *= 20 / sp;
      }
    }
    target.auto = false;
  }
  target.px = x;
  target.py = y;
  target.x = x;
  target.y = y;
}

// ---------- simulation ----------

function stepFlyer(f: Flyer, windX: number, windY: number) {
  const d = f.drone;
  f.prev = pose(d);
  d.step(target.x + f.ox, target.y + f.oy, target.vx, target.vy, windX, windY);
  const p = d.phys;
  if (grab && grab.f === f) {
    const c = Math.cos(p.th);
    const s = Math.sin(p.th);
    const rx = c * grab.ox - s * grab.oy;
    const ry = s * grab.ox + c * grab.oy;
    const avx = p.vx - p.om * ry;
    const avy = p.vy + p.om * rx;
    const m = d.params.mass;
    let fx = m * (140 * (pointer.x - (p.x + rx)) - 9 * avx);
    let fy = m * (140 * (pointer.y - (p.y + ry)) - 9 * avy) + m * 9.81 * 0.6;
    const mag = Math.hypot(fx, fy);
    if (mag > 250) {
      fx *= 250 / mag;
      fy *= 250 / mag;
    }
    p.vx += (fx / m) * DT;
    p.vy += (fy / m) * DT;
    p.om += ((rx * fy - ry * fx) / d.params.inertia) * DT * 0.15;
    p.om = Math.max(-40, Math.min(40, p.om));
  }
  const dth = wrapAngle(p.th - f.lastTh);
  f.unwrapped += dth;
  f.lastTh = p.th;
  if (settings.trail) f.trail.push(p.x, p.y, Math.hypot(p.vx, p.vy));
  const [ex, ey] = extents(view);
  if (!Number.isFinite(p.x + p.y + p.th) || Math.abs(p.x) > ex + 25 || Math.abs(p.y) > ey + 25) {
    d.reset(target.x + f.ox, ey + 1);
    f.trail.clear();
    f.prev = pose(d);
    f.unwrapped = 0;
    f.lastTh = 0;
  }
}

function simStep() {
  simTime += DT;
  updateTarget();
  const gt = SPEC.dr.gust_tau;
  const gs = settings.gusts ? 2.2 : 0;
  const n = () => Math.sqrt(-2 * Math.log(Math.random() || 1e-9)) * Math.cos(2 * Math.PI * Math.random());
  gust.x += -gust.x / gt * DT + gs * Math.sqrt((2 * DT) / gt) * n();
  gust.y += -gust.y / gt * DT + gs * Math.sqrt((2 * DT) / gt) * n();
  const wx = settings.wind + gust.x;
  const wy = gust.y;
  const list = settings.mode === "fly" ? flyers : evolvers;
  for (const f of list) stepFlyer(f, wx, wy);
}

function kick() {
  const list = settings.mode === "fly" ? flyers : evolvers;
  for (const f of list) {
    const p = f.drone.phys;
    const a = Math.random() * Math.PI * 2;
    p.vx += Math.cos(a) * 9;
    p.vy += Math.sin(a) * 9;
    p.om += (Math.random() < 0.5 ? -1 : 1) * (20 + Math.random() * 15);
  }
}

// ---------- plan preview ----------

const ghost = new Drone(pd);
const PLAN_N = 70;
const planX = new Float32Array(PLAN_N);
const planY = new Float32Array(PLAN_N);

function computePlan(f: Flyer) {
  const d = f.drone;
  ghost.brain = d.brain;
  ghost.params = d.params;
  Object.assign(ghost.phys, d.phys);
  ghost.past.set(d.past);
  for (let k = 0; k < PLAN_N; k++) {
    ghost.step(target.x + f.ox, target.y + f.oy, target.vx, target.vy, settings.wind, 0);
    planX[k] = ghost.phys.x;
    planY[k] = ghost.phys.y;
  }
}

// ---------- evolution ----------

const evo = {
  pool: null as EvoPool | null,
  running: false,
  gen: 0,
  val: 0,
  series: [] as { gen: number; val: number }[],
  refs: [] as { label: string; value: number; color: string }[],
  theta: null as Float32Array | null,
  started: 0,
  gens0: 0,
  saved: 0,
};

function ensurePool() {
  if (evo.pool) return evo.pool;
  const pool = new EvoPool();
  pool.onRefs = (pdScore, ppo) => {
    evo.refs = [{ label: "classic PD", value: pdScore, color: `rgb(${ORANGE})` }];
    if (ppo !== null) evo.refs.push({ label: "PPO", value: ppo, color: `rgb(${CYAN})` });
  };
  pool.onGen = (u) => {
    evo.gen = u.gen;
    evo.theta = u.theta;
    if (u.val !== null) {
      evo.val = u.val;
      evo.series.push({ gen: u.gen, val: u.val });
      if (evo.series.length > 600) evo.series = evo.series.filter((_, i) => i % 2 === 0);
      setEvolvedBrains(u.theta, u.pop);
    }
    if (performance.now() - evo.saved > 2000) {
      evo.saved = performance.now();
      localStorage.setItem("bicopter.evolved", JSON.stringify({ gen: u.gen, theta: Array.from(u.theta), val: evo.val }));
    }
  };
  evo.pool = pool;
  return pool;
}

function setEvolvedBrains(theta: Float32Array, pop: Float32Array[]) {
  if (evolvers.length === 0) {
    evolvers.push(makeFlyer(pd, 0, 0, GREEN, undefined, 1));
    for (let i = 0; i < 12; i++) evolvers.push(makeFlyer(pd, (Math.random() - 0.5) * 6, (Math.random() - 0.5) * 4, GREEN, undefined, 0.28));
    applyPhysics();
  }
  evolvers[0].drone.brain = MLP.fromFlat("evolved", ES_SIZES, theta);
  pop.slice(0, evolvers.length - 1).forEach((p, i) => (evolvers[i + 1].drone.brain = MLP.fromFlat("mutant", ES_SIZES, p)));
}

async function startEvolution(fresh: boolean) {
  const pool = ensurePool();
  const saved = localStorage.getItem("bicopter.evolved");
  const seed = (Math.random() * 1e6) | 0;
  evo.series = [];
  if (fresh || !saved) {
    pool.reset(seed);
    evo.gen = 0;
  } else {
    const s = JSON.parse(saved);
    pool.reset(seed, Float32Array.from(s.theta), s.gen);
    evo.gen = s.gen;
  }
  resumeEvolution();
  const ppoId = manifest.default;
  if (manifest.brains.some((b) => b.id === ppoId) && !brainFiles.has(ppoId)) await loadBrain(ppoId);
  pool.refs(brainFiles.get(ppoId) ?? null);
}

function resumeEvolution() {
  evo.pool?.start();
  evo.running = true;
  evo.started = performance.now();
  evo.gens0 = evo.gen;
  syncEvoButtons();
}

function registerEvolved(theta: number[] | Float32Array, gen: number) {
  const b = MLP.fromFlat(`Evolved · gen ${gen}`, ES_SIZES, Float32Array.from(theta));
  brains.set("evolved", b);
  brainInfo.set("evolved", `Evolved in your browser for ${gen} generations with an evolution strategy. 24×24 network, no gradients.`);
  const sel = document.getElementById("brain") as HTMLSelectElement;
  let opt = sel.querySelector<HTMLOptionElement>('option[value="evolved"]');
  if (!opt) {
    opt = document.createElement("option");
    opt.value = "evolved";
    sel.appendChild(opt);
  }
  opt.textContent = b.name;
}

// ---------- UI ----------

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

interface SliderDef {
  key: "mass" | "thrust" | "imbalance" | "wind" | "lag" | "latency";
  label: string;
  min: number;
  max: number;
  step: number;
  train: [number, number];
  fmt: (v: number) => string;
}

const SLIDERS: SliderDef[] = [
  { key: "mass", label: "Mass", min: 0.5, max: 2.2, step: 0.01, train: SPEC.dr.mass as [number, number], fmt: (v) => `${v.toFixed(2)} kg` },
  { key: "thrust", label: "Motor power", min: 0.6, max: 1.4, step: 0.01, train: SPEC.dr.thrust_scale as [number, number], fmt: (v) => `${Math.round(v * 100)}%` },
  { key: "imbalance", label: "Motor imbalance", min: -0.3, max: 0.3, step: 0.01, train: [-0.1, 0.1], fmt: (v) => `${v > 0 ? "+" : ""}${Math.round(v * 100)}%` },
  { key: "wind", label: "Wind", min: -8, max: 8, step: 0.1, train: [-SPEC.dr.wind_max, SPEC.dr.wind_max], fmt: (v) => `${v.toFixed(1)} N` },
  { key: "lag", label: "Motor lag", min: 0.005, max: 0.15, step: 0.005, train: SPEC.dr.motor_tau as [number, number], fmt: (v) => `${Math.round(v * 1000)} ms` },
  { key: "latency", label: "Control latency", min: 0, max: 6, step: 1, train: [0, SPEC.max_latency], fmt: (v) => `${Math.round(v * DT * 1000)} ms` },
];

function buildSliders() {
  const box = $("sliders");
  box.innerHTML = "";
  for (const s of SLIDERS) {
    const el = document.createElement("div");
    el.className = "slider";
    const pct = (v: number) => `${((v - s.min) / (s.max - s.min)) * 100}%`;
    el.innerHTML = `<div class="top"><span>${s.label}</span><output></output></div><input type="range" min="${s.min}" max="${s.max}" step="${s.step}">`;
    const input = el.querySelector("input")!;
    const out = el.querySelector("output")!;
    input.style.setProperty("--lo", pct(s.train[0]));
    input.style.setProperty("--hi", pct(s.train[1]));
    input.value = String(settings[s.key]);
    const sync = () => {
      const v = Number(input.value);
      settings[s.key] = v;
      out.textContent = s.fmt(v);
      out.classList.toggle("out", v < s.train[0] - 1e-9 || v > s.train[1] + 1e-9);
      out.title = out.classList.contains("out") ? "outside what the network saw in training" : "";
      applyPhysics();
    };
    input.addEventListener("input", sync);
    sync();
    box.appendChild(el);
  }
}

function setMode(mode: "fly" | "evolve") {
  settings.mode = mode;
  document.querySelectorAll<HTMLButtonElement>(".tabs button").forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
  $("fly-panel").hidden = mode !== "fly";
  $("evolve-panel").hidden = mode !== "evolve";
  $("hint").textContent =
    mode === "fly"
      ? `A neural net flying a drone with two fixed motors. ${coarse ? "Touch to steer" : "Move your mouse"}. Grab the drone and throw it.`
      : "Watch a pilot evolve from nothing. Every few seconds it gets a little less bad.";
  if (mode === "evolve") {
    // Restore from storage only on first entry; afterwards the in-memory run simply continues.
    if (!evo.pool) void startEvolution(false);
    else if (!evo.running) resumeEvolution();
  } else if (evo.running) {
    evo.pool?.pause();
    evo.running = false;
  }
  syncEvoButtons();
}

function syncEvoButtons() {
  $("evo-run").textContent = evo.running ? "Pause" : "Resume";
}

function initUI() {
  const sel = $<HTMLSelectElement>("brain");
  for (const b of manifest.brains) {
    const o = document.createElement("option");
    o.value = b.id;
    o.textContent = b.label;
    sel.appendChild(o);
    brainInfo.set(b.id, b.info);
  }
  const o = document.createElement("option");
  o.value = "pd";
  o.textContent = "Classic PD controller";
  sel.appendChild(o);
  const saved = localStorage.getItem("bicopter.evolved");
  if (saved) {
    const s = JSON.parse(saved);
    registerEvolved(s.theta, s.gen);
  }
  sel.value = settings.brainId;
  const syncInfo = () => ($("brain-info").textContent = brainInfo.get(settings.brainId) ?? "");
  syncInfo();
  sel.addEventListener("change", async () => {
    settings.brainId = sel.value;
    syncInfo();
    await rebuildFlyers();
  });

  document.querySelectorAll<HTMLButtonElement>("#swarm button").forEach((b) =>
    b.addEventListener("click", async () => {
      settings.swarm = Number(b.dataset.n);
      document.querySelectorAll("#swarm button").forEach((x) => x.classList.toggle("on", x === b));
      await rebuildFlyers();
    }),
  );
  $<HTMLInputElement>("race").addEventListener("change", async (e) => {
    settings.race = (e.target as HTMLInputElement).checked;
    await rebuildFlyers();
  });
  $<HTMLInputElement>("gusts").addEventListener("change", (e) => (settings.gusts = (e.target as HTMLInputElement).checked));
  const bindCheck = (id: string, key: "plan" | "trail" | "slow" | "net") => {
    const el = $<HTMLInputElement>(id);
    el.checked = settings[key];
    el.addEventListener("change", () => {
      settings[key] = el.checked;
      if (key === "net") $("net").hidden = !el.checked;
      if (key === "trail") for (const f of [...flyers, ...evolvers]) f.trail.clear();
    });
  };
  bindCheck("v-plan", "plan");
  bindCheck("v-trail", "trail");
  bindCheck("v-slow", "slow");
  bindCheck("v-net", "net");
  $("kick").addEventListener("click", kick);
  $("reset-phys").addEventListener("click", () => {
    Object.assign(settings, { mass: 1, thrust: 1, imbalance: 0, wind: 0, lag: SPEC.motor_tau, latency: 0, gusts: false });
    $<HTMLInputElement>("gusts").checked = false;
    buildSliders();
  });
  buildSliders();

  document.querySelectorAll<HTMLButtonElement>(".tabs button").forEach((b) =>
    b.addEventListener("click", () => setMode(b.dataset.mode as "fly" | "evolve")),
  );
  $("evo-run").addEventListener("click", () => {
    if (evo.running) {
      evo.pool?.pause();
      evo.running = false;
    } else resumeEvolution();
    syncEvoButtons();
  });
  $("evo-reset").addEventListener("click", () => {
    localStorage.removeItem("bicopter.evolved");
    void startEvolution(true);
    syncEvoButtons();
  });
  $("evo-fly").addEventListener("click", async () => {
    if (!evo.theta) return;
    registerEvolved(evo.theta, evo.gen);
    settings.brainId = "evolved";
    sel.value = "evolved";
    syncInfo();
    setMode("fly");
    await rebuildFlyers();
  });
  $("panel-toggle").addEventListener("click", () => $("panel").classList.toggle("open"));

  window.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
    const k = e.key.toLowerCase();
    if (k === "k") kick();
    else if (k === "s") {
      settings.slow = !settings.slow;
      $<HTMLInputElement>("v-slow").checked = settings.slow;
    } else if (k === "a") pointer.lastMove = autopilotActive() ? simTime : -1e9;
  });
}

// ---------- input ----------

function pointerWorld(e: PointerEvent): [number, number] {
  const r = canvas.getBoundingClientRect();
  return toWorld(view, e.clientX - r.left, e.clientY - r.top);
}

function hitFlyer(x: number, y: number): Flyer | null {
  const list = settings.mode === "fly" ? flyers : evolvers.slice(0, 1);
  let best: Flyer | null = null;
  let bd = 0.7;
  for (const f of list) {
    const d = Math.hypot(f.drone.phys.x - x, f.drone.phys.y - y);
    if (d < bd) {
      bd = d;
      best = f;
    }
  }
  return best;
}

canvas.addEventListener("pointermove", (e) => {
  [pointer.x, pointer.y] = pointerWorld(e);
  pointer.touch = e.pointerType !== "mouse";
  if (e.pointerType === "mouse" || pointer.down) pointer.lastMove = simTime;
  if (!grab && e.pointerType === "mouse") canvas.className = hitFlyer(pointer.x, pointer.y) ? "grab" : "";
});
canvas.addEventListener("pointerdown", (e) => {
  [pointer.x, pointer.y] = pointerWorld(e);
  pointer.down = true;
  pointer.lastMove = simTime;
  try {
    canvas.setPointerCapture(e.pointerId);
  } catch {
    // Synthetic events have no capturable pointer.
  }
  const f = hitFlyer(pointer.x, pointer.y);
  if (f) {
    const p = f.drone.phys;
    const dx = pointer.x - p.x;
    const dy = pointer.y - p.y;
    const c = Math.cos(p.th);
    const s = Math.sin(p.th);
    grab = { f, ox: c * dx + s * dy, oy: -s * dx + c * dy };
    canvas.className = "grabbing";
  }
});
const release = () => {
  pointer.down = false;
  grab = null;
  canvas.className = "";
};
canvas.addEventListener("pointerup", release);
canvas.addEventListener("pointercancel", release);

function resize() {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.round(innerWidth * dpr);
  canvas.height = Math.round(innerHeight * dpr);
  view = makeView(innerWidth, innerHeight, dpr, panelInset());
}
window.addEventListener("resize", resize);

// ---------- frame ----------

let acc = 0;
let last = performance.now();
let statT = 0;
const netCanvas = $<HTMLCanvasElement>("net");
const netCtx = netCanvas.getContext("2d")!;
const chartCtx = $<HTMLCanvasElement>("chart").getContext("2d")!;

function lerpPose(a: Pose, b: Pose, t: number): Pose {
  return {
    x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t,
    th: a.th + wrapAngle(b.th - a.th) * t,
    fl: b.fl,
    fr: b.fr,
  };
}

function frame(now: number) {
  const real = Math.min(0.1, (now - last) / 1000);
  last = now;
  acc += real * (settings.slow ? 0.25 : 1);
  let steps = 0;
  while (acc >= DT && steps < 12) {
    simStep();
    acc -= DT;
    steps++;
  }
  if (steps === 12) acc = 0;
  const alpha = acc / DT;
  const t = simTime;

  ctx.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
  drawBackground(ctx, view, t, settings.wind + gust.x, windParticles);
  const list = settings.mode === "fly" ? flyers : evolvers;
  const main = list[0];
  if (settings.mode === "fly" && settings.plan && main) {
    computePlan(main);
    drawPlan(ctx, view, planX, planY, PLAN_N, main.rgb);
  }
  if (settings.trail) for (const f of list) drawTrail(ctx, view, f.trail, f.rgb);
  const auto = autopilotActive();
  const tColor = settings.mode === "fly" ? "rgba(255,255,255,0.85)" : `rgba(${GREEN},0.85)`;
  drawTarget(ctx, view, target.x, target.y, t, tColor, auto);
  if (settings.mode === "fly" && flyers.length > 1)
    for (const f of flyers) {
      const [sx, sy] = [view.cx + (target.x + f.ox) * view.ppm, view.cy - (target.y + f.oy) * view.ppm];
      ctx.fillStyle = "rgba(255,255,255,0.18)";
      ctx.fillRect(sx - 1.5, sy - 1.5, 3, 3);
    }
  // Dim drones first so the main one draws on top.
  for (let i = list.length - 1; i >= 0; i--) {
    const f = list[i];
    const p = lerpPose(f.prev, pose(f.drone), alpha);
    drawDrone(ctx, view, p, f.rgb, t, { alpha: f.alpha, glow: i === 0 || f.rgb === ORANGE, label: f.label });
    drawOffscreen(ctx, view, p.x, p.y, f.rgb);
  }

  if (now - statT > 100) {
    statT = now;
    updateStats(main);
  }
  if (settings.mode === "fly" && settings.net && main && main.drone.brain instanceof MLP) {
    const b = main.drone.brain;
    // The plan rollout overwrote activations; recompute for the live observation.
    b.forward(main.drone.obs);
    drawNet(netCtx, netCanvas.width, netCanvas.height, main.drone.obs, b.activations, main.drone.action);
  } else if (settings.mode === "fly" && settings.net) {
    netCtx.clearRect(0, 0, netCanvas.width, netCanvas.height);
    netCtx.fillStyle = "rgba(160,180,220,0.5)";
    netCtx.font = "11px ui-monospace, monospace";
    netCtx.fillText("no network: hand-written controller", 12, 64);
  }
  if (settings.mode === "evolve") drawChart(chartCtx, 248, 130, evo.series, evo.refs);
  requestAnimationFrame(frame);
}

function updateStats(main?: Flyer) {
  if (!main) return;
  const p = main.drone.phys;
  const speed = Math.hypot(p.vx, p.vy);
  const dist = Math.hypot(target.x + main.ox - p.x, target.y + main.oy - p.y);
  const flips = Math.floor(Math.abs(main.unwrapped) / (2 * Math.PI));
  const tilt = (p.th * 180) / Math.PI;
  let s =
    `speed  <b>${speed.toFixed(1).padStart(5)}</b> m/s\n` +
    `off    <b>${dist.toFixed(2).padStart(5)}</b> m\n` +
    `tilt   <b>${tilt.toFixed(0).padStart(5)}</b>°\n` +
    `flips  <b>${String(flips).padStart(5)}</b>`;
  if (settings.mode === "evolve") {
    const el = (performance.now() - evo.started) / 1000;
    const rate = evo.running && el > 1 ? (evo.gen - evo.gens0) / el : 0;
    $("evo-stats").textContent =
      `generation  ${evo.gen}\n` +
      `score       ${evo.val.toFixed(3)}\n` +
      (evo.refs.length ? evo.refs.map((r) => `${r.label.padEnd(12)}${r.value.toFixed(3)}`).join("\n") + "\n" : "") +
      `speed       ${rate.toFixed(1)} gen/s on ${evo.pool?.workers.length ?? 0} cores`;
    s = `generation <b>${String(evo.gen).padStart(5)}</b>\n` + s;
  }
  $("stats").innerHTML = s;
}

// ---------- boot ----------

async function boot() {
  resize();
  try {
    manifest = await (await fetch("/brains/manifest.json")).json();
    settings.brainId = manifest.default;
  } catch {
    settings.brainId = "pd";
  }
  initUI();
  setMode("fly");
  await rebuildFlyers();
  flyers[0]?.drone.reset(0, 0);
  requestAnimationFrame(frame);
  // Handle for poking at the sim from the devtools console.
  Object.assign(window, { bicopter: { get flyers() { return flyers; }, get view() { return view; }, settings, target } });
}

void boot();
