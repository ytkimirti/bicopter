// OpenAI-style evolution strategy: antithetic Gaussian perturbations, centered ranks, Adam.
// Runs in a worker in the browser and headless under bun for tests.
import { Drone } from "../sim/drone";
import { reward } from "../sim/physics";
import { type Brain, MLP } from "../sim/policy";

export const ES_SIZES = [15, 24, 24, 2];

export interface ESConfig {
  pop: number; // even
  sigma: number;
  lr: number;
  episodes: number;
  steps: number;
  seed: number;
}

export const defaultES: ESConfig = { pop: 64, sigma: 0.06, lr: 0.03, episodes: 3, steps: 300, seed: 1 };

export interface GenStats {
  gen: number;
  best: number;
  mean: number;
  evalSteps: number;
  ms: number;
}

export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rand: () => number) {
  let u = 0;
  while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

interface Scenario {
  x: number;
  y: number;
  vx: number;
  vy: number;
  th: number;
  targets: Float64Array; // (steps/jump) pairs
}

const JUMP = 90;
const ESCAPE = 25;

export function makeScenarios(cfg: ESConfig, seed: number): Scenario[] {
  const r = mulberry32(seed);
  const u = (a: number, b: number) => a + (b - a) * r();
  return Array.from({ length: cfg.episodes }, () => {
    const n = Math.ceil(cfg.steps / JUMP);
    const targets = new Float64Array(n * 2);
    for (let i = 0; i < n; i++) {
      targets[2 * i] = u(-8, 8);
      targets[2 * i + 1] = u(-5, 5);
    }
    return { x: u(-6, 6), y: u(-4, 4), vx: u(-2, 2), vy: u(-2, 2), th: u(-0.6, 0.6), targets };
  });
}

/** Mean per-step reward of a policy over the scenarios. Escaping forfeits the rest of the episode. */
export function fitness(net: Brain, scenarios: Scenario[], steps: number, drone = new Drone(net)) {
  drone.brain = net;
  let total = 0;
  for (const s of scenarios) {
    drone.reset(s.x, s.y);
    drone.phys.vx = s.vx;
    drone.phys.vy = s.vy;
    drone.phys.th = s.th;
    for (let t = 0; t < steps; t++) {
      const j = Math.floor(t / JUMP);
      const tx = s.targets[2 * j];
      const ty = s.targets[2 * j + 1];
      const pl = drone.past[0];
      const pr = drone.past[1];
      drone.step(tx, ty, 0, 0, 0, 0);
      const p = drone.phys;
      if (!(Math.abs(p.x) < ESCAPE && Math.abs(p.y) < ESCAPE)) {
        total -= 10;
        break;
      }
      total += reward(p, tx, ty, drone.action[0], drone.action[1], pl, pr);
    }
  }
  return total / (scenarios.length * steps);
}

export function initParams(seed: number) {
  const r = mulberry32(seed);
  const theta = new Float32Array(MLP.flatSize(ES_SIZES));
  let o = 0;
  for (let i = 0; i + 1 < ES_SIZES.length; i++) {
    const nIn = ES_SIZES[i];
    const nOut = ES_SIZES[i + 1];
    const last = i + 2 === ES_SIZES.length;
    const scale = (last ? 0.1 : 1) / Math.sqrt(nIn);
    for (let k = 0; k < nIn * nOut; k++) theta[o++] = gaussian(r) * scale;
    // Output bias starts near hover thrust so generation zero does not just fall.
    for (let k = 0; k < nOut; k++) theta[o++] = last ? -0.43 : 0;
  }
  return theta;
}

/** Noise for pair i of a generation, regenerated from seeds so workers never ship noise vectors around. */
export function pairNoise(seed: number, gen: number, i: number, out: Float32Array) {
  const r = mulberry32((Math.imul(seed, 0x9e3779b1) ^ Math.imul(gen + 1, 0x85ebca6b) ^ Math.imul(i + 1, 0xc2b2ae35)) >>> 0);
  for (let k = 0; k < out.length; k++) out[k] = gaussian(r);
  return out;
}

export const scenarioSeed = (cfg: ESConfig, gen: number) => cfg.seed * 1_000_003 + gen;

/** Evaluates antithetic pairs [from, to) of a generation. Returns fitness as [+0, -0, +1, -1, ...]. */
export class PairEvaluator {
  private eps: Float32Array;
  private trial: Float32Array;
  private net: MLP;
  private drone: Drone;
  constructor(n: number) {
    this.eps = new Float32Array(n);
    this.trial = new Float32Array(n);
    this.net = MLP.fromFlat("trial", ES_SIZES, this.trial);
    this.drone = new Drone(this.net);
  }
  run(theta: Float32Array, cfg: ESConfig, gen: number, from: number, to: number) {
    const scen = makeScenarios(cfg, scenarioSeed(cfg, gen));
    const f = new Float64Array(2 * (to - from));
    for (let i = from; i < to; i++) {
      pairNoise(cfg.seed, gen, i, this.eps);
      for (let s = 0; s < 2; s++) {
        const sign = s === 0 ? cfg.sigma : -cfg.sigma;
        for (let k = 0; k < theta.length; k++) this.trial[k] = theta[k] + sign * this.eps[k];
        f[2 * (i - from) + s] = fitness(this.net, scen, cfg.steps, this.drone);
      }
    }
    return f;
  }
}

export class Evolution {
  theta: Float32Array;
  gen = 0;
  /** Adam's own step count: a resumed run restarts its moments, so bias correction must restart too. */
  private t = 0;
  private m: Float32Array;
  private v: Float32Array;
  private evaluator: PairEvaluator;
  /** A few perturbations from the last generation, for drawing the population. */
  lastPop: Float32Array[] = [];

  constructor(
    public cfg: ESConfig = defaultES,
    theta?: Float32Array,
  ) {
    this.theta = theta ?? initParams(cfg.seed);
    const n = this.theta.length;
    this.m = new Float32Array(n);
    this.v = new Float32Array(n);
    this.evaluator = new PairEvaluator(n);
  }

  centerNet(name = "evolved") {
    return MLP.fromFlat(name, ES_SIZES, Float32Array.from(this.theta));
  }

  /** Single-threaded generation, used headless. The browser splits run() across workers and calls update(). */
  generation(): GenStats {
    const t0 = performance.now();
    const f = this.evaluator.run(this.theta, this.cfg, this.gen, 0, this.cfg.pop / 2);
    return { ...this.update(f), ms: performance.now() - t0 };
  }

  update(f: Float64Array): GenStats {
    const { pop, sigma, lr } = this.cfg;
    const half = pop / 2;
    const n = this.theta.length;
    // Centered ranks in [-0.5, 0.5].
    const order = Array.from(f.keys()).sort((a, b) => f[a] - f[b]);
    const rank = new Float64Array(pop);
    order.forEach((idx, r) => (rank[idx] = r / (pop - 1) - 0.5));
    const g = new Float32Array(n);
    const e = new Float32Array(n);
    const shown: Float32Array[] = [];
    for (let i = 0; i < half; i++) {
      pairNoise(this.cfg.seed, this.gen, i, e);
      const w = rank[2 * i] - rank[2 * i + 1];
      for (let k = 0; k < n; k++) g[k] += w * e[k];
      if (i < 6) shown.push(Float32Array.from(e));
    }
    // Adam ascent with light weight decay.
    this.gen++;
    this.t++;
    const b1 = 0.9;
    const b2 = 0.999;
    const c1 = 1 - b1 ** this.t;
    const c2 = 1 - b2 ** this.t;
    for (let k = 0; k < n; k++) {
      const gk = g[k] / (pop * sigma) - 0.005 * this.theta[k];
      this.m[k] = b1 * this.m[k] + (1 - b1) * gk;
      this.v[k] = b2 * this.v[k] + (1 - b2) * gk * gk;
      this.theta[k] += (lr * this.m[k]) / c1 / (Math.sqrt(this.v[k] / c2) + 1e-8);
    }
    this.lastPop = shown.flatMap((e) => {
      const plus = new Float32Array(n);
      const minus = new Float32Array(n);
      for (let k = 0; k < n; k++) {
        plus[k] = this.theta[k] + sigma * e[k];
        minus[k] = this.theta[k] - sigma * e[k];
      }
      return [plus, minus];
    });
    let best = -Infinity;
    let mean = 0;
    for (const x of f) {
      best = Math.max(best, x);
      mean += x / pop;
    }
    return { gen: this.gen, best, mean, evalSteps: pop * this.cfg.episodes * this.cfg.steps, ms: 0 };
  }
}
