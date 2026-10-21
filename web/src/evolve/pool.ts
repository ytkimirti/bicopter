import type { BrainFile } from "../sim/policy";
import { defaultES, type ESConfig, Evolution } from "./es";
import type { FromWorker, ToWorker } from "./worker";

export interface GenUpdate {
  gen: number;
  val: number | null;
  mean: number;
  best: number;
  theta: Float32Array;
  pop: Float32Array[];
}

/** Runs the evolution on the main thread and evaluates each generation across every core. */
export class EvoPool {
  readonly workers: Worker[];
  evo: Evolution;
  running = false;
  onGen: (u: GenUpdate) => void = () => {};
  onRefs: (pd: number, ppo: number | null) => void = () => {};
  private id = 0;
  private pending = 0;
  private fit = new Float64Array(0);
  private val: number | null = null;
  private lastVal = 0;

  constructor(n = Math.max(1, Math.min(12, (navigator.hardwareConcurrency || 4) - 1))) {
    this.workers = Array.from({ length: n }, () => {
      const w = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
      w.onmessage = (e: MessageEvent<FromWorker>) => this.receive(e.data);
      return w;
    });
    this.evo = new Evolution(this.config(1));
  }

  /** Bigger populations when there are more cores to evaluate them. */
  private config(seed: number): ESConfig {
    const pop = this.workers.length >= 6 ? 128 : this.workers.length >= 3 ? 96 : 64;
    return { ...defaultES, pop, seed };
  }

  reset(seed: number, theta?: Float32Array, gen = 0) {
    this.evo = new Evolution(this.config(seed), theta);
    this.evo.gen = gen;
    this.id++;
    this.pending = 0;
    if (this.running) this.launch();
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.launch();
  }

  pause() {
    this.running = false;
  }

  refs(ppo: BrainFile | null) {
    this.workers[0].postMessage({ type: "refs", ppo } satisfies ToWorker);
  }

  private launch() {
    if (this.pending > 0) return;
    const { cfg, theta, gen } = this.evo;
    const pairs = cfg.pop / 2;
    const n = this.workers.length;
    this.fit = new Float64Array(cfg.pop);
    this.val = null;
    const now = performance.now();
    const wantVal = now - this.lastVal > 150;
    if (wantVal) this.lastVal = now;
    let from = 0;
    this.workers.forEach((w, i) => {
      const to = Math.round(((i + 1) * pairs) / n);
      if (to <= from) return;
      this.pending++;
      w.postMessage({ type: "eval", id: this.id, theta, cfg, gen, from, to, val: wantVal && i === 0 } satisfies ToWorker);
      from = to;
    });
  }

  private receive(m: FromWorker) {
    if (m.type === "refs") return this.onRefs(m.pd, m.ppo);
    if (m.id !== this.id) return;
    this.fit.set(m.f, 2 * m.from);
    if (m.val !== null) this.val = m.val;
    if (--this.pending > 0) return;
    const s = this.evo.update(this.fit);
    this.onGen({ gen: s.gen, val: this.val, mean: s.mean, best: s.best, theta: this.evo.theta, pop: this.evo.lastPop });
    if (this.running) this.launch();
  }
}
