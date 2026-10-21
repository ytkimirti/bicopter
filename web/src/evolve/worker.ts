/// <reference lib="webworker" />
// Stateless evaluator: the main thread owns the evolution and farms antithetic pairs out to a pool of these.
import { Drone } from "../sim/drone";
import { type Brain, type BrainFile, MLP, PID } from "../sim/policy";
import { defaultES, ES_SIZES, type ESConfig, fitness, makeScenarios, PairEvaluator } from "./es";

export type ToWorker =
  | { type: "eval"; id: number; theta: Float32Array; cfg: ESConfig; gen: number; from: number; to: number; val: boolean }
  | { type: "refs"; ppo: BrainFile | null };

export type FromWorker =
  | { type: "fit"; id: number; from: number; f: Float64Array; val: number | null }
  | { type: "refs"; pd: number; ppo: number | null };

const VAL = makeScenarios({ ...defaultES, episodes: 12 }, 424242);
export const valScore = (b: Brain) => fitness(b, VAL, defaultES.steps, new Drone(b));

let evaluator: PairEvaluator | null = null;

self.onmessage = (e: MessageEvent<ToWorker>) => {
  const m = e.data;
  if (m.type === "eval") {
    evaluator ??= new PairEvaluator(m.theta.length);
    const val = m.val ? valScore(MLP.fromFlat("center", ES_SIZES, m.theta)) : null;
    const f = evaluator.run(m.theta, m.cfg, m.gen, m.from, m.to);
    postMessage({ type: "fit", id: m.id, from: m.from, f, val } satisfies FromWorker);
  } else if (m.type === "refs") {
    postMessage({
      type: "refs",
      pd: valScore(new PID()),
      ppo: m.ppo ? valScore(MLP.fromFile("ppo", m.ppo)) : null,
    } satisfies FromWorker);
  }
};
