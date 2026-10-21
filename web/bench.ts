// Fixed-scenario benchmark for brains, run headless: bun bench.ts public/brains/ppo-best.json [more.json...]
import { defaultES, fitness, makeScenarios } from "./src/evolve/es";
import { Drone } from "./src/sim/drone";
import { DT, SPEC } from "./src/sim/physics";
import { type Brain, type BrainFile, MLP, PID } from "./src/sim/policy";

const VAL = makeScenarios({ ...defaultES, episodes: 24 }, 424242);

function settleTime(d: Drone, tx: number, ty: number, maxT = 6, radius = 0.3, hold = 0.5) {
  let inside = 0;
  let maxTilt = 0;
  for (let t = 0; t < maxT / DT; t++) {
    d.step(tx, ty, 0, 0, 0, 0);
    maxTilt = Math.max(maxTilt, Math.abs(d.phys.th));
    inside = Math.hypot(d.phys.x - tx, d.phys.y - ty) < radius ? inside + 1 : 0;
    if (inside * DT >= hold) return { t: (t + 1) * DT - hold, maxTilt };
  }
  return { t: Infinity, maxTilt };
}

function dive(b: Brain) {
  const d = new Drone(b, 0, 5);
  return settleTime(d, 0, -5);
}

function across(b: Brain) {
  const d = new Drone(b, -8, -4);
  return settleTime(d, 8, 4);
}

function recover(b: Brain) {
  const d = new Drone(b, 0, 0);
  d.phys.th = Math.PI * 0.95;
  d.phys.om = 25;
  d.phys.vx = 4;
  return settleTime(d, 0, 0);
}

function track(b: Brain) {
  const d = new Drone(b, 4, 0);
  const w = 1.2;
  const r = 4;
  let sum = 0;
  let n = 0;
  for (let t = 0; t < 8 / DT; t++) {
    const a = t * DT * w;
    d.step(r * Math.cos(a), r * Math.sin(a), -r * w * Math.sin(a), r * w * Math.cos(a), 0, 0);
    if (t * DT > 2) {
      sum += Math.hypot(d.phys.x - r * Math.cos(a + w * DT), d.phys.y - r * Math.sin(a + w * DT));
      n++;
    }
  }
  return sum / n;
}

function windHold(b: Brain) {
  const d = new Drone(b, 0, 0);
  let gx = 0;
  let gy = 0;
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;
  let sum = 0;
  const steps = 6 / DT;
  for (let t = 0; t < steps; t++) {
    gx += -gx * DT / 0.6 + 1.5 * Math.sqrt((2 * DT) / 0.6) * rnd() * 1.7;
    gy += -gy * DT / 0.6 + 1.5 * Math.sqrt((2 * DT) / 0.6) * rnd() * 1.7;
    d.step(0, 0, 0, 0, 3 + gx, gy);
    sum += Math.hypot(d.phys.x, d.phys.y);
  }
  return sum / steps;
}

function robust(b: Brain) {
  const d = new Drone(b);
  d.params.mass = SPEC.mass * 1.3;
  d.params.inertia = SPEC.inertia * 1.3;
  d.params.latency = 2;
  d.params.tau = 0.06;
  d.params.tmaxL = SPEC.max_thrust * 0.92;
  return fitness(b, VAL, defaultES.steps, d);
}

const fmtT = (x: { t: number; maxTilt: number }) => `${Number.isFinite(x.t) ? x.t.toFixed(2) + "s" : "  fail"}${x.maxTilt > 2.1 ? "*" : " "}`;

const brains: Brain[] = [new PID()];
for (const p of process.argv.slice(2)) brains.push(MLP.fromFile(p.split("/").pop()!.replace(".json", ""), (await Bun.file(p).json()) as BrainFile));

console.log("brain".padEnd(22), "score  robust  dive    across  recover  track  wind");
for (const b of brains) {
  console.log(
    b.name.padEnd(22).slice(0, 22),
    fitness(b, VAL, defaultES.steps, new Drone(b)).toFixed(3),
    "",
    robust(b).toFixed(3),
    "",
    fmtT(dive(b)).padEnd(7),
    fmtT(across(b)).padEnd(7),
    fmtT(recover(b)).padEnd(8),
    track(b).toFixed(2).padStart(5),
    windHold(b).toFixed(2).padStart(5),
  );
}
console.log("score/robust: mean reward per step on 24 fixed jump courses (robust = heavy, laggy, weak motor). * = flipped past 120°.");
