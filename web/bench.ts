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

type Settle = { t: number; maxTilt: number };
const mean = (xs: Settle[]): Settle => ({
  t: xs.reduce((a, x) => a + x.t, 0) / xs.length,
  maxTilt: Math.max(...xs.map((x) => x.maxTilt)),
});

/** Straight drops of 6 to 11 m. */
function dive(b: Brain) {
  return mean([6, 8, 10, 11].map((h) => settleTime(new Drone(b, 0, h / 2), 0, -h / 2)));
}

/** Corner to corner, both diagonals and directions. */
function across(b: Brain) {
  return mean(
    [
      [-8, -4, 8, 4],
      [8, 4, -8, -4],
      [-8, 4, 8, -4],
      [8, -4, -8, 4],
    ].map(([x0, y0, x1, y1]) => settleTime(new Drone(b, x0, y0), x1, y1)),
  );
}

/** Thrown while spinning, at a range of angles and spin rates. */
function recover(b: Brain) {
  const runs: Settle[] = [];
  for (const th of [Math.PI * 0.95, Math.PI / 2, -Math.PI * 0.7])
    for (const om of [-25, 15, 30]) {
      const d = new Drone(b, 0, 0);
      d.phys.th = th;
      d.phys.om = om;
      d.phys.vx = 4;
      runs.push(settleTime(d, 0, 0));
    }
  return mean(runs);
}

/** Mean distance while chasing circles of different radius and speed. */
function track(b: Brain) {
  let total = 0;
  const cases = [
    [4, 1.2],
    [2, 2.0],
    [6, 0.8],
  ];
  for (const [r, w] of cases) {
    const d = new Drone(b, r, 0);
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
    total += sum / n / cases.length;
  }
  return total;
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

const fmtT = (x: Settle) => `${Number.isFinite(x.t) ? x.t.toFixed(2) + "s" : "  fail"}${x.maxTilt > 2.1 ? "*" : " "}`;

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
console.log("score/robust: mean reward per step on 24 fixed jump courses (robust = heavy, laggy, weak motor).");
console.log("dive/across/recover: mean seconds to settle within 0.3 m (fail = any run never settled). * = flipped past 120°.");
console.log("track: mean metres behind circling targets. wind: mean metres off target in 3 N wind with gusts.");
