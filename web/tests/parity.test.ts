import { expect, test } from "bun:test";
import brainFile from "./fixtures/brain.json";
import physics from "./fixtures/physics.json";
import policy from "./fixtures/policy.json";
import { observe, OBS_DIM, Phys, physicsStep } from "../src/sim/physics";
import { type BrainFile, MLP } from "../src/sim/policy";

test("physics matches JAX step for step", () => {
  let worst = 0;
  for (const c of physics) {
    const p = new Phys();
    [p.x, p.y] = c.init.pos;
    [p.vx, p.vy] = c.init.vel;
    p.th = c.init.th;
    p.om = c.init.om;
    [p.fl, p.fr] = c.init.f;
    const obs = new Float32Array(OBS_DIM);
    c.actions.forEach((a, i) => {
      observe(p, c.target[0], c.target[1], c.target_vel[0], c.target_vel[1], a[0], a[1], obs);
      for (let j = 0; j < OBS_DIM; j++) worst = Math.max(worst, Math.abs(obs[j] - c.obs[i][j]));
      physicsStep(p, a[0], a[1], c.params, c.wind[0], c.wind[1]);
      const got = [p.x, p.y, p.vx, p.vy, p.th, p.om, p.fl, p.fr];
      got.forEach((v, j) => {
        let d = Math.abs(v - c.states[i][j]);
        if (j === 4) d = Math.min(d, Math.abs(d - 2 * Math.PI));
        worst = Math.max(worst, d);
      });
    });
  }
  expect(worst).toBeLessThan(1e-5);
});

test("policy forward matches flax actor", () => {
  const net = MLP.fromFile("fixture", brainFile as BrainFile);
  let worst = 0;
  policy.obs.forEach((o, i) => {
    const y = net.forward(Float32Array.from(o));
    worst = Math.max(worst, Math.abs(y[0] - policy.out[i][0]), Math.abs(y[1] - policy.out[i][1]));
  });
  expect(worst).toBeLessThan(1e-3);
});
