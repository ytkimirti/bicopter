import { expect, test } from "bun:test";
import { Drone } from "../src/sim/drone";
import type { Brain } from "../src/sim/policy";

// Brain whose t-th command is a distinct value, so the motor thrust reveals which command arrived.
class Counter implements Brain {
  name = "counter";
  t = 0;
  act(_o: Float32Array, _c: unknown, out: Float32Array) {
    out[0] = out[1] = -1 + 0.1 * this.t++;
  }
}

test("latency k delivers the command from k steps ago, like env.py hist[latency]", () => {
  for (const k of [0, 1, 2, 5]) {
    const b = new Counter();
    const d = new Drone(b);
    d.params.tau = 1e-12;
    d.params.latency = k;
    const seen: number[] = [];
    for (let t = 0; t < 10; t++) {
      d.step(0, 0, 0, 0, 0, 0);
      seen.push(Math.round(((2 * d.phys.fl) / d.params.tmaxL - 1 + 1) / 0.1));
    }
    // Before k steps have passed the buffer holds zero commands (index 10 = command 0.0).
    const want = Array.from({ length: 10 }, (_, t) => (t >= k ? t - k : 10));
    expect(seen).toEqual(want);
  }
});

test("the policy observes its previous command", () => {
  const b = new Counter();
  const d = new Drone(b);
  d.step(0, 0, 0, 0, 0, 0);
  d.step(0, 0, 0, 0, 0, 0);
  // obs is built before the second act(), so it holds the first command.
  expect(d.obs[13]).toBeCloseTo(-1, 5);
});
