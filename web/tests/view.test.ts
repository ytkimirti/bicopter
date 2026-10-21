import { expect, test } from "bun:test";
import { extents, makeView, MIN_PPM } from "../src/render";
import { SPEC } from "../src/sim/physics";

const inset = (w: number) => (w > 720 ? 300 : 0); // main.ts panelInset

test("large windows show the whole training arena left of the panel", () => {
  for (const [w, h] of [[1000, 800], [1280, 800], [1440, 900], [1920, 1080], [3440, 1440]]) {
    const [ex, ey] = extents(makeView(w, h, 1, inset(w)));
    expect(ex).toBeGreaterThanOrEqual(SPEC.arena.half_w);
    expect(ey).toBeGreaterThanOrEqual(SPEC.arena.half_h);
  }
});

test("small screens keep a readable scale", () => {
  for (const [w, h] of [[390, 844], [375, 667], [768, 1024], [844, 390]]) {
    expect(makeView(w, h, 1, inset(w)).ppm).toBeGreaterThanOrEqual(MIN_PPM);
  }
});

test("scale changes continuously with window width (no snapping)", () => {
  // Within each panel layout, one pixel of width never changes the scale by more than a few percent.
  let prev = makeView(300, 800, 1, inset(300)).ppm;
  for (let w = 301; w <= 2400; w++) {
    const ppm = makeView(w, 800, 1, inset(w)).ppm;
    if (w !== 721) expect(Math.abs(ppm - prev) / prev).toBeLessThan(0.02);
    prev = ppm;
  }
});
