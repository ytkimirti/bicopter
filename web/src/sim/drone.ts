import { type Brain, type Ctx } from "./policy";
import { nominalParams, observe, OBS_DIM, type Params, Phys, physicsStep } from "./physics";

export const MAX_LATENCY = 8;

/** A drone flown by a brain. Holds the action-latency buffer the training env uses. */
export class Drone {
  phys = new Phys();
  params: Params = nominalParams();
  obs = new Float32Array(OBS_DIM);
  action = new Float32Array(2);
  /** Past commands, newest first: [a_{t-1}, a_{t-2}, ...] as (L, R) pairs. The policy sees a_{t-1};
   * latency k sends a_{t-k} to the motors, as hist[latency] does in train/bicopter/env.py. */
  past = new Float32Array(2 * MAX_LATENCY);
  ctx: Ctx;

  constructor(
    public brain: Brain,
    x = 0,
    y = 0,
  ) {
    this.phys.x = x;
    this.phys.y = y;
    const hover = (this.params.mass * 9.81) / 2;
    this.phys.fl = this.phys.fr = hover;
    this.ctx = { phys: this.phys, params: this.params, tx: 0, ty: 0, tvx: 0, tvy: 0 };
  }

  step(tx: number, ty: number, tvx: number, tvy: number, windX: number, windY: number) {
    const c = this.ctx;
    c.params = this.params;
    c.tx = tx;
    c.ty = ty;
    c.tvx = tvx;
    c.tvy = tvy;
    const past = this.past;
    observe(this.phys, tx, ty, tvx, tvy, past[0], past[1], this.obs);
    this.brain.act(this.obs, c, this.action);
    const lat = Math.max(0, Math.min(MAX_LATENCY, Math.round(this.params.latency)));
    const cl = lat === 0 ? this.action[0] : past[2 * (lat - 1)];
    const cr = lat === 0 ? this.action[1] : past[2 * (lat - 1) + 1];
    past.copyWithin(2, 0, past.length - 2);
    past[0] = this.action[0];
    past[1] = this.action[1];
    physicsStep(this.phys, cl, cr, this.params, windX, windY);
  }

  reset(x: number, y: number) {
    const p = this.phys;
    p.x = x;
    p.y = y;
    p.vx = p.vy = p.th = p.om = 0;
    p.fl = p.fr = (this.params.mass * 9.81) / 2;
    this.past.fill(0);
  }
}
