import { type Brain, type Ctx } from "./policy";
import { nominalParams, observe, OBS_DIM, type Params, Phys, physicsStep } from "./physics";

/** A drone flown by a brain. Holds the action-latency buffer the training env uses. */
export class Drone {
  phys = new Phys();
  params: Params = nominalParams();
  obs = new Float32Array(OBS_DIM);
  action = new Float32Array(2);
  // a_{t-1} and a_{t-2}: the policy sees a_{t-1}; latency picks which command reaches the motors.
  prev0 = new Float32Array(2);
  prev1 = new Float32Array(2);
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
    observe(this.phys, tx, ty, tvx, tvy, this.prev0[0], this.prev0[1], this.obs);
    this.brain.act(this.obs, c, this.action);
    const lat = this.params.latency;
    const src = lat === 0 ? this.action : lat === 1 ? this.prev0 : this.prev1;
    const cl = src[0];
    const cr = src[1];
    this.prev1.set(this.prev0);
    this.prev0.set(this.action);
    physicsStep(this.phys, cl, cr, this.params, windX, windY);
  }

  reset(x: number, y: number) {
    const p = this.phys;
    p.x = x;
    p.y = y;
    p.vx = p.vy = p.th = p.om = 0;
    p.fl = p.fr = (this.params.mass * 9.81) / 2;
    this.prev0.fill(0);
    this.prev1.fill(0);
  }
}
