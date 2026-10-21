// Bicopter physics and observation. Mirrors train/bicopter/sim.py exactly; tests/parity.test.ts checks it.
import spec from "../../../spec.json";

export const SPEC = spec;
export const OBS_DIM = spec.obs.dim;
export const DT = spec.dt;

export interface Params {
  mass: number;
  inertia: number;
  tmaxL: number;
  tmaxR: number;
  tau: number;
  latency: number;
}

export const nominalParams = (): Params => ({
  mass: spec.mass,
  inertia: spec.inertia,
  tmaxL: spec.max_thrust,
  tmaxR: spec.max_thrust,
  tau: spec.motor_tau,
  latency: 0,
});

export class Phys {
  x = 0;
  y = 0;
  vx = 0;
  vy = 0;
  th = 0;
  om = 0;
  fl = 0;
  fr = 0;
}

const TAU = 2 * Math.PI;
export const wrapAngle = (th: number) => {
  const t = th + Math.PI;
  return t - TAU * Math.floor(t / TAU) - Math.PI;
};

export const clamp = (x: number, lo: number, hi: number) => (x < lo ? lo : x > hi ? hi : x);

/** Advance one control step. cmd in [-1, 1] maps to motor thrust 0..tmax. */
export function physicsStep(p: Phys, cmdL: number, cmdR: number, q: Params, windX: number, windY: number) {
  const targetL = clamp((cmdL + 1) * 0.5, 0, 1) * q.tmaxL;
  const targetR = clamp((cmdR + 1) * 0.5, 0, 1) * q.tmaxR;
  const h = spec.dt / spec.substeps;
  const alpha = h / (q.tau + h);
  const g = spec.gravity;
  const arm = spec.arm;
  for (let i = 0; i < spec.substeps; i++) {
    p.fl += (targetL - p.fl) * alpha;
    p.fr += (targetR - p.fr) * alpha;
    const thrust = p.fl + p.fr;
    const s = Math.sin(p.th);
    const c = Math.cos(p.th);
    const k = spec.drag_lin + spec.drag_quad * Math.sqrt(p.vx * p.vx + p.vy * p.vy);
    const ax = (-s * thrust + windX - k * p.vx) / q.mass;
    const ay = (c * thrust + windY - k * p.vy) / q.mass - g;
    const aa = ((p.fr - p.fl) * arm - spec.ang_drag * p.om) / q.inertia;
    p.vx += ax * h;
    p.vy += ay * h;
    p.x += p.vx * h;
    p.y += p.vy * h;
    p.om += aa * h;
    p.th += p.om * h;
  }
  p.th = wrapAngle(p.th);
}

const O = spec.obs;

export function observe(
  p: Phys,
  tx: number,
  ty: number,
  tvx: number,
  tvy: number,
  prevL: number,
  prevR: number,
  out: Float32Array,
) {
  const dx = tx - p.x;
  const dy = ty - p.y;
  out[0] = clamp(dx * O.pos_scale, -O.pos_clip, O.pos_clip);
  out[1] = clamp(dy * O.pos_scale, -O.pos_clip, O.pos_clip);
  out[2] = clamp(dx, -O.fine_clip, O.fine_clip);
  out[3] = clamp(dy, -O.fine_clip, O.fine_clip);
  out[4] = clamp(p.vx * O.vel_scale, -O.vel_clip, O.vel_clip);
  out[5] = clamp(p.vy * O.vel_scale, -O.vel_clip, O.vel_clip);
  out[6] = clamp(tvx * O.vel_scale, -O.vel_clip, O.vel_clip);
  out[7] = clamp(tvy * O.vel_scale, -O.vel_clip, O.vel_clip);
  out[8] = Math.sin(p.th);
  out[9] = Math.cos(p.th);
  out[10] = clamp(p.om * O.omega_scale, -O.omega_clip, O.omega_clip);
  out[11] = p.fl / spec.max_thrust;
  out[12] = p.fr / spec.max_thrust;
  out[13] = prevL;
  out[14] = prevR;
}

/** Per-step reward, same terms as train/bicopter/env.py. */
export function reward(p: Phys, tx: number, ty: number, aL: number, aR: number, prevL: number, prevR: number) {
  const R = spec.reward;
  const d = Math.hypot(tx - p.x, ty - p.y);
  const r = d / R.precise_radius;
  const n = d / R.near_radius;
  const near = Math.exp(-n * n);
  return (
    R.track_scale / (R.track_scale + d) +
    R.precise_weight * Math.exp(-r * r) +
    near * (R.upright_weight * Math.cos(p.th) - R.near_omega_weight * Math.min(p.om * p.om, 100)) -
    R.omega_weight * p.om * p.om -
    R.action_rate_weight * ((aL - prevL) ** 2 + (aR - prevR) ** 2)
  );
}
