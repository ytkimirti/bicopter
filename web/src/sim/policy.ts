import { clamp, OBS_DIM, type Phys, type Params, SPEC, wrapAngle } from "./physics";

export interface BrainFile {
  kind: "mlp";
  activation: "tanh";
  obs_dim: number;
  steps: number;
  hidden: number[];
  eval?: Record<string, number>;
  layers: { in: number; out: number; w: number[] | string; b: number[] | string }[];
}

/** Weights are base64 little-endian float32 (or plain arrays in older files). */
function decode(x: number[] | string): Float32Array {
  if (typeof x !== "string") return Float32Array.from(x);
  const bin = atob(x);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const view = new DataView(bytes.buffer);
  const out = new Float32Array(bytes.length / 4);
  for (let i = 0; i < out.length; i++) out[i] = view.getFloat32(4 * i, true);
  return out;
}

export interface Ctx {
  phys: Phys;
  params: Params;
  tx: number;
  ty: number;
  tvx: number;
  tvy: number;
}

export interface Brain {
  readonly name: string;
  /** Writes two commands in [-1, 1] to out. */
  act(obs: Float32Array, ctx: Ctx, out: Float32Array): void;
  /** Hidden-layer activations of the last act() call, for visualization. */
  activations?: Float32Array[];
}

interface Layer {
  nIn: number;
  nOut: number;
  w: Float32Array; // row-major (nIn, nOut)
  b: Float32Array;
}

/** tanh MLP, the actor exported by train/bicopter/export.py or evolved in the browser. */
export class MLP implements Brain {
  readonly layers: Layer[];
  readonly activations: Float32Array[];

  constructor(
    readonly name: string,
    layers: Layer[],
  ) {
    this.layers = layers;
    this.activations = layers.map((l) => new Float32Array(l.nOut));
  }

  static fromFile(name: string, f: BrainFile) {
    if (f.obs_dim !== OBS_DIM) throw new Error(`brain expects obs_dim ${f.obs_dim}, sim has ${OBS_DIM}`);
    return new MLP(
      name,
      f.layers.map((l) => ({ nIn: l.in, nOut: l.out, w: decode(l.w), b: decode(l.b) })),
    );
  }

  /** Layer views over one flat parameter vector, the layout used by the evolution worker. */
  static fromFlat(name: string, sizes: number[], flat: Float32Array) {
    const layers: Layer[] = [];
    let o = 0;
    for (let i = 0; i + 1 < sizes.length; i++) {
      const nIn = sizes[i];
      const nOut = sizes[i + 1];
      const w = flat.subarray(o, o + nIn * nOut);
      o += nIn * nOut;
      const b = flat.subarray(o, o + nOut);
      o += nOut;
      layers.push({ nIn, nOut, w, b });
    }
    return new MLP(name, layers);
  }

  static flatSize(sizes: number[]) {
    let n = 0;
    for (let i = 0; i + 1 < sizes.length; i++) n += sizes[i] * sizes[i + 1] + sizes[i + 1];
    return n;
  }

  forward(x: Float32Array): Float32Array {
    let input = x;
    const last = this.layers.length - 1;
    for (let li = 0; li <= last; li++) {
      const { nIn, nOut, w, b } = this.layers[li];
      const y = this.activations[li];
      for (let j = 0; j < nOut; j++) y[j] = b[j];
      for (let i = 0; i < nIn; i++) {
        const xi = input[i];
        if (xi === 0) continue;
        const row = i * nOut;
        for (let j = 0; j < nOut; j++) y[j] += xi * w[row + j];
      }
      if (li < last) for (let j = 0; j < nOut; j++) y[j] = Math.tanh(y[j]);
      input = y;
    }
    return input;
  }

  act(obs: Float32Array, _ctx: Ctx, out: Float32Array) {
    const y = this.forward(obs);
    out[0] = clamp(y[0], -1, 1);
    out[1] = clamp(y[1], -1, 1);
  }
}

/** Classic cascaded PD controller, tuned by hand against nominal parameters. The baseline to beat. */
export class PID implements Brain {
  readonly name = "Classic PD";
  kp = 5;
  kd = 4;
  kth = 160;
  kom = 22;
  maxTilt = 1.1;

  act(_obs: Float32Array, c: Ctx, out: Float32Array) {
    const p = c.phys;
    const m = SPEC.mass;
    const g = SPEC.gravity;
    const T = SPEC.max_thrust;
    const ex = c.tx - p.x;
    const ey = c.ty - p.y;
    let axd = clamp(this.kp * ex + this.kd * (c.tvx - p.vx), -14, 14);
    const ayd = clamp(this.kp * ey + this.kd * (c.tvy - p.vy), -g * 0.8, 22) + g;
    let thd = Math.atan2(-axd, ayd);
    thd = clamp(thd, -this.maxTilt, this.maxTilt);
    axd = -Math.tan(thd) * ayd;
    const s = Math.sin(p.th);
    const co = Math.cos(p.th);
    // Thrust only along body up; scale by how well we are aligned.
    const thrust = clamp(m * (axd * -s + ayd * co), 0, 2 * T);
    const torque = SPEC.inertia * (this.kth * wrapAngle(thd - p.th) - this.kom * p.om);
    const diff = torque / SPEC.arm;
    let fl = (thrust - diff) / 2;
    let fr = (thrust + diff) / 2;
    // Keep attitude authority when saturated: shift both motors together.
    if (fl < 0) (fr -= fl), (fl = 0);
    if (fr < 0) (fl -= fr), (fr = 0);
    if (fl > T) (fr -= fl - T), (fl = T);
    if (fr > T) (fl -= fr - T), (fr = T);
    out[0] = clamp((2 * fl) / T - 1, -1, 1);
    out[1] = clamp((2 * fr) / T - 1, -1, 1);
  }
}
