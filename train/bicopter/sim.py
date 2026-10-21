"""Bicopter physics and observation in JAX. Mirrors web/src/sim/physics.ts exactly."""

import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp

SPEC = json.loads((Path(__file__).resolve().parents[2] / "spec.json").read_text())
OBS = SPEC["obs"]
OBS_DIM = OBS["dim"]


class Params(NamedTuple):
    mass: jax.Array
    inertia: jax.Array
    tmax: jax.Array  # (2,) max thrust of left, right motor
    tau: jax.Array
    latency: jax.Array  # int, control steps of action delay


class Phys(NamedTuple):
    pos: jax.Array  # (2,)
    vel: jax.Array  # (2,)
    th: jax.Array
    om: jax.Array
    f: jax.Array  # (2,) current motor thrust, left and right


def nominal_params() -> Params:
    t = SPEC["max_thrust"]
    return Params(
        mass=jnp.float32(SPEC["mass"]),
        inertia=jnp.float32(SPEC["inertia"]),
        tmax=jnp.array([t, t], jnp.float32),
        tau=jnp.float32(SPEC["motor_tau"]),
        latency=jnp.int32(0),
    )


def wrap_angle(th):
    return jnp.mod(th + jnp.pi, 2 * jnp.pi) - jnp.pi


def physics_step(p: Phys, cmd: jax.Array, params: Params, wind: jax.Array) -> Phys:
    """Advance one control step. cmd in [-1, 1]^2 maps to motor thrust 0..tmax."""
    u = jnp.clip((cmd + 1.0) * 0.5, 0.0, 1.0)
    target = u * params.tmax
    h = SPEC["dt"] / SPEC["substeps"]
    alpha = h / (params.tau + h)
    g = SPEC["gravity"]
    arm = SPEC["arm"]
    kl, kq, ka = SPEC["drag_lin"], SPEC["drag_quad"], SPEC["ang_drag"]

    def sub(p: Phys, _):
        f = p.f + (target - p.f) * alpha
        thrust = f[0] + f[1]
        s, c = jnp.sin(p.th), jnp.cos(p.th)
        speed = jnp.sqrt(p.vel[0] ** 2 + p.vel[1] ** 2)
        k = kl + kq * speed
        ax = (-s * thrust + wind[0] - k * p.vel[0]) / params.mass
        ay = (c * thrust + wind[1] - k * p.vel[1]) / params.mass - g
        alpha_ang = ((f[1] - f[0]) * arm - ka * p.om) / params.inertia
        vel = p.vel + jnp.stack([ax, ay]) * h
        pos = p.pos + vel * h
        om = p.om + alpha_ang * h
        th = p.th + om * h
        return Phys(pos, vel, th, om, f), None

    p, _ = jax.lax.scan(sub, p, None, length=SPEC["substeps"])
    return p._replace(th=wrap_angle(p.th))


def observe(p: Phys, target_pos, target_vel, prev_action) -> jax.Array:
    d = target_pos - p.pos
    return jnp.concatenate(
        [
            jnp.clip(d * OBS["pos_scale"], -OBS["pos_clip"], OBS["pos_clip"]),
            jnp.clip(d, -OBS["fine_clip"], OBS["fine_clip"]),
            jnp.clip(p.vel * OBS["vel_scale"], -OBS["vel_clip"], OBS["vel_clip"]),
            jnp.clip(target_vel * OBS["vel_scale"], -OBS["vel_clip"], OBS["vel_clip"]),
            jnp.stack([jnp.sin(p.th), jnp.cos(p.th)]),
            jnp.clip(p.om * OBS["omega_scale"], -OBS["omega_clip"], OBS["omega_clip"])[None],
            p.f / SPEC["max_thrust"],
            prev_action,
        ]
    )
