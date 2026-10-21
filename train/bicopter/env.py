"""Training environment: domain randomization, moving targets, reward. Pure JAX, vmap-friendly."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .sim import OBS_DIM, SPEC, Params, Phys, observe, physics_step

DR = SPEC["dr"]
RW = SPEC["reward"]
AR = SPEC["arena"]
HW, HH = AR["half_w"], AR["half_h"]
DT = SPEC["dt"]
EPISODE_STEPS = 600

# Target behaviours sampled per episode.
JUMPS, SMOOTH, SMOOTH_JUMPS, HOLD, ORBIT = 0, 1, 2, 3, 4
MODE_PROBS = jnp.array([0.35, 0.25, 0.15, 0.05, 0.2])


class State(NamedTuple):
    phys: Phys
    params: Params
    buf: jax.Array  # (2, 2) last two commands, newest first
    target: jax.Array
    target_vel: jax.Array
    mode: jax.Array
    timer: jax.Array
    speed: jax.Array  # target wander speed scale
    orbit: jax.Array  # (7,) Lissajous center x, y, amplitude x, y, frequency x, y, phase
    wind_mean: jax.Array
    gust: jax.Array
    t: jax.Array
    key: jax.Array


def lerp_range(key, lo_hi, level, nominal):
    lo, hi = lo_hi
    x = jax.random.uniform(key, minval=lo, maxval=hi)
    return nominal + (x - nominal) * level


def sample_params(key, level) -> Params:
    k = jax.random.split(key, 6)
    tmax = SPEC["max_thrust"] * jnp.stack(
        [lerp_range(k[2], DR["thrust_scale"], level, 1.0), lerp_range(k[3], DR["thrust_scale"], level, 1.0)]
    )
    lat = jnp.where(
        jax.random.uniform(k[5]) < level,
        jax.random.randint(k[5], (), 0, SPEC["max_latency"] + 1),
        0,
    )
    return Params(
        mass=SPEC["mass"] * lerp_range(k[0], DR["mass"], level, 1.0),
        inertia=SPEC["inertia"] * lerp_range(k[1], DR["inertia"], level, 1.0),
        tmax=tmax,
        tau=lerp_range(k[4], DR["motor_tau"], level, SPEC["motor_tau"]),
        latency=lat.astype(jnp.int32),
    )


def random_box_point(key, shrink=1.0):
    return jax.random.uniform(key, (2,), minval=-1.0, maxval=1.0) * jnp.array([HW, HH]) * shrink


def reset(key, level=1.0) -> State:
    k = jax.random.split(key, 16)
    tumble = jax.random.uniform(k[0]) < 0.3
    th = jnp.where(tumble, jax.random.uniform(k[1], minval=-jnp.pi, maxval=jnp.pi), 0.3 * jax.random.normal(k[1]))
    om = jnp.where(tumble, 12.0 * jax.random.normal(k[2]), 2.0 * jax.random.normal(k[2]))
    phys = Phys(
        pos=random_box_point(k[3]),
        vel=2.0 * jax.random.normal(k[4], (2,)),
        th=th,
        om=om,
        f=jax.random.uniform(k[5], (2,), maxval=SPEC["max_thrust"]),
    )
    mode = jax.random.choice(k[6], 5, p=MODE_PROBS)
    ko = jax.random.split(k[15], 7)
    amp = jnp.stack([jax.random.uniform(ko[0], minval=1.0, maxval=HW - 2), jax.random.uniform(ko[1], minval=0.5, maxval=HH - 1.5)])
    orbit = jnp.concatenate(
        [
            jax.random.uniform(ko[2], (2,), minval=-1.0, maxval=1.0) * (jnp.array([HW, HH]) - amp),
            amp,
            jax.random.uniform(ko[3], (2,), minval=0.3, maxval=1.8) * jnp.where(jax.random.uniform(ko[4]) < 0.5, 1.0, -1.0),
            jax.random.uniform(ko[5], (1,), maxval=2 * jnp.pi),
        ]
    )
    wind_on = jax.random.uniform(k[7]) < DR["wind_prob"] * level
    ang = jax.random.uniform(k[8], maxval=2 * jnp.pi)
    mag = DR["wind_max"] * jnp.sqrt(jax.random.uniform(k[9])) * wind_on
    # An orbit episode starts on its orbit, moving, rather than teleporting there on the first step.
    orb = mode == ORBIT
    target = jnp.where(orb, orbit[0:2] + orbit[2:4] * jnp.sin(jnp.stack([orbit[6], 0.0])), random_box_point(k[11]))
    target_vel = jnp.where(orb, orbit[2:4] * orbit[4:6] * jnp.cos(jnp.stack([orbit[6], 0.0])), jnp.zeros(2))
    return State(
        phys=phys,
        params=sample_params(k[10], level),
        buf=jnp.zeros((2, 2)),
        target=target,
        target_vel=target_vel,
        mode=mode,
        timer=jax.random.randint(k[12], (), 40, 200),
        speed=jax.random.uniform(k[13], minval=0.5, maxval=8.0),
        orbit=orbit,
        wind_mean=mag * jnp.stack([jnp.cos(ang), jnp.sin(ang)]),
        gust=jnp.zeros(2),
        t=jnp.int32(0),
        key=k[14],
    )


def advance_target(s: State, key):
    k = jax.random.split(key, 3)
    smooth = (s.mode == SMOOTH) | (s.mode == SMOOTH_JUMPS)
    # Ornstein-Uhlenbeck velocity with stationary std = speed, reflected at the arena box.
    tau_v = 1.0
    tv = s.target_vel + (-s.target_vel / tau_v) * DT + s.speed * jnp.sqrt(2 * DT / tau_v) * jax.random.normal(k[0], (2,))
    tv = jnp.where(smooth, tv, 0.0)
    tp = s.target + tv * DT
    lim = jnp.array([HW, HH])
    over = jnp.abs(tp) > lim
    tp = jnp.where(over, jnp.sign(tp) * lim * 2 - tp, tp)
    tv = jnp.where(over, -tv, tv)
    timer = s.timer - 1
    jump = (timer <= 0) & (s.mode != SMOOTH) & (s.mode != HOLD)
    tp = jnp.where(jump, random_box_point(k[1]), tp)
    lo = jnp.where(s.mode == SMOOTH_JUMPS, 60, 40)
    hi = jnp.where(s.mode == SMOOTH_JUMPS, 300, 200)
    timer = jnp.where(jump, jax.random.randint(k[2], (), lo, hi), timer)
    # Lissajous orbit with exact velocity, like a circling or figure-eight cursor.
    o = s.orbit
    ph = (s.t + 1) * DT * o[4:6] + jnp.stack([o[6], 0.0])
    op = o[0:2] + o[2:4] * jnp.sin(ph)
    ov = o[2:4] * o[4:6] * jnp.cos(ph)
    orb = s.mode == ORBIT
    tp = jnp.where(orb, op, tp)
    tv = jnp.where(orb, ov, tv)
    return tp, tv, timer


def noisy_obs(s: State, key, level):
    k = jax.random.split(key, 5)
    n = lambda kk, std, shape=(): std * level * jax.random.normal(kk, shape)
    p = s.phys._replace(
        pos=s.phys.pos + n(k[0], DR["pos_noise"], (2,)),
        vel=s.phys.vel + n(k[1], DR["vel_noise"], (2,)),
        th=s.phys.th + n(k[2], DR["angle_noise"]),
        om=s.phys.om + n(k[3], DR["omega_noise"]),
    )
    tv = s.target_vel + n(k[4], DR["target_vel_noise"], (2,)) * (jnp.linalg.norm(s.target_vel) > 0)
    return observe(p, s.target, tv, s.buf[0])


def step(s: State, action, level=1.0):
    """Returns (state, obs, reward, done, truncated). Does not auto-reset."""
    key, k_t, k_g, k_o = jax.random.split(s.key, 4)
    a = jnp.clip(action, -1.0, 1.0)
    hist = jnp.concatenate([a[None], s.buf], axis=0)  # (3, 2): a_t, a_t-1, a_t-2
    cmd = hist[s.params.latency]
    gt = DR["gust_tau"]
    gust = s.gust - s.gust / gt * DT + DR["gust_std"] * level * jnp.sqrt(2 * DT / gt) * jax.random.normal(k_g, (2,))
    target, target_vel, timer = advance_target(s, k_t)
    phys = physics_step(s.phys, cmd, s.params, s.wind_mean + gust)
    s2 = s._replace(
        phys=phys, buf=hist[:2], target=target, target_vel=target_vel, timer=timer, gust=gust, t=s.t + 1, key=key
    )
    d = jnp.linalg.norm(target - phys.pos)
    # Near the target: stay upright and stop spinning. Far away, flips are free.
    near = jnp.exp(-((d / RW["near_radius"]) ** 2))
    r = (
        RW["track_scale"] / (RW["track_scale"] + d)
        + RW["precise_weight"] * jnp.exp(-((d / RW["precise_radius"]) ** 2))
        + near * (RW["upright_weight"] * jnp.cos(phys.th) - RW["near_omega_weight"] * jnp.minimum(phys.om**2, 100.0))
        - RW["omega_weight"] * phys.om**2
        - RW["action_rate_weight"] * jnp.sum((a - s.buf[0]) ** 2)
    )
    m = AR["escape_margin"]
    escaped = (jnp.abs(phys.pos[0]) > HW + m) | (jnp.abs(phys.pos[1]) > HH + m) | ~jnp.isfinite(phys.pos).all()
    r = jnp.where(escaped, r - RW["escape_penalty"], r)
    truncated = (s2.t >= EPISODE_STEPS) & ~escaped
    return s2, noisy_obs(s2, k_o, level), r, escaped | truncated, truncated


def reset_obs(key, level=1.0):
    s = reset(key, level)
    return s, noisy_obs(s, jax.random.fold_in(key, 7), level)


assert OBS_DIM == 15
