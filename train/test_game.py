"""Invariant checks for the hide and seek env. uv run python test_game.py"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from bicopter import game

pilot = game.load_pilot()
step = jax.jit(jax.vmap(game.step, in_axes=(0, None, 0, 0, 0)))
reset = jax.jit(jax.vmap(game.reset_obs))


def overlap_depth(s):
    """Deepest overlap of any free box or drone with a static wall, per env."""
    dc, dh = jax.vmap(game.drone_shapes)(s)
    n, d, _ = jax.vmap(game.mtv)(dc, dh, s.wc, s.wh)
    _, db, _ = jax.vmap(game.mtv)(s.bpos, jnp.broadcast_to(game.BOX_HALF, s.bpos.shape), s.wc, s.wh)
    free = (s.held < 0)[..., None]
    return jnp.maximum(d.max((1, 2)), jnp.where(free, db, 0).max((1, 2)))


def run(n=256, steps=game.EP_LEN, policy="random", seed=0):
    key = jax.random.PRNGKey(seed)
    s, obs, cobs = reset(jax.random.split(key, n))
    assert obs.shape == (n, game.ND, game.OBS_DIM), obs.shape
    worst = 0.0
    for t in range(steps):
        key, k1, k2, k3 = jax.random.split(key, 4)
        if policy == "hover":
            cont = jnp.zeros((n, game.ND, game.N_CONT))
        else:
            cont = jax.random.normal(k1, (n, game.ND, game.N_CONT)) * jnp.array([0.7, 0.7, 0.5, 0.5, 0.1, 0.1])
        grab = jax.random.bernoulli(k2, 0.1, (n, game.ND)).astype(jnp.int32)
        lock = jax.random.bernoulli(k3, 0.1, (n, game.ND)).astype(jnp.int32)
        was_locked, prev_bpos = s.locked >= 0, s.bpos
        s, obs, cobs, r, done, st = step(s, pilot, cont, grab, lock)
        # A box locked before and after the step did not move.
        keep = (was_locked & (s.locked >= 0))[..., None]
        moved = float(jnp.where(keep, jnp.abs(s.bpos - prev_bpos), 0.0).max())
        assert moved < 1e-5, f"locked box moved {moved} at {t}"
        if t > 3:
            worst = max(worst, float(overlap_depth(s).max()))
        assert bool(jnp.isfinite(obs).all()), f"non-finite obs at {t}"
        assert bool(jnp.isfinite(s.pos).all()) and bool(jnp.isfinite(s.bpos).all())
        inside = (jnp.abs(s.pos[..., 0]) < game.HW + 0.01) & (jnp.abs(s.pos[..., 1]) < game.HH + 0.01)
        assert bool(inside.all()), f"drone left the arena at {t}"
        binside = (jnp.abs(s.bpos[..., 0]) < game.HW + 0.01) & (jnp.abs(s.bpos[..., 1]) < game.HH + 0.01)
        assert bool(binside.all()), f"box left the arena at {t}"
        held_ok = jnp.all(jnp.where(s.holding >= 0, jnp.take_along_axis(s.held, jnp.maximum(s.holding, 0), 1) == jnp.arange(game.ND), True))
        assert bool(held_ok), "holding/held disagree"
    return s, st, worst


def doorway_ram():
    """A block taller than the door, shoved outward at the speed cap every step, stays inside and on the floor."""
    n = 64
    s, _, _ = reset(jax.random.split(jax.random.PRNGKey(3), n))
    side = jnp.where(s.doors[:, 0], 0, 1)
    hx, hy = game.BOX_HALF[0]
    x = jnp.where(side == 0, s.room[:, 0] + hx + 0.01, s.room[:, 1] - hx - 0.01)
    push = jnp.where(side == 0, -game.G["max_speed"], game.G["max_speed"])
    s = s._replace(bpos=s.bpos.at[:, 0].set(jnp.stack([x, s.room[:, 2] + hy], -1)), locked=jnp.full_like(s.locked, -1))
    cont = jnp.zeros((n, game.ND, game.N_CONT))
    z = jnp.zeros((n, game.ND), jnp.int32)
    worst_x, worst_y = 0.0, 0.0
    for _ in range(30):
        s = s._replace(bvel=s.bvel.at[:, 0].set(jnp.stack([push, jnp.zeros(n)], -1)))
        s, *_ = step(s, pilot, cont, z, z)
        out = jnp.where(side == 0, s.room[:, 0] - (s.bpos[:, 0, 0] - hx), (s.bpos[:, 0, 0] + hx) - s.room[:, 1])
        worst_x = max(worst_x, float(out.max()))
        worst_y = max(worst_y, float((s.room[:, 2] + hy - s.bpos[:, 0, 1]).max()))
    print(f"doorway ram: block past the wall face {worst_x:.3f} m, sunk {worst_y:.3f} m")
    assert worst_x < 0.02 and worst_y < 0.02, (worst_x, worst_y)


def forts():
    """Fort episodes start with every door sealed, or half of them nearly so."""
    s, _, _ = reset(jax.random.split(jax.random.PRNGKey(5), 1024))
    full = np.asarray(s.stats.spawn_sealed == s.doors.sum(-1))
    two = np.asarray(s.doors.all(-1))
    print(f"forts: {full.mean():.2f} of episodes start fully sealed, {full[two].mean():.2f} of two-door rooms")
    assert 0.2 < full.mean() < 0.36, full.mean()
    # Random blocks almost never seal both doors, so this is the fort placing block 1.
    assert 0.1 < full[two].mean() < 0.25, full[two].mean()


def main():
    doorway_ram()
    forts()
    s, st, worst = run(policy="hover")
    sp = np.asarray(jnp.linalg.norm(s.vel, axis=-1))
    print(f"hover: max wall overlap {worst:.3f} m, final speed mean {sp.mean():.2f} m/s, tilt max {float(jnp.abs(s.th).max()):.2f}")
    s, st, worst = run(policy="random")
    print(f"random: max wall overlap {worst:.3f} m")
    print("stats (random):", {k: float(np.asarray(v).mean()) for k, v in st._asdict().items() if np.asarray(v).ndim == 1})
    print("grabs", np.asarray(st.grabs).mean(0), "locks", np.asarray(st.locks).mean(0))
    assert worst < 0.12, worst

    # Throughput of the bare env.
    n = 512
    s, obs, cobs = reset(jax.random.split(jax.random.PRNGKey(1), n))
    cont = jnp.zeros((n, game.ND, game.N_CONT))
    z = jnp.zeros((n, game.ND), jnp.int32)
    out = step(s, pilot, cont, z, z)
    jax.block_until_ready(out[1])
    t0 = time.time()
    for _ in range(20):
        s, *_ = step(s, pilot, cont, z, z)
    jax.block_until_ready(s.pos)
    dt = time.time() - t0
    print(f"env: {20 * n / dt:.0f} decisions/s ({20 * n * game.REPEAT / dt:.0f} physics steps/s) on one device")
    print("ok")


if __name__ == "__main__":
    main()
