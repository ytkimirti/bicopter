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


def lock_reach():
    """A hider half a metre from a block can lock it, but not from 0.7 m or through the roof."""
    n = 16
    s, _, _ = reset(jax.random.split(jax.random.PRNGKey(7), n))
    hx, hy = game.BOX_HALF[0]
    cx = (s.room[:, 0] + s.room[:, 1]) / 2
    floor_block = jnp.stack([cx, s.room[:, 2] + hy], -1)
    under_roof = jnp.stack([cx, s.room[:, 3] - hy - 0.01], -1)
    beside = jnp.stack([cx + hx + game.DRONE_HALF[0] + 0.45, s.room[:, 2] + hy], -1)
    too_far = jnp.stack([cx + hx + game.DRONE_HALF[0] + 0.7, s.room[:, 2] + hy], -1)
    on_roof = jnp.stack([cx, s.room[:, 3] + game.WALL_T + game.DRONE_HALF[1] + 0.05], -1)
    cont = jnp.zeros((n, game.ND, game.N_CONT))
    z = jnp.zeros((n, game.ND), jnp.int32)
    lock = z.at[:, 0].set(1)
    far = jnp.array([0.0, 4.0])  # every other drone and box well away
    cases = (("beside", floor_block, beside, True), ("too far", floor_block, too_far, False), ("through roof", under_roof, on_roof, False))
    for name, block, drone, want in cases:
        t = s._replace(
            bpos=jnp.broadcast_to(far, s.bpos.shape).at[:, 0].set(block) + jnp.arange(game.NB)[:, None] * jnp.array([1.0, 0.0]),
            pos=s.pos.at[:, 0].set(drone),
            locked=jnp.full_like(s.locked, -1),
        )
        t, *_ = step(t, pilot, cont, z, lock)
        got = np.asarray(t.locked[:, 0] == 0)
        print(f"lock reach: {name} locked in {got.mean():.2f} of envs")
        assert (got == want).all(), got


def key_start():
    """A quarter of episodes start with the first seeker carrying the key, clear of walls; it unlocks a hider lock."""
    n = 256
    s, _, _ = reset(jax.random.split(jax.random.PRNGKey(9), n))
    carry = np.asarray(s.holding[:, 2] == game.KEY)
    assert bool(jnp.all((s.held[:, game.KEY] == 2) == (s.holding[:, 2] == game.KEY)))
    print(f"key start: {carry.mean():.2f} of episodes, max overlap at reset {float(overlap_depth(s)[carry].max()):.3f} m")
    assert 0.15 < carry.mean() < 0.35, carry.mean()
    assert float(overlap_depth(s)[carry].max()) < 0.03
    # The carrier keeps the key through prep, frozen at its spawn with the key hanging under it.
    hang = jnp.array([0.0, float(game.hold_offset(game.BOX_HALF[game.KEY, 1]))])
    t, z = s, jnp.zeros((n, game.ND), jnp.int32)
    for _ in range(10):
        t, *_ = step(t, pilot, jnp.zeros((n, game.ND, game.N_CONT)), z.at[:, 2].set(1), z)
    assert bool(jnp.all(t.holding[carry, 2] == game.KEY))
    assert float(jnp.abs(s.spawn[carry, 0] - s.pos[carry, 2]).max()) < 1e-4  # spawn is the drone, not its grown shape
    assert float(jnp.abs(t.pos[carry, 2] - s.pos[carry, 2]).max()) < 1e-4
    assert float(jnp.abs(t.bpos[carry, game.KEY] - t.pos[carry, 2] - hang).max()) < 1e-4
    # A carrier spawned just above a platform settles clear of it, key included.
    plat = s.wc[:, 7] + jnp.array([0.0, 0.15 + 0.2])
    t = s._replace(pos=s.pos.at[:, 2].set(plat), bpos=s.bpos.at[:, game.KEY].set(jnp.where(carry[:, None], plat + hang, s.bpos[:, game.KEY])))
    t = jax.vmap(game.settle_all)(t)
    kept = np.asarray(t.holding[:, 2] == game.KEY) & carry
    assert bool(jnp.all((t.held[:, game.KEY] == 2) == (t.holding[:, 2] == game.KEY)))
    worst = float(overlap_depth(t)[kept].max())
    print(f"key start: carriers over a platform keep the key in {kept.sum() / carry.sum():.2f}, overlap {worst:.3f} m")
    assert worst < 0.03 and kept.sum() > 0.8 * carry.sum(), (worst, kept.sum(), carry.sum())
    # After prep, the key carrier comes near a hider-locked block and opens it, no button pressed.
    s = s._replace(t=jnp.full_like(s.t, game.PREP + 1))
    hx, hy = game.BOX_HALF[0]
    cx = (s.room[:, 0] + s.room[:, 1]) / 2
    block = jnp.stack([cx, s.room[:, 2] + hy], -1)
    drone = jnp.stack([cx + hx + game.DRONE_HALF[0] + 0.3, s.room[:, 2] + 1.2], -1)
    far = jnp.array([0.0, 4.0]) + jnp.arange(game.NB)[:, None] * jnp.array([1.0, 0.0])
    t = s._replace(
        bpos=jnp.broadcast_to(far, s.bpos.shape).at[:, 0].set(block),
        pos=s.pos.at[:, 2].set(drone),
        locked=jnp.full_like(s.locked, -1).at[:, 0].set(0),
    )
    t = t._replace(bpos=jnp.where(carry[:, None, None], t.bpos.at[:, game.KEY].set(drone + jnp.array([0.0, float(game.hold_offset(game.BOX_HALF[game.KEY, 1]))])), t.bpos))
    z = jnp.zeros((n, game.ND), jnp.int32)
    t, *_ = step(t, pilot, jnp.zeros((n, game.ND, game.N_CONT)), z, z)
    unlocked = np.asarray(t.locked[:, 0] < 0)
    print(f"key unlock: carriers {unlocked[carry].mean():.2f}, others {unlocked[~carry].mean():.2f}")
    assert unlocked[carry].all() and not unlocked[~carry].any()


def own_box_sight():
    """A seeker sees past the box it carries, not past one its teammate carries."""
    s, _, _ = game.reset_obs(jax.random.PRNGKey(4))
    far = jnp.array([50.0, 50.0])
    eye = jnp.array([0.0, 2.0])
    hang = jnp.array([0.0, float(game.hold_offset(game.BOX_HALF[game.KEY, 1]))])
    base = s._replace(
        wc=s.wc + far,  # walls out of the way
        bpos=jnp.broadcast_to(-far, s.bpos.shape).at[game.KEY].set(eye + hang),
        pos=s.pos.at[0].set(eye - jnp.array([0.0, 1.5])).at[1].set(-far).at[2].set(eye).at[3].set(-far + 3.0),
    )
    for holder, want in ((2, True), (3, False)):
        t = base._replace(holding=jnp.full((game.ND,), -1).at[holder].set(game.KEY), held=jnp.full((game.NB,), -1).at[game.KEY].set(holder))
        got = bool(game.seen_matrix(t)[0, 0])
        print(f"sight: key carried by drone {holder}, seeker 2 sees the hider below: {got}")
        assert got == want


def main():
    doorway_ram()
    own_box_sight()
    key_start()
    lock_reach()
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
