"""Hide and seek for bicopters. Pure JAX, vmap-friendly.

Two hiders and two seekers fly the same drones as sim.py, each steered by the frozen PPO pilot from web/public/brains.
A shared high-level policy acts at 10 Hz: it gives the pilot a target (offset and velocity), adds a residual to the
motor commands, and presses a grab toggle and a lock button. A toggle (rather than a hold-to-keep button) lets even a
random policy carry a box for a while, and a lock button that cannot undo itself leaves locks in place; exploration
needs both to find forts. The arena is a closed box with a room on the floor whose side
walls have doors. Blocks are taller than the doors, so only blocks already inside can seal them. Any team can lock a
free box in place, for good: locks never undo, except that a seeker carrying the key unlocks hider locks.

Hiders get +1 per decision while no seeker has line of sight to any hider, -1 otherwise; seekers get the opposite.
During the prep phase seekers are frozen and blind, and nobody is rewarded.
"""

import base64
import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .sim import SPEC, Phys, observe

G = SPEC["game"]
HW, HH = SPEC["arena"]["half_w"], SPEC["arena"]["half_h"]
DT = SPEC["dt"]
SUB = SPEC["substeps"]
H_SUB = DT / SUB
GRAV = SPEC["gravity"]

REPEAT = G["decision_repeat"]
DEC_DT = DT * REPEAT
PREP = int(round(G["prep_s"] / DEC_DT))
EP_LEN = PREP + int(round(G["seek_s"] / DEC_DT))

ND, NB, NW = 4, 4, 9  # drones (0, 1 hiders; 2, 3 seekers), boxes (0-2 blocks, 3 key), static walls
KEY = 3
TEAM = jnp.array([0, 0, 1, 1])
DRONE_HALF = jnp.array(G["drone_half"])
BOX_HALF = jnp.array([G["block_half"]] * 3 + [G["key_half"]])
BOX_MASS = jnp.array([G["block_mass"]] * 3 + [G["key_mass"]])
FLOOR = -HH
DOOR_H = G["door_h"]
WALL_T = G["wall_t"]
REACH = G["reach"]
LOCK_REACH = G["lock_reach"]

N_CONT = 6  # target offset (2), target velocity (2), motor residual (2)
N_GRAB, N_LOCK = 2, 2  # nothing / toggle grab; nothing / lock
OBS_DIM = 12 + 3 * 7 + NB * 10 + 9


# ---------- pilot ----------


def load_pilot(path=None):
    path = path or Path(__file__).resolve().parents[2] / "web" / "public" / "brains" / "ppo-best.json"
    doc = json.loads(Path(path).read_text())
    dec = lambda s: np.frombuffer(base64.b64decode(s), "<f4")
    return [(jnp.asarray(dec(l["w"]).reshape(l["in"], l["out"])), jnp.asarray(dec(l["b"]))) for l in doc["layers"]]


def pilot_forward(layers, x):
    for i, (w, b) in enumerate(layers):
        x = x @ w + b
        if i < len(layers) - 1:
            x = jnp.tanh(x)
    return jnp.clip(x, -1.0, 1.0)


# ---------- state ----------


class Stats(NamedTuple):
    hidden: jax.Array  # seek decisions with every hider hidden
    in_room: jax.Array  # hiders inside the room at the end of prep
    doors: jax.Array  # doors in this layout
    sealed: jax.Array  # doors blocked by a box at the end of prep
    locked_prep: jax.Array  # boxes in a doorway locked by hiders at the end of prep
    key_denied: jax.Array  # key locked by hiders or inside the room at the end of prep
    unsealed: jax.Array  # doors sealed at prep end that opened later with a seeker at the door
    key_unlocks: jax.Array  # hider locks opened by a seeker with the key
    grabs: jax.Array  # (2,) grabs per team
    locks: jax.Array  # (2,) locks per team
    throws: jax.Array  # boxes released faster than 6 m/s
    flips: jax.Array  # drone-decisions spent more than 120 degrees from upright
    spawn_sealed: jax.Array  # doors already sealed when the episode started


class State(NamedTuple):
    pos: jax.Array  # (ND, 2)
    vel: jax.Array
    th: jax.Array  # (ND,)
    om: jax.Array
    f: jax.Array  # (ND, 2) motor thrusts
    prev: jax.Array  # (ND, 2) last motor command, what the pilot sees as its previous action
    holding: jax.Array  # (ND,) box index or -1
    bpos: jax.Array  # (NB, 2)
    bvel: jax.Array
    locked: jax.Array  # (NB,) locking team or -1
    held: jax.Array  # (NB,) holder drone or -1
    wc: jax.Array  # (NW, 2) static wall centers
    wh: jax.Array  # (NW, 2) half extents
    room: jax.Array  # (4,) interior x0, x1, y0, y1
    doors: jax.Array  # (2,) left, right door present
    spawn: jax.Array  # (2, 2) seeker positions held during prep
    mem: jax.Array  # (2, 2, 2) per team, last seen position of each opponent
    since: jax.Array  # (2, 2) seconds since each opponent was seen by the team
    seal0: jax.Array  # (2,) doors sealed at the end of prep
    t: jax.Array
    stats: Stats
    key: jax.Array


def zero_stats():
    z = jnp.float32(0.0)
    return Stats(z, z, z, z, z, z, z, z, jnp.zeros(2), jnp.zeros(2), z, z, z)


def aabb(cx, cy, hx, hy):
    return jnp.array([cx, cy]), jnp.array([hx, hy])


def make_layout(key):
    k = jax.random.split(key, 8)
    w = jax.random.uniform(k[0], minval=G["room_w"][0], maxval=G["room_w"][1])
    h = jax.random.uniform(k[1], minval=G["room_h"][0], maxval=G["room_h"][1])
    margin = w / 2 + WALL_T + 1.5
    rx = jax.random.uniform(k[2], minval=-HW + margin, maxval=HW - margin)
    which = jax.random.choice(k[3], 3, p=jnp.array([0.4, 0.4, 0.2]))
    doors = jnp.stack([which != 1, which != 0])
    x0, x1, y0, y1 = rx - w / 2, rx + w / 2, FLOOR, FLOOR + h
    walls = [
        aabb(0.0, FLOOR - 1.0, HW + 2, 1.0),
        aabb(0.0, HH + 1.0, HW + 2, 1.0),
        aabb(-HW - 1.0, 0.0, 1.0, HH + 2),
        aabb(HW + 1.0, 0.0, 1.0, HH + 2),
        aabb(rx, y1 + WALL_T / 2, w / 2 + WALL_T, WALL_T / 2),
    ]
    for side, xc in ((0, x0 - WALL_T / 2), (1, x1 + WALL_T / 2)):
        bottom = y0 + DOOR_H * doors[side]
        walls.append(aabb(xc, (bottom + y1) / 2, WALL_T / 2, (y1 - bottom) / 2))
    for i in range(2):
        kk = jax.random.split(k[4 + i], 3)
        pw = jax.random.uniform(kk[0], minval=0.75, maxval=1.5)
        px = jax.random.uniform(kk[1], minval=-HW + 1.5, maxval=HW - 1.5)
        py = jax.random.uniform(kk[2], minval=-0.5, maxval=HH - 1.5)
        walls.append(aabb(px, py, pw, 0.15))
    wc = jnp.stack([c for c, _ in walls])
    wh = jnp.stack([h for _, h in walls])
    return wc, wh, jnp.stack([x0, x1, y0, y1]), doors


def sample_outside(key, room, n):
    """Points in the air, away from the room (lifted above its roof if they land on it)."""
    k1, k2 = jax.random.split(key)
    x = jax.random.uniform(k1, (n,), minval=-HW + 0.6, maxval=HW - 0.6)
    y = jax.random.uniform(k2, (n,), minval=FLOOR + 0.5, maxval=HH - 0.6)
    over = (x > room[0] - WALL_T - 0.5) & (x < room[1] + WALL_T + 0.5) & (y < room[3] + WALL_T + 0.5)
    top = room[3] + WALL_T + 0.6
    y = jnp.where(over, top + (y - FLOOR) / (HH - FLOOR) * (HH - 0.6 - top), y)
    return jnp.stack([x, y], -1)


def sample_inside(key, room, n, half):
    k1, k2 = jax.random.split(key)
    x = jax.random.uniform(k1, (n,), minval=room[0] + half[0] + 0.1, maxval=room[1] - half[0] - 0.1)
    y = jax.random.uniform(k2, (n,), minval=room[2] + half[1] + 0.1, maxval=room[3] - half[1] - 0.1)
    return jnp.stack([x, y], -1)


def reset(key) -> State:
    k = jax.random.split(key, 10)
    wc, wh, room, doors = make_layout(k[0])
    # In some episodes the room starts as a fort: a block stands against the inside of every door, and the hiders
    # usually start inside it. Keeping (and locking) it is the first thing hiders can learn, and breaking it the
    # first thing seekers can.
    kd = jax.random.split(jax.random.fold_in(key, 11), 2)
    fort = jax.random.uniform(kd[1]) < 0.3
    # Otherwise a third of episodes start the hiders inside the room, so they find it before they could learn to look
    # for it.
    hid_in = jax.random.uniform(k[1]) < jnp.where(fort, 0.7, 0.3)
    hp = jnp.where(hid_in, sample_inside(k[2], room, 2, DRONE_HALF), sample_outside(k[3], room, 2))
    sp = sample_outside(k[4], room, 2)
    pos = jnp.concatenate([hp, sp])
    # Blocks start inside the room half the time, on the floor elsewhere otherwise; the key always starts outside.
    bh = BOX_HALF
    inside = jax.random.uniform(k[5], (NB,)) < jnp.array([0.6, 0.5, 0.4, 0.0])
    bin_ = sample_inside(k[6], room, NB, bh[0])
    bout = sample_outside(k[7], room, NB)
    bpos = jnp.where(inside[:, None], jnp.stack([bin_[:, 0], room[2] + bh[:, 1]], -1), bout)
    # Fort: block 0 seals the left door if there is one, else the right; block 1 seals the right door of a two-door room.
    # Half the forts are unfinished, each block standing up to a metre inward from its door; past about 0.45 m it no
    # longer seals. Finishing the fort is then a short push that pays off long before the hiders could build a whole
    # one by chance (a reverse curriculum).
    y = room[2] + bh[0, 1]
    ko = jax.random.split(jax.random.fold_in(key, 12), 2)
    off = jnp.where(jax.random.uniform(ko[0]) < 0.5, 0.0, jax.random.uniform(ko[1], (2,)))
    left = jnp.stack([room[0] + bh[0, 0] + 0.05 + off[0], y])
    right = jnp.stack([room[1] - bh[0, 0] - 0.05 - off[1], y])
    bpos = bpos.at[0].set(jnp.where(fort, jnp.where(doors[0], left, right), bpos[0]))
    bpos = bpos.at[1].set(jnp.where(fort & doors[0] & doors[1], right, bpos[1]))
    # A quarter of episodes hand the key to the first seeker from the start, so using it on a locked fort is something
    # seekers can find before they have learned to fetch it.
    key_start = jax.random.uniform(jax.random.fold_in(key, 13)) < 0.25
    carry_at = pos[ND // 2] + jnp.stack([0.0, hold_offset(BOX_HALF[KEY, 1])])
    bpos = bpos.at[KEY].set(jnp.where(key_start, carry_at, bpos[KEY]))
    holding = jnp.full((ND,), -1).at[ND // 2].set(jnp.where(key_start, KEY, -1))
    held = jnp.full((NB,), -1).at[KEY].set(jnp.where(key_start, ND // 2, -1))
    hover = SPEC["mass"] * GRAV / 2
    return State(
        pos=pos,
        vel=jnp.zeros((ND, 2)),
        th=0.2 * jax.random.normal(k[8], (ND,)),
        om=jnp.zeros(ND),
        f=jnp.full((ND, 2), hover),
        prev=jnp.full((ND, 2), 2 * hover / SPEC["max_thrust"] - 1),
        holding=holding,
        bpos=bpos,
        bvel=jnp.zeros((NB, 2)),
        locked=jnp.full((NB,), -1),
        held=held,
        wc=wc,
        wh=wh,
        room=room,
        doors=doors,
        spawn=sp,
        mem=jnp.zeros((2, 2, 2)),
        since=jnp.full((2, 2), G["memory_s"]),
        seal0=jnp.zeros(2, bool),
        t=jnp.int32(0),
        stats=zero_stats(),
        key=k[9],
    )


# ---------- geometry ----------


def pick(idx, table):
    """table[idx] as a one-hot product (zeros where idx is -1). Gathers under vmap are slow on CPU."""
    oh = (idx[..., None] == jnp.arange(table.shape[0])).astype(table.dtype)
    return oh @ table


def hold_offset(hy):
    """Vertical offset from a drone's center to the center of a carried box of half height hy."""
    return -(DRONE_HALF[1] + G["hold_gap"] + hy)


def drone_shapes(s: State):
    """Collision AABB of each drone, grown to include the box it carries."""
    has = s.holding >= 0
    bh = pick(s.holding, BOX_HALF)
    top = s.pos[:, 1] + DRONE_HALF[1]
    bottom = jnp.where(has, s.pos[:, 1] + hold_offset(bh[:, 1]) - bh[:, 1], s.pos[:, 1] - DRONE_HALF[1])
    hx = jnp.where(has, jnp.maximum(DRONE_HALF[0], bh[:, 0]), DRONE_HALF[0])
    c = jnp.stack([s.pos[:, 0], (top + bottom) / 2], -1)
    h = jnp.stack([hx, (top - bottom) / 2], -1)
    return c, h


def mtv(ca, ha, cb, hb):
    """Pairwise overlap of AABB sets. Returns normal from a to b (n, m, 2), depth (n, m), overlap mask."""
    d = cb[None, :, :] - ca[:, None, :]
    p = ha[:, None, :] + hb[None, :, :] - jnp.abs(d)
    over = (p[..., 0] > 0) & (p[..., 1] > 0)
    use_x = p[..., 0] < p[..., 1]
    sx = jnp.where(d[..., 0] >= 0, 1.0, -1.0)
    sy = jnp.where(d[..., 1] >= 0, 1.0, -1.0)
    n = jnp.stack([jnp.where(use_x, sx, 0.0), jnp.where(use_x, 0.0, sy)], -1)
    depth = jnp.where(over, jnp.where(use_x, p[..., 0], p[..., 1]), 0.0)
    return n, depth, over


def gap(ca, ha, cb, hb):
    """Distance between AABB sets (0 when touching or overlapping), shape (n, m)."""
    d = jnp.abs(cb[None, :, :] - ca[:, None, :]) - ha[:, None, :] - hb[None, :, :]
    return jnp.linalg.norm(jnp.maximum(d, 0.0), axis=-1)


def segment_hits(p0, p1, bc, bh):
    """Which of the AABBs (k, 2) each segment p0 -> p1 (..., 2) crosses, shape (..., k). Slab test."""
    d = p1 - p0
    inv = 1.0 / jnp.where(jnp.abs(d) < 1e-6, 1e-6, d)
    lo = (bc - bh)[(None,) * (p0.ndim - 1)]
    hi = (bc + bh)[(None,) * (p0.ndim - 1)]
    t1 = (lo - p0[..., None, :]) * inv[..., None, :]
    t2 = (hi - p0[..., None, :]) * inv[..., None, :]
    tmin = jnp.max(jnp.minimum(t1, t2), -1)
    tmax = jnp.min(jnp.maximum(t1, t2), -1)
    return (tmax >= tmin) & (tmax > 0.0) & (tmin < 1.0)


def segment_blocked(p0, p1, bc, bh):
    return jnp.any(segment_hits(p0, p1, bc, bh), -1)


def occluders(s: State):
    return jnp.concatenate([s.wc, s.bpos]), jnp.concatenate([s.wh, BOX_HALF])


def seen_matrix(s: State):
    """(2 seekers, 2 hiders): seeker has line of sight to the hider's center or either rotor."""
    oc, oh = occluders(s)
    eye = s.pos[2:]
    pts = s.pos[:2, None, :] + jnp.array([[0.0, 0.0], [-0.2, 0.0], [0.2, 0.0]])  # (2, 3, 2)
    p0 = jnp.broadcast_to(eye[:, None, None, :], (2, 2, 3, 2))
    p1 = jnp.broadcast_to(pts[None], (2, 2, 3, 2))
    return jnp.any(~segment_blocked(p0, p1, oc, oh), -1)


def door_probes(s: State):
    """Two segments through each door at shin and head height, reaching one drone width past each wall face: a box
    farther away leaves a gap a drone fits through."""
    x0, x1 = s.room[0], s.room[1]
    ys = FLOOR + jnp.array([0.25, DOOR_H - 0.2])
    m = 2 * DRONE_HALF[0]
    out = jnp.stack([jnp.full(2, x0 - WALL_T - m), jnp.full(2, x1 + WALL_T + m)])
    inn = jnp.stack([jnp.full(2, x0 + m), jnp.full(2, x1 - m)])
    p0 = jnp.stack([out, jnp.broadcast_to(ys, (2, 2))], -1)
    p1 = jnp.stack([inn, jnp.broadcast_to(ys, (2, 2))], -1)
    return p0, p1


def sealed_doors(s: State):
    p0, p1 = door_probes(s)
    blocked = segment_blocked(p0, p1, s.bpos, BOX_HALF)  # (2 doors, 2 probes)
    return s.doors & jnp.all(blocked, -1)


def door_boxes(s: State):
    """(2 doors, NB) boxes crossing a probe of each existing door."""
    p0, p1 = door_probes(s)
    hits = segment_hits(p0, p1, s.bpos, BOX_HALF)  # (2 doors, 2 probes, NB)
    return jnp.any(hits, 1) & s.doors[:, None]


def in_doorway(s: State):
    """(NB,) boxes crossing a probe of an existing door."""
    return jnp.any(door_boxes(s), 0)


def seekers_at_doors(s: State):
    """(2 doors,) a seeker is within reach of the doorway, so a seal that breaks now was broken by them."""
    door_x = jnp.stack([s.room[0] - WALL_T / 2, s.room[1] + WALL_T / 2])
    near_x = jnp.abs(s.pos[2:, 0][None, :] - door_x[:, None]) < WALL_T / 2 + 2 * DRONE_HALF[0] + 0.8
    near_y = s.pos[2:, 1][None, :] < FLOOR + DOOR_H + 1.0
    return jnp.any(near_x & near_y, 1)


def in_room(s: State, p):
    r = s.room
    return (p[..., 0] > r[0]) & (p[..., 0] < r[1]) & (p[..., 1] > r[2]) & (p[..., 1] < r[3])


# ---------- physics ----------


def _pairs(cx, cy, hx, hy, ox, oy, gx, gy, old=None):
    """Overlap of AABBs (n,) against (m,), as (n, m) arrays: normal from self to other, depth, overlap.

    With old (cx, cy) of the self bodies from before this step, the push-out axis is the one the body entered along
    (the axis on which the pair did not overlap before), not merely the shallower one. Otherwise a tall block shoved
    into a doorway would be pushed down through the floor by the wall above the door, rather than stopped."""
    dx = ox[None, :] - cx[:, None]
    dy = oy[None, :] - cy[:, None]
    px = hx[:, None] + gx[None, :] - jnp.abs(dx)
    py = hy[:, None] + gy[None, :] - jnp.abs(dy)
    over = (px > 0) & (py > 0)
    use_x = px < py
    if old is not None:
        was_x = hx[:, None] + gx[None, :] - jnp.abs(ox[None, :] - old[0][:, None]) > 0
        was_y = hy[:, None] + gy[None, :] - jnp.abs(oy[None, :] - old[1][:, None]) > 0
        use_x = jnp.where(was_y & ~was_x, True, jnp.where(was_x & ~was_y, False, use_x))
    nx = jnp.where(use_x, jnp.sign(dx) + (dx == 0), 0.0)
    ny = jnp.where(use_x, 0.0, jnp.sign(dy) + (dy == 0))
    depth = jnp.where(over, jnp.where(use_x, px, py), 0.0)
    return nx, ny, depth, over


def _impulse(vx, vy, ox, oy, nx, ny, over, w_self, w_sum):
    """Velocity change of each self body from its contacts: restitution along the normal, Coulomb friction across."""
    e, mu = G["restitution"], G["friction"]
    rx = ox - vx[:, None]
    ry = oy - vy[:, None]
    vn = rx * nx + ry * ny
    j = jnp.where(over & (vn < 0), -(1 + e) * vn / w_sum, 0.0)
    vt = -rx * ny + ry * nx
    # Resting contacts carry weight, so friction holds a box still on a floor or another box.
    lim = mu * (j + jnp.where(over & (ny != 0), GRAV * DT / w_sum, 0.0))
    jt = jnp.where(over, jnp.clip(vt / w_sum, -lim, lim), 0.0)
    dvx = jnp.sum((-j * nx - jt * ny) * w_self, 1)
    dvy = jnp.sum((-j * ny + jt * nx) * w_self, 1)
    return vx + dvx, vy + dvy


def resolve(c, h, v, invm, active, wc, wh, c_old, bodies=True):
    """One pass of contacts: dynamic AABBs against the static walls, then (if bodies) against each other.
    c_old: centers before this step, to pick the axis each body hit a wall along."""
    cx, cy, hx, hy, vx, vy = c[:, 0], c[:, 1], h[:, 0], h[:, 1], v[:, 0], v[:, 1]
    act = active & (invm > 0)
    old = (c_old[:, 0], c_old[:, 1])
    nx, ny, depth, over = _pairs(cx, cy, hx, hy, wc[:, 0], wc[:, 1], wh[:, 0], wh[:, 1], old)
    over = over & act[:, None]
    contact = jnp.any(over, 1)
    cx = cx - jnp.sum(nx * depth * over, 1)
    cy = cy - jnp.sum(ny * depth * over, 1)
    w = jnp.maximum(invm, 1e-6)[:, None]
    vx, vy = _impulse(vx, vy, 0.0, 0.0, nx, ny, over, w, w)
    if not bodies:
        return jnp.stack([cx, cy], -1), jnp.stack([vx, vy], -1), contact

    nx, ny, depth, over = _pairs(cx, cy, hx, hy, cx, cy, hx, hy)
    k = c.shape[0]
    w_sum = invm[:, None] + invm[None, :]
    over = over & ~jnp.eye(k, dtype=bool) & active[:, None] & active[None, :] & (w_sum > 0)
    w_sum = jnp.maximum(w_sum, 1e-6)
    share = jnp.where(over, depth * invm[:, None] / w_sum, 0.0)
    cx = cx - jnp.sum(nx * share, 1)
    cy = cy - jnp.sum(ny * share, 1)
    vx, vy = _impulse(vx, vy, vx[None, :], vy[None, :], nx, ny, over, invm[:, None], w_sum)
    contact = contact | jnp.any(over, 1)
    return jnp.stack([cx, cy], -1), jnp.stack([vx, vy], -1), contact


def fly(s: State, target_f, frozen):
    """One physics substep of free flight and falling, no contacts. Same equations as sim.py, without wind."""
    mass = SPEC["mass"] + pick(s.holding, BOX_MASS)
    tau = SPEC["motor_tau"]
    alpha = H_SUB / (tau + H_SUB)
    f = s.f + (target_f - s.f) * alpha
    thrust = f[:, 0] + f[:, 1]
    sn, cs = jnp.sin(s.th), jnp.cos(s.th)
    speed = jnp.linalg.norm(s.vel, axis=-1)
    k = SPEC["drag_lin"] + SPEC["drag_quad"] * speed
    ax = (-sn * thrust - k * s.vel[:, 0]) / mass
    ay = (cs * thrust - k * s.vel[:, 1]) / mass - GRAV
    vel = s.vel + jnp.stack([ax, ay], -1) * H_SUB
    om = s.om + ((f[:, 1] - f[:, 0]) * SPEC["arm"] - SPEC["ang_drag"] * s.om) / SPEC["inertia"] * H_SUB
    th = s.th + om * H_SUB
    pos = s.pos + vel * H_SUB

    free = (s.held < 0) & (s.locked < 0)
    bvel = jnp.where(free[:, None], s.bvel + jnp.array([0.0, -GRAV]) * H_SUB - 0.05 * s.bvel * H_SUB, 0.0)
    bpos = s.bpos + bvel * H_SUB

    # Frozen seekers hover at their spawn.
    hover = SPEC["mass"] * GRAV / 2
    fz = frozen[:, None]
    pos = jnp.where(fz, jnp.concatenate([pos[:2], s.spawn]), pos)
    vel = jnp.where(fz, 0.0, vel)
    th = jnp.where(frozen, 0.0, th)
    om = jnp.where(frozen, 0.0, om)
    f = jnp.where(fz, hover, f)
    return s._replace(pos=pos, vel=vel, th=th, om=om, f=f, bpos=bpos, bvel=bvel)


def collide(s: State, frozen, pos0, bpos0):
    """Contacts, once per control step. Speeds are capped so nothing crosses a wall in one step.
    pos0, bpos0: drone and box positions at the start of the step."""
    mass = SPEC["mass"] + pick(s.holding, BOX_MASS)
    free = (s.held < 0) & (s.locked < 0)
    dc, dh = drone_shapes(s)
    off = dc[:, 1] - s.pos[:, 1]
    c = jnp.concatenate([dc, s.bpos])
    c_old = jnp.concatenate([pos0 + jnp.stack([jnp.zeros(ND), off], -1), bpos0])
    h = jnp.concatenate([dh, BOX_HALF])
    v = jnp.concatenate([s.vel, s.bvel])
    # Frozen seekers and locked boxes are immovable; carried boxes are part of their drone's shape.
    invm = jnp.concatenate([jnp.where(frozen, 0.0, 1.0 / mass), jnp.where(free, 1.0 / BOX_MASS, 0.0)])
    active = jnp.concatenate([jnp.ones(ND, bool), s.held < 0])
    touched = jnp.zeros(ND + NB, bool)
    for _ in range(2):
        c, v, contact = resolve(c, h, v, invm, active, s.wc, s.wh, c_old)
        touched = touched | contact
    # Walls get the last word, so a body pushed by another body does not end the step inside a wall. Two passes, so
    # a push out of one wall (the floor) into another (the wall above a door) is resolved too.
    for _ in range(2):
        c, v, contact = resolve(c, h, v, invm, active, s.wc, s.wh, c_old, bodies=False)
        touched = touched | contact
    vmax = G["max_speed"]
    sp = jnp.linalg.norm(v, axis=-1, keepdims=True)
    v = v * jnp.minimum(1.0, vmax / jnp.maximum(sp, 1e-6))
    pos = c[:ND] - jnp.stack([jnp.zeros(ND), off], -1)
    vel = v[:ND]
    # Scraping along walls bleeds spin, so a drone resting on the floor does not twirl in place.
    om = jnp.where(touched[:ND], s.om * 0.7, s.om)
    bpos = jnp.where(free[:, None], c[ND:], s.bpos)
    bvel = jnp.where(free[:, None], v[ND:], s.bvel)
    # Carried boxes ride under their drone.
    carried = s.held >= 0
    hp = pick(s.held, pos) + jnp.stack([jnp.zeros(NB), hold_offset(BOX_HALF[:, 1])], -1)
    bpos = jnp.where(carried[:, None], hp, bpos)
    bvel = jnp.where(carried[:, None], pick(s.held, vel), bvel)
    return s._replace(pos=pos, vel=vel, om=om, bpos=bpos, bvel=bvel)


# ---------- grabbing and locking ----------


def grab_and_lock(s: State, grab, lock, frozen):
    st = s.stats
    holding, held, locked = s.holding, s.held, s.locked
    bvel = s.bvel
    grabs, locks, throws, key_unlocks = st.grabs, st.locks, st.throws, st.key_unlocks
    bottom = s.pos[:, 1] - DRONE_HALF[1]
    for i in range(ND):
        team = i // 2
        can = ~frozen[i]
        # Grab toggle while holding releases; the box keeps the drone's velocity, so a release at speed is a throw.
        was_holding = holding[i] >= 0
        rel = can & was_holding & (grab[i] == 1)
        b = jnp.maximum(holding[i], 0)
        fast = jnp.linalg.norm(s.vel[i]) > 6.0
        throws = throws + (rel & fast)
        bvel = jnp.where(rel & (jnp.arange(NB) == b)[:, None], s.vel[i], bvel)
        held = jnp.where(rel & (jnp.arange(NB) == b), -1, held)
        holding = holding.at[i].set(jnp.where(rel, -1, holding[i]))
        # Grab toggle while empty-handed picks up a free, unlocked box just under the drone.
        top = s.bpos[:, 1] + BOX_HALF[:, 1]
        under = (jnp.abs(s.bpos[:, 0] - s.pos[i, 0]) < BOX_HALF[:, 0] + REACH) & (top < bottom[i] + 0.1) & (
            top > bottom[i] - REACH - 0.15
        )
        # Not through a roof or platform: the line from the drone to the box top must be clear of walls.
        clear = ~segment_blocked(
            jnp.broadcast_to(s.pos[i], (NB, 2)), jnp.stack([s.bpos[:, 0], top], -1), s.wc, s.wh
        )
        ok = under & clear & (held < 0) & (locked < 0)
        dist = jnp.where(ok, jnp.abs(s.bpos[:, 0] - s.pos[i, 0]) + jnp.abs(top - bottom[i]), 1e9)
        bj = jnp.argmin(dist)
        do = can & ~was_holding & (grab[i] == 1) & ok[bj]
        grabs = grabs + jnp.where(do, jax.nn.one_hot(team, 2), 0.0)
        held = held.at[bj].set(jnp.where(do, i, held[bj]))
        holding = holding.at[i].set(jnp.where(do, bj, holding[i]))
        # Lock button: locks the nearest loose box within reach of the drone or the box it carries, not through a
        # wall. Locks are permanent, so a stray press never undoes a fort; the one exception is a seeker carrying the
        # key, which unlocks hider locks with it before locking anything. The reach is generous so that pressing lock
        # by a block is something exploration actually stumbles on.
        dc, dh = drone_shapes(s._replace(holding=holding))
        d = gap(dc[i : i + 1], dh[i : i + 1], s.bpos, BOX_HALF)[0]
        near_pt = jnp.clip(s.pos[i], s.bpos - BOX_HALF, s.bpos + BOX_HALF)
        d = jnp.where(segment_blocked(jnp.broadcast_to(s.pos[i], (NB, 2)), near_pt, s.wc, s.wh), 1e9, d)
        free = held < 0
        has_key = holding[i] == KEY
        lk = free & (locked < 0)
        ul_key = free & (locked == 0) & (team == 1) & has_key & (jnp.arange(NB) != KEY)
        cand = (lk | ul_key) & (d < LOCK_REACH)
        bt = jnp.argmin(jnp.where(cand, d + 1e3 * jnp.where(ul_key, 0.0, 1.0), 1e9))
        press = can & (lock[i] == 1) & cand[bt]
        do_l = press & lk[bt]
        do_u = press & ul_key[bt]
        locks = locks + jnp.where(do_l, jax.nn.one_hot(team, 2), 0.0)
        key_unlocks = key_unlocks + do_u
        locked = locked.at[bt].set(jnp.where(do_l, team, jnp.where(do_u, -1, locked[bt])))
    bvel = jnp.where((locked >= 0)[:, None], 0.0, bvel)
    st = st._replace(grabs=grabs, locks=locks, throws=throws, key_unlocks=key_unlocks)
    return s._replace(holding=holding, held=held, locked=locked, bvel=bvel, stats=st)


# ---------- observation ----------


def team_sees(s: State):
    """What each team can see this decision: opponents (2 teams, 2) and boxes (2 teams, NB).

    A box is visible to a team that carries it or has line of sight to its center (the box itself does not occlude).
    Seekers are blind during prep."""
    m = seen_matrix(s)  # (seeker, hider)
    hider_seen = jnp.any(m, 0)
    seeker_seen = jnp.any(m, 1)
    oc, oh = occluders(s)
    p0 = jnp.broadcast_to(s.pos[:, None, :], (ND, NB, 2))
    p1 = jnp.broadcast_to(s.bpos[None], (ND, NB, 2))
    hits = segment_hits(p0, p1, oc, oh)  # (ND, NB, NW + NB)
    own = jnp.arange(NW + NB)[None, None, :] == NW + jnp.arange(NB)[None, :, None]
    los = ~jnp.any(hits & ~own, -1)  # (ND, NB)
    holder_team = jnp.where(s.held >= 0, s.held // 2, -1)
    box_vis = jnp.stack([jnp.any(los[2 * t : 2 * t + 2], 0) | (holder_team == t) for t in range(2)])
    awake = jnp.array([True, False]) | (s.t >= PREP)
    return jnp.stack([seeker_seen, hider_seen]) & awake[:, None], box_vis & awake[:, None]


def observe_all(s: State, vis, box_vis):
    """Per-agent observations (ND, OBS_DIM). vis (2 teams, 2 opponents) and box_vis (2 teams, NB) mask what the
    agent's team cannot see; pass all-true for the critic."""
    prep = (s.t < PREP).astype(jnp.float32)
    tf = s.t / EP_LEN
    # A team knows a door is sealed only if it can see a box in that doorway.
    sealed = sealed_doors(s)[None, :] & jnp.any(door_boxes(s)[None] & box_vis[:, None, :], -1)  # (2 teams, 2 doors)
    clip = lambda x, lim=4.0: jnp.clip(x, -lim, lim)

    def one(i):
        team = i // 2
        me = s.pos[i]
        has = s.holding[i] >= 0
        own = [
            me / jnp.array([HW, HH]),
            clip(s.vel[i] * 0.2, 5.0),
            jnp.stack([jnp.sin(s.th[i]), jnp.cos(s.th[i]), jnp.clip(s.om[i] * 0.1, -5, 5)]),
            jnp.stack([has, s.holding[i] == KEY, team == 1, prep, tf]).astype(jnp.float32),
        ]
        mate = i ^ 1
        others = [
            clip((s.pos[mate] - me) * 0.2),
            clip(s.vel[mate] * 0.2, 5.0),
            jnp.stack([1.0, (s.holding[mate] >= 0).astype(jnp.float32), 0.0]),
        ]
        for k in range(2):
            j = 2 * (1 - team) + k
            v = vis[team, k]
            p = jnp.where(v, s.pos[j], s.mem[team, k])
            others += [
                clip((p - me) * 0.2),
                jnp.where(v, clip(s.vel[j] * 0.2, 5.0), 0.0),
                jnp.stack(
                    [
                        v.astype(jnp.float32),
                        jnp.where(v, s.holding[j] >= 0, 0.0).astype(jnp.float32),
                        jnp.where(v, 0.0, s.since[team, k] / G["memory_s"]),
                    ]
                ),
            ]
        boxes = []
        for b in range(NB):
            h = s.held[b]
            bv = box_vis[team, b]
            boxes += [
                jnp.where(bv, clip((s.bpos[b] - me) * 0.2), 0.0),
                jnp.where(bv, clip(s.bvel[b] * 0.2, 5.0), 0.0),
                (
                    bv
                    & jnp.stack(
                        [
                            jnp.bool_(True),
                            s.locked[b] == team,
                            (s.locked[b] >= 0) & (s.locked[b] != team),
                            h == i,
                            h == mate,
                            (h >= 0) & (TEAM[jnp.maximum(h, 0)] != team),
                        ]
                    )
                ).astype(jnp.float32),
            ]
        r = s.room
        door_y = FLOOR + DOOR_H / 2
        room = [
            clip((jnp.array([r[0] - WALL_T / 2, door_y]) - me) * 0.2),
            clip((jnp.array([r[1] + WALL_T / 2, door_y]) - me) * 0.2),
            s.doors.astype(jnp.float32),
            sealed[team].astype(jnp.float32),
            in_room(s, me)[None].astype(jnp.float32),
        ]
        return jnp.concatenate(own + others + boxes + room)

    return jnp.stack([one(i) for i in range(ND)])


def observe_pair(s: State, vis=None, box_vis=None):
    if vis is None:
        vis, box_vis = team_sees(s)
    return observe_all(s, vis, box_vis), observe_all(s, jnp.ones((2, 2), bool), jnp.ones((2, NB), bool))


# ---------- step ----------


def frozen_mask(s: State):
    return (TEAM == 1) & (s.t < PREP)


def control_step(s: State, pilot, tgt, tv, residual, frozen):
    def pobs(i):
        p = Phys(s.pos[i], s.vel[i], s.th[i], s.om[i], s.f[i])
        return observe(p, tgt[i], tv[i], s.prev[i])

    obs = jnp.stack([pobs(i) for i in range(ND)])
    cmd = jnp.clip(pilot_forward(pilot, obs) + residual, -1.0, 1.0)
    target_f = (cmd + 1.0) * 0.5 * SPEC["max_thrust"]

    def sub(s, _):
        return fly(s, target_f, frozen), None

    pos0, bpos0 = s.pos, s.bpos
    s, _ = jax.lax.scan(sub, s, None, length=SUB)
    s = collide(s, frozen, pos0, bpos0)
    s = s._replace(th=jnp.mod(s.th + jnp.pi, 2 * jnp.pi) - jnp.pi, prev=jnp.where(frozen[:, None], s.prev, cmd))
    return s


def frame(s: State, seen):
    """What a replay needs to draw one control step."""
    return {
        "pos": s.pos,
        "th": s.th,
        "f": s.f / SPEC["max_thrust"],
        "bpos": s.bpos,
        "locked": s.locked,
        "held": s.held,
        "seen": seen,
    }


def step(s: State, pilot, cont, grab, lock, record=False):
    """One 10 Hz decision for all four drones. Returns (state, obs, critic_obs, reward (ND,), done, info[, frames])."""
    frozen = frozen_mask(s)
    s = grab_and_lock(s, grab, lock, frozen)
    a = jnp.clip(cont, -1.0, 1.0)
    tgt0 = s.pos + a[:, 0:2] * G["offset_scale"]
    tv = a[:, 2:4] * G["target_vel_scale"]
    residual = a[:, 4:6] * G["residual_scale"]
    lim = jnp.array([HW - 0.3, HH - 0.2])

    def ctrl(s, k):
        tgt = jnp.clip(tgt0 + tv * (k * DT), -lim, lim)
        s = control_step(s, pilot, tgt, tv, residual, frozen)
        return s, (frame(s, jnp.zeros(2, bool)) if record else None)

    s, frames = jax.lax.scan(ctrl, s, jnp.arange(REPEAT, dtype=jnp.float32))
    s = s._replace(t=s.t + 1)

    m = seen_matrix(s)
    hider_seen = jnp.any(m, 0)
    any_seen = jnp.any(hider_seen)
    seek = s.t > PREP
    r_h = jnp.where(seek, jnp.where(any_seen, -1.0, 1.0), 0.0)
    reward = jnp.stack([r_h, r_h, -r_h, -r_h])

    vis, box_vis = team_sees(s)
    opp_pos = jnp.stack([s.pos[2:], s.pos[:2]])
    mem = jnp.where(vis[..., None], opp_pos, s.mem)
    since = jnp.minimum(jnp.where(vis, 0.0, s.since + DEC_DT), G["memory_s"])
    s = s._replace(mem=mem, since=since)

    st = s.stats
    sealed = sealed_doors(s)
    at_prep_end = s.t == PREP
    hiders_in = jnp.sum(in_room(s, s.pos[:2]))
    key_denied = (s.locked[KEY] == 0) | in_room(s, s.bpos[KEY])
    st = st._replace(
        hidden=st.hidden + (seek & ~any_seen),
        in_room=jnp.where(at_prep_end, hiders_in, st.in_room),
        doors=jnp.sum(s.doors).astype(jnp.float32),
        sealed=jnp.where(at_prep_end, jnp.sum(sealed), st.sealed),
        locked_prep=jnp.where(at_prep_end, jnp.sum((s.locked == 0) & in_doorway(s)), st.locked_prep),
        key_denied=jnp.where(at_prep_end, key_denied, st.key_denied),
        unsealed=st.unsealed + jnp.sum(s.seal0 & ~sealed & seek & seekers_at_doors(s)),
        flips=st.flips + jnp.sum(jnp.cos(s.th) < -0.5),
    )
    seal0 = jnp.where(at_prep_end, sealed, s.seal0 & sealed)
    s = s._replace(stats=st, seal0=seal0)

    obs, cobs = observe_pair(s, vis, box_vis)
    done = s.t >= EP_LEN
    if record:
        frames["seen"] = jnp.broadcast_to(hider_seen, (REPEAT, 2))
        return s, obs, cobs, reward, done, st, frames
    return s, obs, cobs, reward, done, st


def settle(s: State) -> State:
    """Push anything that spawned inside a wall or another body out of it, before anyone moves."""
    c = jnp.concatenate([s.pos, s.bpos])
    c0 = c
    h = jnp.concatenate([jnp.broadcast_to(DRONE_HALF, (ND, 2)), BOX_HALF])
    v = jnp.zeros_like(c)
    for _ in range(6):
        c, v, _ = resolve(c, h, v, jnp.ones(ND + NB), jnp.ones(ND + NB, bool), s.wc, s.wh, c0)
    # A carried box was settled as a body of its own; hang it back under its drone.
    pos = c[:ND]
    hp = pick(s.held, pos) + jnp.stack([jnp.zeros(NB), hold_offset(BOX_HALF[:, 1])], -1)
    bpos = jnp.where((s.held >= 0)[:, None], hp, c[ND:])
    return s._replace(pos=pos, bpos=bpos, spawn=c[2:ND])


def reset_obs(key):
    s = settle(reset(key))
    s = s._replace(stats=s.stats._replace(spawn_sealed=jnp.sum(sealed_doors(s)).astype(jnp.float32)))
    obs, cobs = observe_pair(s)
    return s, obs, cobs


def valid_mask(s: State):
    """Agents whose actions matter this decision: everyone except frozen seekers."""
    return ~frozen_mask(s)
