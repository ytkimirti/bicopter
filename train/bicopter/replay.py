"""Checkpoints and replay files for hide and seek."""

import base64
import json
import math

import jax
import numpy as np
from flax import serialization

from . import game

# Replay channels per frame, stored as int16 scaled by 1000 (positions, angles, thrust) or raw (indices, flags).
CHANNELS = (
    [f"d{i}_{c}" for i in range(game.ND) for c in ("x", "y", "th", "fl", "fr")]
    + [f"b{i}_{c}" for i in range(game.NB) for c in ("x", "y")]
    + [f"b{i}_locked" for i in range(game.NB)]
    + [f"b{i}_held" for i in range(game.NB)]
    + [f"h{i}_seen" for i in range(2)]
)


def save_checkpoint(path, ts, update_i, decisions, elapsed):
    host = jax.tree.map(lambda x: np.asarray(x[0]), ts)
    with open(path, "wb") as f:
        f.write(serialization.to_bytes({"ts": host, "update": update_i, "decisions": decisions, "elapsed": elapsed}))


def load_checkpoint(path, ts):
    """Returns (ts replicated per device as numpy, which pmap shards on the next call), update, decisions, elapsed."""
    host = jax.tree.map(lambda x: np.asarray(x[0]), ts)
    with open(path, "rb") as f:
        raw = serialization.msgpack_restore(f.read())
    restored = serialization.from_state_dict(host, raw["ts"])
    n = jax.local_device_count()
    ts = jax.tree.map(lambda x: np.stack([x] * n), restored)
    # Checkpoints from before elapsed was saved have no such key.
    return ts, int(raw["update"]), int(raw["decisions"]), float(raw.get("elapsed", 0.0))


def finite(x):
    """Metrics rows with NaN would make the browser's JSON.parse reject the whole replay."""
    if isinstance(x, float):
        return x if math.isfinite(x) else 0.0
    if isinstance(x, list):
        return [finite(v) for v in x]
    return x


def b64_i16(a):
    return base64.b64encode(np.ascontiguousarray(a, "<i2").tobytes()).decode()


def episode_frames(frames, e, every=2):
    """(T, REPEAT, ...) frames of environment e flattened to control steps, every `every`th kept, as (n, CH) int16."""
    get = lambda k: np.asarray(frames[k])[:, e].reshape((-1,) + np.asarray(frames[k]).shape[3:])[::every]
    pos, th, f = get("pos"), get("th"), get("f")
    bpos, locked, held, seen = get("bpos"), get("locked"), get("held"), get("seen")
    n = pos.shape[0]
    drones = np.concatenate([pos, th[..., None], f], -1).reshape(n, -1)
    cont = np.concatenate([drones, bpos.reshape(n, -1)], -1)
    q = np.clip(np.round(cont * 1000), -32767, 32767)
    flags = np.concatenate([locked, held, seen.astype(np.int32)], -1)
    return np.concatenate([q, flags], -1).astype(np.int16)


def write_replay(path, layout, frames, rewards, stats, metrics):
    n_env = np.asarray(layout["room"]).shape[0]
    eps = []
    for e in range(n_env):
        walls = np.concatenate([np.asarray(layout["wc"])[e], np.asarray(layout["wh"])[e]], -1)
        st = jax.tree.map(lambda x: np.asarray(x)[e].tolist(), stats)
        data = episode_frames(frames, e)
        eps.append(
            {
                "walls": np.round(walls, 3).tolist(),
                "room": np.round(np.asarray(layout["room"])[e], 3).tolist(),
                "doors": np.asarray(layout["doors"])[e].astype(int).tolist(),
                "frames": int(data.shape[0]),
                "data": b64_i16(data),
                "hider_reward": float(np.asarray(rewards)[:, e].sum()),
                "stats": st._asdict(),
            }
        )
    doc = {
        "kind": "hideseek-replay",
        "fps": 1.0 / (game.DT * 2),
        "decision_hz": 1.0 / game.DEC_DT,
        "prep_frames": game.PREP * game.REPEAT // 2,
        "channels": CHANNELS,
        "box_half": np.asarray(game.BOX_HALF).tolist(),
        "drone_half": np.asarray(game.DRONE_HALF).tolist(),
        "metrics": {k: finite(v) for k, v in metrics.items()},
        "episodes": eps,
    }
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
