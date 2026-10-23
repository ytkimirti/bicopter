"""Train hide and seek with self-play PPO. Writes metrics, checkpoints and replays the web app can play.

uv run python train_game.py --hours 12 --out runs/hideseek
"""

import argparse
import json
import os
import time

p = argparse.ArgumentParser()
p.add_argument("--hours", type=float, default=12.0)
p.add_argument("--out", default="runs/hideseek")
p.add_argument("--devices", type=int, default=8)
p.add_argument("--envs-per-device", type=int, default=512)
p.add_argument("--num-steps", type=int, default=40)
p.add_argument("--minibatches", type=int, default=8)
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--actor", default="256,256")
p.add_argument("--critic", default="256,256")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--snapshot-every", type=int, default=10, help="updates between opponent-pool snapshots")
p.add_argument("--replay-every", type=float, default=20.0, help="minutes between recorded checkpoints")
p.add_argument("--max-updates", type=int, default=0, help="stop after N updates (benchmarking)")
p.add_argument("--resume", default="", help="checkpoint .npz to continue from")
args = p.parse_args()

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.devices} " + os.environ.get("XLA_FLAGS", "")

import jax  # noqa: E402
import numpy as np  # noqa: E402

from bicopter import game  # noqa: E402
from bicopter.game_ppo import Config, Trainer, episode_summary, make_recorder  # noqa: E402
from bicopter.replay import save_checkpoint, load_checkpoint, write_replay  # noqa: E402

os.makedirs(args.out, exist_ok=True)
cfg = Config(
    envs_per_device=args.envs_per_device,
    num_steps=args.num_steps,
    minibatches=args.minibatches,
    epochs=args.epochs,
    actor_hidden=tuple(int(x) for x in args.actor.split(",")),
    critic_hidden=tuple(int(x) for x in args.critic.split(",")),
)
tr = Trainer(cfg, args.seed)
if args.resume:
    tr.ts, tr.update_i, done_decisions, elapsed0 = load_checkpoint(args.resume, tr.ts)
    tr.pool = [tr.snapshot_params()]
else:
    done_decisions, elapsed0 = 0, 0.0
record = make_recorder(tr.actor, tr.pilot)
rng = np.random.default_rng(args.seed)
print(
    f"devices={tr.n_dev} envs={tr.n_env} decisions/update={tr.batch} agents={game.ND} obs={game.OBS_DIM} "
    f"episode={game.EP_LEN} prep={game.PREP}",
    flush=True,
)

log = open(os.path.join(args.out, "metrics.jsonl"), "a")
t0 = time.time()
deadline = t0 + args.hours * 3600
next_replay = t0
u = 0
while time.time() < deadline:
    tu = time.time()
    m = tr.update(rng)
    tu = time.time() - tu
    u += 1
    done_decisions += tr.batch
    if tr.update_i % args.snapshot_every == 0:
        tr.snapshot()
    el = elapsed0 + time.time() - t0
    row = {
        "update": tr.update_i,
        "decisions": done_decisions,
        "agent_steps": done_decisions * game.ND,
        "physics_steps": done_decisions * game.REPEAT,
        "episodes": float(m["episodes"]),
        "elapsed": round(el, 1),
        "dps": round(tr.batch / tu),
        "r_hider": float(m["r_hider"]),
        "vloss": float(m["vloss"]),
        "entropy": float(m["entropy"]),
        "ent_grab": float(m["ent_grab"]),
        "ent_lock": float(m["ent_lock"]),
        "ent_lock_seeker": float(m["ent_lock_seeker"]),
        "kl": float(m["kl"]),
        "log_std": [round(float(x), 3) for x in m["log_std"]],
        **episode_summary(m["ep"], m["episodes"]),
    }
    log.write(json.dumps(row) + "\n")
    log.flush()
    if tr.update_i % 5 == 1 or u <= 3:
        keys = ["update", "decisions", "dps", "r_hider", "hidden", "in_room", "sealed", "locked_prep", "key_unlocks"]
        print("  ".join(f"{k}={row[k]:.3g}" if isinstance(row[k], float) else f"{k}={row[k]}" for k in keys), flush=True)
    last = time.time() >= deadline or (args.max_updates and u >= args.max_updates)
    if time.time() >= next_replay or last:
        next_replay = time.time() + args.replay_every * 60
        tag = f"{done_decisions // 1_000_000:05d}M"
        params = tr.host_params()
        save_checkpoint(os.path.join(args.out, f"ckpt_{tag}.npz"), tr.ts, tr.update_i, done_decisions, el)
        layout, frames, r, st = record(params, jax.random.PRNGKey(1000 + tr.update_i))
        write_replay(os.path.join(args.out, f"replay_{tag}.json"), layout, frames, r, st, row)
        print(f"checkpoint {tag}", flush=True)
    if last:
        break
print("done", flush=True)
