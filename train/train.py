"""Train a bicopter brain with PPO. Exports web-loadable JSON brains at milestones.

uv run python train.py --steps 300_000_000 --out runs/main
"""

import argparse
import os
import time

p = argparse.ArgumentParser()
p.add_argument("--steps", type=float, default=300e6)
p.add_argument("--out", default="runs/main")
p.add_argument("--devices", type=int, default=1)
p.add_argument("--envs-per-device", type=int, default=4096)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--eval-every", type=int, default=20)
p.add_argument("--milestones", default="1e6,3e6,10e6,30e6,100e6")
p.add_argument("--minibatches", type=int, default=32)
p.add_argument("--epochs", type=int, default=4)
p.add_argument("--num-steps", type=int, default=32)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--gamma", type=float, default=0.99)
p.add_argument("--actor", default="128,128")
p.add_argument("--critic", default="256,256")
p.add_argument("--max-updates", type=int, default=0, help="stop after N updates (benchmarking)")
args = p.parse_args()

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.devices} " + os.environ.get("XLA_FLAGS", "")

import jax  # noqa: E402

from bicopter.export import save_brain  # noqa: E402
from bicopter.ppo import Config, Trainer, make_evaluate  # noqa: E402

os.makedirs(args.out, exist_ok=True)
cfg = Config(
    envs_per_device=args.envs_per_device,
    minibatches=args.minibatches,
    epochs=args.epochs,
    num_steps=args.num_steps,
    lr=args.lr,
    gamma=args.gamma,
    actor_hidden=tuple(int(x) for x in args.actor.split(",")),
    critic_hidden=tuple(int(x) for x in args.critic.split(",")),
)
tr = Trainer(cfg, int(args.steps), args.seed)
evaluate = make_evaluate(tr.actor)
eval_key = jax.random.PRNGKey(12345)
milestones = sorted(float(x) for x in args.milestones.split(",") if x)
print(f"devices={tr.n_dev} batch={tr.batch} updates={tr.num_updates}", flush=True)

log = open(os.path.join(args.out, "log.tsv"), "a")
t0 = time.time()
best = -1e9
for u in range(tr.num_updates):
    m = tr.update()
    steps = (u + 1) * tr.batch
    while milestones and steps >= milestones[0]:
        ms = milestones.pop(0)
        save_brain(tr.host_params()["actor"], cfg, os.path.join(args.out, f"brain_{int(ms / 1e6)}M.json"), steps)
    if u % args.eval_every == 0 or u == tr.num_updates - 1:
        e = jax.tree.map(float, evaluate(tr.host_params()["actor"], eval_key, 1.0))
        e0 = jax.tree.map(float, evaluate(tr.host_params()["actor"], eval_key, 0.0))
        sps = steps / (time.time() - t0)
        line = (
            f"{u}\t{steps}\t{time.time() - t0:.0f}s\tsps={sps:.0f}\tlvl={tr.level():.2f}\tr={m['reward']:.3f}"
            f"\tstd={m['log_std']:.2f}\tvl={m['vloss']:.4f}\tdist={e['eval_dist']:.3f}\tdist0={e0['eval_dist']:.3f}"
            f"\tesc={e['eval_escape']:.3f}\tevr={e['eval_reward']:.3f}"
        )
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
        # Only rank checkpoints once domain randomization is fully on.
        if tr.level() >= 1.0 and e["eval_reward"] > best:
            best = e["eval_reward"]
            save_brain(tr.host_params()["actor"], cfg, os.path.join(args.out, "brain_best.json"), steps, e)
    if args.max_updates and u + 1 >= args.max_updates:
        break
save_brain(tr.host_params()["actor"], cfg, os.path.join(args.out, "brain_final.json"), steps)
print("done", flush=True)
