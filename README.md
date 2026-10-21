# bicopter

A 2D drone with two fixed, downward-pointing motors, flown by a neural network. It chases your cursor, flips
upside down to dive faster, shrugs off wind and heavier payloads, and recovers when you grab it and throw it.
You can also evolve your own pilot from scratch in the browser, using every core.

## Results

`bun bench.ts` on the shipped brain (PPO, 120M steps, 45 min on 4 CPU cores) against a hand-tuned cascaded PD
controller:

| | PD | PPO |
| --- | --- | --- |
| Score on 24 jump courses (reward/step) | 0.405 | **0.526** |
| Same, heavy + laggy + weak motor | 0.243 | **0.457** |
| 6-11 m dive, time to settle | 2.58 s | **1.78 s** (flips over to thrust downward) |
| Corner to corner | 3.48 s | **2.49 s** |
| Recover from a spinning throw | 1.88 s | **1.69 s** |
| Distance behind a circling target | 1.09 m | 1.09 m |
| Hold position in 3 N gusty wind | 0.81 m | **0.52 m** |

The in-browser evolution strategy passes the PD controller within a few minutes on a laptop.

## Layout

| Path | What |
| --- | --- |
| `spec.json` | Physics constants, observation scaling, domain randomization, reward. Read by both sides. |
| `train/` | JAX environment and PPO trainer (PureJaxRL style, whole rollout and update jitted). |
| `web/` | Vite + TypeScript app: the same physics, the trained network, an evolution strategy in Web Workers. |

The TypeScript simulator is a line-for-line port of `train/bicopter/sim.py`. `web/tests/parity.test.ts` checks
both the physics and the network forward pass against fixtures generated from JAX, so a brain trained in Python
flies the same in the browser.

## Train

```sh
cd train
uv sync
uv run python train.py --steps 150e6 --out runs/main     # ~45k steps/s on 4 CPU cores
uv run python make_fixtures.py                          # after changing sim.py or the network export
python3 publish.py runs/main                            # copy brains into web/public/brains
```

Checkpoints are written at 1M, 3M, 10M, 30M and 100M steps, plus `brain_best.json`, ranked by a deterministic
evaluation once domain randomization is fully on.

## Web

```sh
cd web
bun install
bun run dev
bun test                 # physics and policy parity with JAX
bun bench.ts public/brains/ppo-best.json   # fixed-scenario benchmark vs the classic PD controller
bun run build
```

## What the network sees and controls

Observation (15): target offset (coarse and fine), velocity, target velocity, sin/cos of the angle, angular
velocity, current motor thrusts, previous command. Action (2): a thrust command per motor. Motors only push, have
first-order lag, and commands can arrive up to two control steps late.

Training randomizes mass, inertia, per-motor thrust, motor lag, latency, wind and gusts, and sensor noise. The
sliders in the app shade the range the network saw, so you can push it past what it was trained for and watch it
fail.
