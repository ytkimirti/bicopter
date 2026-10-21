"""Write web/tests/fixtures/ so the TypeScript sim and policy can be checked against JAX.

uv run python make_fixtures.py
"""

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from bicopter.export import save_brain  # noqa: E402
from bicopter.ppo import Actor, Config  # noqa: E402
from bicopter.sim import OBS_DIM, SPEC, Params, Phys, observe, physics_step  # noqa: E402

out = Path(__file__).resolve().parents[1] / "web" / "tests" / "fixtures"
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

step = jax.jit(physics_step)
obs_fn = jax.jit(observe)
cases = []
for i in range(12):
    p = Phys(
        pos=jnp.array(rng.uniform(-8, 8, 2)),
        vel=jnp.array(rng.normal(0, 3, 2)),
        th=jnp.float64(rng.uniform(-np.pi, np.pi)),
        om=jnp.float64(rng.normal(0, 6)),
        f=jnp.array(rng.uniform(0, SPEC["max_thrust"], 2)),
    )
    params = Params(
        mass=jnp.float64(rng.uniform(0.75, 1.35)),
        inertia=jnp.float64(SPEC["inertia"] * rng.uniform(0.8, 1.25)),
        tmax=jnp.array(SPEC["max_thrust"] * rng.uniform(0.88, 1.08, 2)),
        tau=jnp.float64(rng.uniform(0.015, 0.06)),
        latency=jnp.int32(0),
    )
    init = {"pos": p.pos.tolist(), "vel": p.vel.tolist(), "th": float(p.th), "om": float(p.om), "f": p.f.tolist()}
    wind = rng.normal(0, 2, 2)
    target = rng.uniform(-10, 10, 2)
    tvel = rng.normal(0, 4, 2)
    actions = rng.uniform(-1, 1, (40, 2))
    states, obs = [], []
    for a in actions:
        obs.append(np.asarray(obs_fn(p, jnp.array(target), jnp.array(tvel), jnp.array(a))).tolist())
        p = step(p, jnp.array(a), params, jnp.array(wind))
        states.append([float(p.pos[0]), float(p.pos[1]), float(p.vel[0]), float(p.vel[1]), float(p.th), float(p.om), float(p.f[0]), float(p.f[1])])
    cases.append(
        {
            "init": init,
            "params": {
                "mass": float(params.mass),
                "inertia": float(params.inertia),
                "tmaxL": float(params.tmax[0]),
                "tmaxR": float(params.tmax[1]),
                "tau": float(params.tau),
                "latency": 0,
            },
            "wind": wind.tolist(),
            "target": target.tolist(),
            "target_vel": tvel.tolist(),
            "actions": actions.tolist(),
            "states": states,
            "obs": obs,
        }
    )

(out / "physics.json").write_text(json.dumps(cases))

# Policy: a randomly initialised actor with non-trivial output layer, exported the same way training does.
jax.config.update("jax_enable_x64", False)
cfg = Config()
actor = Actor(cfg.actor_hidden)
params = actor.init(jax.random.PRNGKey(3), jnp.zeros((1, OBS_DIM)))
last = sorted(params["params"])[-1]
params["params"][last]["kernel"] = params["params"][last]["kernel"] * 100
save_brain(params, cfg, out / "brain.json", 0)
x = rng.normal(0, 1, (32, OBS_DIM)).astype(np.float32)
y = np.asarray(actor.apply(params, x))
(out / "policy.json").write_text(json.dumps({"obs": x.tolist(), "out": y.tolist()}))
print("wrote", out)
