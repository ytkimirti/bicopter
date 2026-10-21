"""Write an actor network as JSON that web/src/sim/policy.ts loads."""

import json

import numpy as np

from .sim import OBS_DIM


def save_brain(actor_params, cfg, path, steps, evals=None):
    p = actor_params["params"]
    names = sorted(p.keys(), key=lambda k: int(k.split("_")[1]))
    layers = []
    for k in names:
        w = np.asarray(p[k]["kernel"], np.float32)
        b = np.asarray(p[k]["bias"], np.float32)
        layers.append(
            {
                "in": int(w.shape[0]),
                "out": int(w.shape[1]),
                "w": [float(f"{x:.6g}") for x in w.reshape(-1)],
                "b": [float(f"{x:.6g}") for x in b],
            }
        )
    doc = {
        "kind": "mlp",
        "activation": "tanh",
        "obs_dim": OBS_DIM,
        "steps": int(steps),
        "hidden": list(cfg.actor_hidden),
        "eval": {k: float(v) for k, v in (evals or {}).items()},
        "layers": layers,
    }
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
