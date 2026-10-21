"""PPO in the PureJaxRL style: whole rollout + update jitted, data-parallel across CPU devices with pmap."""

from dataclasses import dataclass
from typing import Any, NamedTuple, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from . import env
from .sim import OBS_DIM


@dataclass(frozen=True)
class Config:
    envs_per_device: int = 4096
    num_steps: int = 32
    epochs: int = 3
    minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    reward_scale: float = 0.1
    actor_hidden: Sequence[int] = (128, 128)
    critic_hidden: Sequence[int] = (128, 128)
    init_log_std: float = -0.5
    dr_warmup: float = 0.15  # fraction of training to ramp domain randomization 0 -> 1


class Actor(nn.Module):
    hidden: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for h in self.hidden:
            x = jnp.tanh(nn.Dense(h, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        return nn.Dense(2, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)


class Critic(nn.Module):
    hidden: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for h in self.hidden:
            x = jnp.tanh(nn.Dense(h, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        return nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)[..., 0]


class Transition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    logp: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array


def gauss_logp(x, mean, log_std):
    return jnp.sum(-0.5 * ((x - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi), axis=-1)


class Trainer:
    def __init__(self, cfg: Config, total_steps: int, seed: int = 0):
        self.cfg = cfg
        self.n_dev = jax.local_device_count()
        self.batch = cfg.envs_per_device * cfg.num_steps * self.n_dev
        self.num_updates = max(1, total_steps // self.batch)
        self.actor = Actor(cfg.actor_hidden)
        self.critic = Critic(cfg.critic_hidden)

        key = jax.random.PRNGKey(seed)
        ka, kc, key = jax.random.split(key, 3)
        dummy = jnp.zeros((1, OBS_DIM))
        params = {
            "actor": self.actor.init(ka, dummy),
            "critic": self.critic.init(kc, dummy),
            "log_std": jnp.full((2,), cfg.init_log_std),
        }
        sched = optax.linear_schedule(cfg.lr, cfg.lr * 0.1, self.num_updates * cfg.epochs * cfg.minibatches)
        tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(sched, eps=1e-5))
        ts = TrainState.create(apply_fn=None, params=params, tx=tx)
        self.ts = jax.tree.map(lambda x: jnp.stack([x] * self.n_dev), ts)

        keys = jax.random.split(key, self.n_dev * cfg.envs_per_device + self.n_dev).reshape(-1, 2)
        env_keys = keys[: -self.n_dev].reshape(self.n_dev, cfg.envs_per_device, 2)
        self.keys = keys[-self.n_dev :]
        self.env_state, self.obs = jax.pmap(jax.vmap(env.reset_obs, in_axes=(0, None)), in_axes=(0, None))(
            env_keys, 0.0
        )
        self._update = jax.pmap(self._update_fn, axis_name="d", in_axes=(0, 0, 0, 0, None))
        self.update_i = 0

    def level(self):
        w = self.cfg.dr_warmup * self.num_updates
        return float(min(1.0, self.update_i / max(w, 1)))

    def policy(self, params, obs):
        return self.actor.apply(params["actor"], obs), params["log_std"]

    def _update_fn(self, ts, env_state, obs, key, level):
        cfg = self.cfg
        n = cfg.envs_per_device

        # Resetting is ~4x the cost of stepping, so draw one pool of fresh episodes per rollout.
        key, kp = jax.random.split(key)
        pool_s, pool_obs = jax.vmap(env.reset_obs, in_axes=(0, None))(jax.random.split(kp, n), level)

        def env_step(carry, _):
            ts, s, obs, key = carry
            key, ka = jax.random.split(key)
            mean, log_std = self.policy(ts.params, obs)
            a = mean + jnp.exp(log_std) * jax.random.normal(ka, mean.shape)
            logp = gauss_logp(a, mean, log_std)
            v = self.critic.apply(ts.params["critic"], obs)
            s2, obs2, r, done, trunc = jax.vmap(env.step, in_axes=(0, 0, None))(s, a, level)
            # Time-limit truncation is not a real terminal: bootstrap through it.
            v_next = self.critic.apply(ts.params["critic"], obs2)
            r_env = r
            r = r * cfg.reward_scale + cfg.gamma * v_next * trunc
            sr, obsr = pool_s, pool_obs
            sel = lambda x, y: jnp.where(done.reshape((-1,) + (1,) * (x.ndim - 1)), x, y)
            s2 = jax.tree.map(sel, sr, s2)
            obs2 = sel(obsr, obs2)
            return (ts, s2, obs2, key), (Transition(obs, a, logp, v, r, done), r_env)

        (ts, env_state, obs, key), (traj, raw_r) = jax.lax.scan(
            env_step, (ts, env_state, obs, key), None, cfg.num_steps
        )
        last_v = self.critic.apply(ts.params["critic"], obs)

        def gae(carry, t):
            adv, nv = carry
            delta = t.reward + cfg.gamma * nv * (1 - t.done) - t.value
            adv = delta + cfg.gamma * cfg.lam * (1 - t.done) * adv
            return (adv, t.value), adv

        _, advs = jax.lax.scan(gae, (jnp.zeros_like(last_v), last_v), traj, reverse=True)
        targets = advs + traj.value

        def loss_fn(params, b, adv, tgt):
            mean, log_std = self.policy(params, b.obs)
            logp = gauss_logp(b.action, mean, log_std)
            ratio = jnp.exp(logp - b.logp)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pl = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
            v = self.critic.apply(params["critic"], b.obs)
            vl = 0.5 * ((v - tgt) ** 2).mean()
            ent = jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))
            return pl + cfg.vf_coef * vl - cfg.ent_coef * ent, (pl, vl, ent)

        def epoch(carry, _):
            ts, key = carry
            key, kp = jax.random.split(key)
            size = cfg.num_steps * n
            perm = jax.random.permutation(kp, size)
            flat = lambda x: x.reshape((size,) + x.shape[2:])[perm].reshape((cfg.minibatches, -1) + x.shape[2:])
            data = jax.tree.map(flat, (traj, advs, targets))

            def mb(ts, d):
                b, adv, tgt = d
                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.params, b, adv, tgt)
                grads = jax.lax.pmean(grads, "d")
                return ts.apply_gradients(grads=grads), (loss, *aux)

            ts, losses = jax.lax.scan(mb, ts, data)
            return (ts, key), losses

        (ts, key), losses = jax.lax.scan(epoch, (ts, key), None, cfg.epochs)
        metrics = {
            "reward": jax.lax.pmean(raw_r.mean(), "d"),
            "done_rate": jax.lax.pmean(traj.done.mean(), "d"),
            "vloss": jax.lax.pmean(losses[2].mean(), "d"),
            "log_std": ts.params["log_std"].mean(),
        }
        return ts, env_state, obs, key, metrics

    def update(self):
        self.ts, self.env_state, self.obs, self.keys, m = self._update(
            self.ts, self.env_state, self.obs, self.keys, self.level()
        )
        self.update_i += 1
        return jax.tree.map(lambda x: float(x[0]), m)

    def host_params(self) -> Any:
        return jax.tree.map(lambda x: np.asarray(x[0]), self.ts.params)


def make_evaluate(actor, n_envs=256, steps=600):
    """Deterministic rollouts. Returns mean reward per step, mean distance, escape rate."""

    @jax.jit
    def evaluate(actor_params, key, level):
        s, obs = jax.vmap(env.reset_obs, in_axes=(0, None))(jax.random.split(key, n_envs), level)

        def body(carry, _):
            s, obs, alive = carry
            a = jnp.clip(actor.apply(actor_params, obs), -1, 1)
            s, obs, r, done, trunc = jax.vmap(env.step, in_axes=(0, 0, None))(s, a, level)
            d = jnp.linalg.norm(s.target - s.phys.pos, axis=-1)
            escaped = done & ~trunc
            return (s, obs, alive * (1 - escaped)), (r * alive, d * alive, alive)

        (_, _, alive), (r, d, al) = jax.lax.scan(body, (s, obs, jnp.ones(n_envs)), None, steps)
        return {
            "eval_reward": r.sum() / al.sum(),
            "eval_dist": d.sum() / al.sum(),
            "eval_escape": 1 - alive.mean(),
        }

    return evaluate
