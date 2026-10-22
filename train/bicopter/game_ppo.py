"""Self-play PPO for hide and seek: one shared policy for both roles, an omniscient critic, a pool of past opponents.

Same shape as ppo.py (whole rollout and update jitted, data-parallel with pmap), extended to four agents per
environment, a mixed action space (Gaussian continuous + two categorical heads), and loss masks for agents that are
frozen (seekers during prep) or played by a past snapshot.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from . import game
from .game import N_CONT, N_GRAB, N_LOCK, ND, OBS_DIM


@dataclass(frozen=True)
class Config:
    envs_per_device: int = 512
    num_steps: int = 40
    epochs: int = 2
    minibatches: int = 8
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0  # an entropy bonus drags the rarely-pressed buttons toward 50% presses
    max_grad_norm: float = 0.5
    reward_scale: float = 0.1
    actor_hidden: Sequence[int] = (256, 256)
    critic_hidden: Sequence[int] = (256, 256)
    # Pilot target offset and velocity start fairly wide; the motor residual starts nearly off.
    init_log_std: Sequence[float] = (-0.5, -0.5, -0.7, -0.7, -2.0, -2.0)
    pool_frac: float = 0.2  # share of environments where one role is played by a past snapshot


N_OUT = N_CONT + N_GRAB + N_LOCK


class MLP(nn.Module):
    hidden: Sequence[int]
    out: int
    out_scale: float

    @nn.compact
    def __call__(self, x):
        for h in self.hidden:
            x = jnp.tanh(nn.Dense(h, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        return nn.Dense(self.out, kernel_init=orthogonal(self.out_scale), bias_init=constant(0.0))(x)


class Transition(NamedTuple):
    obs: jax.Array  # (E, ND, OBS_DIM)
    cobs: jax.Array
    cont: jax.Array  # (E, ND, N_CONT)
    grab: jax.Array  # (E, ND)
    lock: jax.Array
    logp: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array  # (E,)
    act_mask: jax.Array  # (E, ND) learn the policy from this sample
    val_mask: jax.Array  # (E, ND) learn the value from this sample


def split_out(y):
    return y[..., :N_CONT], y[..., N_CONT : N_CONT + N_GRAB], y[..., N_CONT + N_GRAB :]


def log_prob(y, log_std, cont, grab, lock):
    mean, gl, ll = split_out(y)
    g = jnp.sum(-0.5 * ((cont - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi), -1)
    lg = jnp.take_along_axis(jax.nn.log_softmax(gl), grab[..., None], -1)[..., 0]
    lk = jnp.take_along_axis(jax.nn.log_softmax(ll), lock[..., None], -1)[..., 0]
    return g + lg + lk


def cat_entropy(logits):
    lp = jax.nn.log_softmax(logits)
    return -jnp.sum(jnp.exp(lp) * lp, -1)


def sample(key, y, log_std):
    k1, k2, k3 = jax.random.split(key, 3)
    mean, gl, ll = split_out(y)
    cont = mean + jnp.exp(log_std) * jax.random.normal(k1, mean.shape)
    grab = jax.random.categorical(k2, gl)
    lock = jax.random.categorical(k3, ll)
    return cont, grab, lock


def greedy(y):
    mean, gl, ll = split_out(y)
    return mean, jnp.argmax(gl, -1), jnp.argmax(ll, -1)


class Trainer:
    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        self.n_dev = jax.local_device_count()
        self.n_env = cfg.envs_per_device * self.n_dev
        self.batch = self.n_env * cfg.num_steps  # decisions per update, each with ND agents
        self.actor = MLP(cfg.actor_hidden, N_OUT, 0.01)
        self.critic = MLP(cfg.critic_hidden, 1, 1.0)
        self.pilot = game.load_pilot()

        key = jax.random.PRNGKey(seed)
        ka, kc, key = jax.random.split(key, 3)
        dummy = jnp.zeros((1, OBS_DIM))
        actor = self.actor.init(ka, dummy)
        # Toggles start rarely pressed (about 8%), so a grabbed box is carried and a lock stays put for a while.
        last = sorted(actor["params"], key=lambda k: int(k.split("_")[1]))[-1]
        bias = actor["params"][last]["bias"]
        bias = bias.at[N_CONT + 1].set(-2.5).at[N_CONT + N_GRAB + 1].set(-2.5)
        actor["params"][last]["bias"] = bias
        params = {
            "actor": actor,
            "critic": self.critic.init(kc, dummy),
            "log_std": jnp.array(cfg.init_log_std, jnp.float32),
        }
        tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr, eps=1e-5))
        ts = TrainState.create(apply_fn=None, params=params, tx=tx)
        self.ts = jax.tree.map(lambda x: jnp.stack([x] * self.n_dev), ts)

        n = cfg.envs_per_device
        keys = jax.random.split(key, self.n_dev * n + self.n_dev).reshape(-1, 2)
        env_keys = keys[: -self.n_dev].reshape(self.n_dev, n, 2)
        self.keys = keys[-self.n_dev :]
        self.env_state, self.obs, self.cobs = jax.pmap(jax.vmap(game.reset_obs))(env_keys)
        # Stagger the first episodes so episode ends (and resets) are spread over every rollout.
        t0 = jax.random.randint(key, (self.n_dev, n), 0, game.EP_LEN)
        # A staggered first episode that starts after prep never records its prep-end stats; drop its spawn seals too.
        st = self.env_state.stats
        self.env_state = self.env_state._replace(
            t=t0, stats=st._replace(spawn_sealed=jnp.where(t0 > game.PREP, 0.0, st.spawn_sealed))
        )

        # The last pool_frac of each device's environments put a past snapshot in one role: seekers in even
        # environments, hiders in odd ones.
        n_pool = int(round(n * cfg.pool_frac))
        self.n_main = n - n_pool
        e = np.arange(n)
        in_pool = e >= n - n_pool
        old_seekers = in_pool & (e % 2 == 0)
        old_hiders = in_pool & (e % 2 == 1)
        role = np.array([0, 0, 1, 1])
        self.opp_mask = jnp.asarray(
            (old_seekers[:, None] & (role[None] == 1)) | (old_hiders[:, None] & (role[None] == 0))
        )
        self._update = jax.pmap(self._update_fn, axis_name="d", in_axes=(0, 0, 0, 0, 0, None, None))
        self.update_i = 0
        self.pool = [self.snapshot_params()]
        self.pool_on = False

    def _update_fn(self, ts, env_state, obs, cobs, key, opp_actor, pool_on):
        cfg = self.cfg
        n = cfg.envs_per_device
        pilot = self.pilot
        opp_mask = self.opp_mask & pool_on

        key, kp = jax.random.split(key)
        pool_s, pool_obs, pool_cobs = jax.vmap(game.reset_obs)(jax.random.split(kp, n))

        def env_step(carry, _):
            ts, s, obs, cobs, key = carry
            key, ka, kb = jax.random.split(key, 3)
            log_std = ts.params["log_std"]
            y = self.actor.apply(ts.params["actor"], obs)
            # Only the pool environments can have a past snapshot playing; skip its forward pass elsewhere.
            nm = self.n_main
            y_old = self.actor.apply(opp_actor["actor"], obs[nm:])
            y_act = jnp.concatenate([y[:nm], jnp.where(opp_mask[nm:, :, None], y_old, y[nm:])])
            # Past snapshots act with their own exploration noise.
            std_act = jnp.where(opp_mask[..., None], opp_actor["log_std"], log_std)
            cont, grab, lock = sample(ka, y_act, std_act)
            logp = log_prob(y, log_std, cont, grab, lock)
            v = self.critic.apply(ts.params["critic"], cobs)[..., 0]
            valid = jax.vmap(game.valid_mask)(s)
            s2, obs2, cobs2, r, done, st = jax.vmap(game.step, in_axes=(0, None, 0, 0, 0))(s, pilot, cont, grab, lock)
            tr = Transition(obs, cobs, cont, grab, lock, logp, v, r * cfg.reward_scale, done, valid & ~opp_mask, ~opp_mask)
            # Episode statistics, summed over the environments that finished this step.
            dn = done.astype(jnp.float32)
            ep = jax.tree.map(lambda x: jnp.sum(x * dn.reshape((-1,) + (1,) * (x.ndim - 1)), 0), st)
            sel = lambda x, y: jnp.where(done.reshape((-1,) + (1,) * (x.ndim - 1)), x, y)
            s2 = jax.tree.map(sel, pool_s, s2)
            obs2 = sel(pool_obs, obs2)
            cobs2 = sel(pool_cobs, cobs2)
            return (ts, s2, obs2, cobs2, key), (tr, ep, dn.sum(), r[:, 0])

        (ts, env_state, obs, cobs, key), (traj, eps, n_done, r_h) = jax.lax.scan(
            env_step, (ts, env_state, obs, cobs, key), None, cfg.num_steps
        )
        last_v = self.critic.apply(ts.params["critic"], cobs)[..., 0]

        def gae(carry, t):
            adv, nv = carry
            nd = 1.0 - t.done[:, None]
            delta = t.reward + cfg.gamma * nv * nd - t.value
            adv = delta + cfg.gamma * cfg.lam * nd * adv
            return (adv, t.value), adv

        _, advs = jax.lax.scan(gae, (jnp.zeros_like(last_v), last_v), traj, reverse=True)
        targets = advs + traj.value

        def loss_fn(params, b, adv, tgt):
            log_std = params["log_std"]
            y = self.actor.apply(params["actor"], b.obs)
            logp = log_prob(y, log_std, b.cont, b.grab, b.lock)
            am = b.act_mask.astype(jnp.float32)
            vm = b.val_mask.astype(jnp.float32)
            na = jnp.maximum(am.sum(), 1.0)
            mu = (adv * am).sum() / na
            sd = jnp.sqrt((((adv - mu) ** 2) * am).sum() / na)
            adv = (adv - mu) / (sd + 1e-8)
            ratio = jnp.exp(logp - b.logp)
            pg = jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv)
            pl = -(pg * am).sum() / na
            v = self.critic.apply(params["critic"], b.cobs)[..., 0]
            vl = 0.5 * (((v - tgt) ** 2) * vm).sum() / jnp.maximum(vm.sum(), 1.0)
            _, gl, ll = split_out(y)
            ent = ((cat_entropy(gl) + cat_entropy(ll)) * am).sum() / na
            approx_kl = (((ratio - 1) - jnp.log(ratio)) * am).sum() / na
            return pl + cfg.vf_coef * vl - cfg.ent_coef * ent, (pl, vl, ent, approx_kl)

        def epoch(carry, _):
            ts, key = carry
            key, kp = jax.random.split(key)
            size = cfg.num_steps * n
            perm = jax.random.permutation(kp, size)
            # Shuffle whole environment-steps so the four agents of a step stay together; done is per env.
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
        ep = jax.tree.map(lambda x: jax.lax.psum(x.sum(0), "d"), eps)
        metrics = {
            "episodes": jax.lax.psum(n_done.sum(), "d"),
            "r_hider": jax.lax.pmean(r_h.mean(), "d"),
            "vloss": jax.lax.pmean(losses[2].mean(), "d"),
            "entropy": jax.lax.pmean(losses[3].mean(), "d"),
            "kl": jax.lax.pmean(losses[4].mean(), "d"),
            "log_std": ts.params["log_std"],
            "ep": ep,
        }
        return ts, env_state, obs, cobs, key, metrics

    def update(self, rng: np.random.Generator):
        # Opponent: mostly a recent snapshot, sometimes any snapshot from the whole history.
        k = len(self.pool)
        idx = rng.integers(max(0, k - 10), k) if rng.random() < 0.5 else rng.integers(0, k)
        opp = self.pool[idx]
        self.ts, self.env_state, self.obs, self.cobs, self.keys, m = self._update(
            self.ts, self.env_state, self.obs, self.cobs, self.keys, opp, self.pool_on
        )
        self.update_i += 1
        m = jax.tree.map(lambda x: np.asarray(x[0]), m)
        return m

    def snapshot_params(self):
        p = self.host_params()
        return {"actor": p["actor"], "log_std": p["log_std"]}

    def snapshot(self):
        self.pool.append(self.snapshot_params())
        self.pool = self.pool[-64:]
        self.pool_on = True

    def host_params(self) -> Any:
        return jax.tree.map(lambda x: np.asarray(x[0]), self.ts.params)


def episode_summary(ep, episodes):
    """Per-episode means of the env statistics, keyed for the log."""
    n = max(float(episodes), 1.0)
    st = game.Stats(*ep)
    doors = max(float(st.doors), 1.0)
    return {
        "hidden": float(st.hidden) / n / (game.EP_LEN - game.PREP),
        "in_room": float(st.in_room) / n / 2,
        "sealed": float(st.sealed) / doors,
        # Seals the hiders made, beyond the doors that started sealed.
        "sealed_gain": (float(st.sealed) - float(st.spawn_sealed)) / doors,
        "locked_prep": float(st.locked_prep) / n,
        "key_denied": float(st.key_denied) / n,
        "unsealed": float(st.unsealed) / max(float(st.sealed), 1.0),
        "key_unlocks": float(st.key_unlocks) / n,
        "grabs_h": float(st.grabs[0]) / n,
        "grabs_s": float(st.grabs[1]) / n,
        "locks_h": float(st.locks[0]) / n,
        "locks_s": float(st.locks[1]) / n,
        "throws": float(st.throws) / n,
        "flips": float(st.flips) / n / ND / game.EP_LEN,
    }


def make_recorder(actor, pilot, n_envs=4):
    """Plays n_envs full episodes with the stochastic policy and returns per-control-step frames."""

    @jax.jit
    def record(params, key):
        k0, key = jax.random.split(key)
        s, obs, _ = jax.vmap(game.reset_obs)(jax.random.split(k0, n_envs))
        layout = {"wc": s.wc, "wh": s.wh, "room": s.room, "doors": s.doors}

        def body(carry, _):
            s, obs, key = carry
            key, ka = jax.random.split(key)
            y = actor.apply(params["actor"], obs)
            cont, grab, lock = sample(ka, y, params["log_std"])
            s, obs, _, r, _, st, fr = jax.vmap(partial(game.step, record=True), in_axes=(0, None, 0, 0, 0))(
                s, pilot, cont, grab, lock
            )
            return (s, obs, key), (fr, r[:, 0])

        (s, _, _), (frames, r) = jax.lax.scan(body, (s, obs, key), None, game.EP_LEN)
        return layout, frames, r, s.stats

    return record
