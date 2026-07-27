"""Train the spiral diffusion teacher (checkpoint manifest: spiral diffusion).

Used by 2a §3 as the comparison baseline AND by 2c as the teacher (same file).
Laptop-CPU bounded; validated target: visual sample quality in <= 1 min.

Run from the repo root:

    python scripts/train_spiral_diffusion.py
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm import tqdm

import iaifi_gm as gm

# ----- hyperparameters (edit here) -----------------------------------------
SEED = 0
N_STEPS = 12000
BATCH = 256
LR = 1e-3             # peak LR; cosine-decayed to 0 over N_STEPS
HIDDEN, DEPTH, TIME_DIM = 128, 3, 32
# Endpoint clip: t in [T_MIN, 1 - T_MIN]. Load-bearing for the sampler: the
# PF-ODE Euler coefficients scale like 1/t at the noise end, so a uniform grid
# starting at t = 1e-3 wrecks sample quality at <= 128 steps (measured ED
# 0.48 @ 50 steps vs 0.005 with T_MIN = 1e-2, all else equal).
T_MIN = 1e-2
SAMPLE_STEPS = 50     # Euler steps for the diagnostic figure (NB1 in-class count)
EVAL_STEPS = (8, 16, 32, 50, 128)  # energy-distance sweep over PF-ODE step counts
N_EVAL = 2048
CKPT = Path("checkpoints/spiral_diffusion.msgpack")
FIG = Path("checkpoints/spiral_diffusion_samples.png")
# ----------------------------------------------------------------------------

# VP trig schedule; convention: t = 0 noise -> t = 1 data.
alpha = lambda t: jnp.sin(jnp.pi * t / 2)
sigma = lambda t: jnp.cos(jnp.pi * t / 2)


def loss_fn(model, key_t, key_z, x):
    """Epsilon-matching loss; x: (B, 2) -> scalar."""
    t = jax.random.uniform(key_t, (x.shape[0],), minval=T_MIN, maxval=1 - T_MIN)
    z = jax.random.normal(key_z, x.shape)
    x_t = alpha(t)[:, None] * x + sigma(t)[:, None] * z
    return jnp.mean((model(x_t, t) - z) ** 2)


@nnx.jit
def train_step(model, optimizer, key):
    key_x, key_t, key_z = jax.random.split(key, 3)
    x = gm.targets.sample_spiral(key_x, BATCH)
    loss, grads = nnx.value_and_grad(loss_fn)(model, key_t, key_z, x)
    optimizer.update(grads=grads, model=model)
    return loss


@nnx.jit
def euler_step(model, x, t, dt):
    """PF-ODE Euler update: x += dt * (a x + b eps) with trig-schedule coeffs."""
    a = (jnp.pi / 2) / jnp.tan(jnp.pi * t / 2)
    b = -(jnp.pi / 2) / jnp.sin(jnp.pi * t / 2)
    eps = model(x, jnp.full(x.shape[0], t))
    return x + dt * (a * x + b * eps)


def sample(model, key, n, steps=SAMPLE_STEPS):
    """Integrate the PF-ODE t: T_MIN -> 1 - T_MIN from z ~ N(0, I); (n, 2)."""
    ts = jnp.linspace(T_MIN, 1 - T_MIN, steps + 1)
    x = jax.random.normal(key, (n, 2))
    for t, t_next in zip(ts[:-1], ts[1:]):
        x = euler_step(model, x, t, t_next - t)
    return x


def main():
    model = gm.models.TimeMLP(
        dim=2, hidden=HIDDEN, depth=DEPTH, time_dim=TIME_DIM, rngs=nnx.Rngs(params=SEED)
    )
    optimizer = nnx.Optimizer(
        model,
        optax.adam(optax.cosine_decay_schedule(LR, N_STEPS)),
        wrt=nnx.Param,
    )

    losses = np.full(N_STEPS, np.nan)
    keys = jax.random.split(jax.random.key(SEED + 1), N_STEPS)
    t0 = time.perf_counter()
    for i in tqdm(range(N_STEPS), desc="train"):
        losses[i] = train_step(model, optimizer, keys[i])
    print(f"training wall-clock (incl. jit): {time.perf_counter() - t0:.1f} s")
    print(f"final loss (mean of last 100): {losses[-100:].mean():.4f}")

    key_sample, key_ref, key_ref2 = jax.random.split(jax.random.key(SEED + 2), 3)
    x_target = gm.targets.sample_spiral(key_ref, N_EVAL)
    ed_floor = gm.metrics.energy_distance(
        gm.targets.sample_spiral(key_ref2, N_EVAL), x_target
    )
    print(f"energy-distance noise floor (target vs target, n={N_EVAL}): {ed_floor:.4f}")
    for steps in EVAL_STEPS:
        x_model = sample(model, key_sample, N_EVAL, steps=steps)
        ed = gm.metrics.energy_distance(x_model, x_target)
        print(f"energy distance @ {steps:4d}-step PF-ODE Euler: {ed:.4f}")

    gm.checkpoints.save(model, CKPT)
    print(f"saved {CKPT}")

    x_fig = sample(model, key_sample, N_EVAL, steps=SAMPLE_STEPS)
    x_fig128 = sample(model, key_sample, N_EVAL, steps=128)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    gm.plotting.scatter2d(x_target, ax=axes[0])
    axes[0].set_title("target (spiral)")
    gm.plotting.scatter2d(x_fig, ax=axes[1])
    axes[1].set_title(f"model, {SAMPLE_STEPS}-step PF-ODE Euler")
    gm.plotting.scatter2d(x_fig128, ax=axes[2])
    axes[2].set_title("model, 128-step PF-ODE Euler")
    fig.savefig(FIG, bbox_inches="tight")
    print(f"saved {FIG}")


if __name__ == "__main__":
    main()
