"""Train the spiral flow-matching teacher (checkpoint manifest: spiral FM).

Used by 2c's FM-vs-diffusion teacher comparison cell. Laptop-CPU bounded.

Run from the repo root:

    python scripts/train_spiral_fm.py
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
SAMPLE_STEPS = 32     # Euler steps for the diagnostic figure
EVAL_STEPS = (1, 2, 4, 8, 16, 32)  # energy-distance sweep over Euler step counts
N_EVAL = 2048
CKPT = Path("checkpoints/spiral_fm.msgpack")
FIG = Path("checkpoints/spiral_fm_samples.png")
# ----------------------------------------------------------------------------


def loss_fn(model, key_t, key_z, x):
    """Conditional flow-matching loss (linear path x_t = t x + (1-t) z,
    conditional velocity target u = x - z); x: (B, 2) -> scalar."""
    t = jax.random.uniform(key_t, (x.shape[0],))
    z = jax.random.normal(key_z, x.shape)
    x_t = t[:, None] * x + (1 - t[:, None]) * z
    return jnp.mean((model(x_t, t) - (x - z)) ** 2)


@nnx.jit
def train_step(model, optimizer, key):
    key_x, key_t, key_z = jax.random.split(key, 3)
    x = gm.targets.sample_spiral(key_x, BATCH)
    loss, grads = nnx.value_and_grad(loss_fn)(model, key_t, key_z, x)
    optimizer.update(grads=grads, model=model)
    return loss


@nnx.jit
def euler_step(model, x, t, dt):
    return x + dt * model(x, jnp.full(x.shape[0], t))


def sample(model, key, n, steps=SAMPLE_STEPS):
    """Integrate dx/dt = v(x, t) from t = 0 (z ~ N(0, I)) to t = 1; (n, 2)."""
    ts = jnp.linspace(0.0, 1.0, steps + 1)
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
        x_m = sample(model, key_sample, N_EVAL, steps=steps)
        ed = gm.metrics.energy_distance(x_m, x_target)
        print(f"energy distance @ {steps:3d}-step Euler: {ed:.4f}")
    x_model = sample(model, key_sample, N_EVAL)

    gm.checkpoints.save(model, CKPT)
    print(f"saved {CKPT}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    gm.plotting.scatter2d(x_target, ax=axes[0])
    axes[0].set_title("target (spiral)")
    gm.plotting.scatter2d(x_model, ax=axes[1])
    axes[1].set_title(f"FM model, {SAMPLE_STEPS}-step Euler")
    fig.savefig(FIG, bbox_inches="tight")
    print(f"saved {FIG}")


if __name__ == "__main__":
    main()
