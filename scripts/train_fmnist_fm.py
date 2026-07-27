"""Train the fashion-MNIST flow-matching model (checkpoint manifest: fMNIST FM).

Used by 2a's stretch section.  Written for a GPU machine — NOT laptop
material (downloads ~30 MB of data on first use, then trains for tens of
thousands of steps).

Run from the repo root:

    python scripts/train_fmnist_fm.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

import iaifi_gm as gm

# ----- hyperparameters (edit here) -----------------------------------------
SEED = 0
N_STEPS = 30_000
BATCH = 128
LR = 2e-4
CHANNELS = (32, 64, 128)
TIME_DIM = 128
SAMPLE_STEPS = 32     # Euler steps for the post-training sample grid
N_SAMPLE = 64
CKPT = Path("checkpoints/fmnist_fm.msgpack")
FIG = Path("checkpoints/fmnist_fm_samples.png")
# ----------------------------------------------------------------------------


def loss_fn(model, key_t, key_z, x):
    """Conditional flow-matching loss (linear path x_t = t x + (1-t) z,
    conditional velocity target u = x - z); x: (B, 28, 28, 1) -> scalar."""
    t = jax.random.uniform(key_t, (x.shape[0],))
    z = jax.random.normal(key_z, x.shape)
    t_b = t[:, None, None, None]
    x_t = t_b * x + (1 - t_b) * z
    return jnp.mean((model(x_t, t) - (x - z)) ** 2)


@nnx.jit
def train_step(model, optimizer, key, x):
    key_t, key_z = jax.random.split(key)
    loss, grads = nnx.value_and_grad(loss_fn)(model, key_t, key_z, x)
    optimizer.update(grads=grads, model=model)
    return loss


@nnx.jit
def euler_step(model, x, t, dt):
    return x + dt * model(x, jnp.full(x.shape[0], t))


def sample(model, key, n, steps=SAMPLE_STEPS):
    """Integrate dx/dt = v(x, t) from t = 0 (noise) to t = 1 (data)."""
    ts = jnp.linspace(0.0, 1.0, steps + 1)
    x = jax.random.normal(key, (n, 28, 28, 1))
    for t, t_next in zip(ts[:-1], ts[1:]):
        x = euler_step(model, x, t, t_next - t)
    return x


def main():
    images, _ = gm.data.load_fashion_mnist("train")  # (60000, 28, 28, 1) in [-1, 1]
    model = gm.models.SmallUNet(
        channels=CHANNELS, time_dim=TIME_DIM, rngs=nnx.Rngs(params=SEED)
    )
    optimizer = nnx.Optimizer(model, optax.adam(LR), wrt=nnx.Param)

    rng = np.random.default_rng(SEED)
    keys = jax.random.split(jax.random.key(SEED + 1), N_STEPS)
    losses = np.full(N_STEPS, np.nan)
    for i in tqdm(range(N_STEPS), desc="train"):
        batch = jnp.asarray(images[rng.choice(len(images), BATCH, replace=False)])
        losses[i] = train_step(model, optimizer, keys[i], batch)
        if (i + 1) % 1000 == 0:
            tqdm.write(f"step {i + 1}: loss {losses[max(0, i - 99):i + 1].mean():.4f}")

    gm.checkpoints.save(model, CKPT)
    print(f"saved {CKPT}")

    grid = sample(model, jax.random.key(SEED + 2), N_SAMPLE)
    fig = gm.plotting.image_grid(np.asarray(grid))
    fig.savefig(FIG, bbox_inches="tight")
    print(f"saved {FIG}")


if __name__ == "__main__":
    main()
