"""Train the tiny spiral CNF (checkpoint manifest: tiny spiral CNF, fallback only).

NB1 §1 trains this fresh in class (~30 s budget); the checkpoint exists only as
a belt-and-braces fallback if the live train misbehaves.  Stack per the outline:
``bijx.AutoJacVF`` over a small MLP vector field + ``bijx.ContFlowRK4``, wrapped
in ``bijx.Transformed`` — samples AND exact log-density.

Trained by maximum likelihood (forward KL) on spiral samples.

Run from the repo root:

    python scripts/train_spiral_cnf.py
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import bijx
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
N_STEPS = 600
BATCH = 256
LR = 3e-3             # peak LR; cosine-decayed to 0 over N_STEPS
HIDDEN, DEPTH, TIME_DIM = 64, 2, 32
RK4_STEPS = 16
N_EVAL = 2048
CKPT = Path("checkpoints/spiral_cnf.msgpack")
FIG = Path("checkpoints/spiral_cnf_samples.png")
# ----------------------------------------------------------------------------


class SpiralVF(nnx.Module):
    """CNF vector-field body ``(t, x) -> dx/dt`` delegating to a TimeMLP.

    (bijx's VF contract is ``vf(t, x)``; ``gm.models.TimeMLP`` is ``model(x, t)``.
    The wrapper must be an nnx.Module — a bare lambda would hide the params
    from nnx.state/optimizer.)
    """

    def __init__(self, hidden: int, depth: int, time_dim: int, *, rngs: nnx.Rngs):
        self.net = gm.models.TimeMLP(
            dim=2, hidden=hidden, depth=depth, time_dim=time_dim, rngs=rngs
        )

    def __call__(self, t, x):
        return self.net(x, t)


def build_model(seed: int = SEED) -> bijx.Transformed:
    """Construct the CNF exactly as checkpointed (NB1 fallback load cell)."""
    vf = bijx.AutoJacVF(SpiralVF(HIDDEN, DEPTH, TIME_DIM, rngs=nnx.Rngs(params=seed)))
    flow = bijx.ContFlowRK4(vf, steps=RK4_STEPS)
    prior = bijx.IndependentNormal((2,), rngs=nnx.Rngs(sample=seed + 1))
    return bijx.Transformed(prior, flow)


def main():
    model = build_model()

    # NOTE: @nnx.jit around a train step that differentiates through
    # ContFlowRK4 fails ("No constant handler for DynamicJaxprTracer" in the
    # odeint_rk4 closure_convert adjoint).  Manual functionalization via
    # nnx.split/merge + plain jax.jit works and is ~30x faster than the eager
    # nnx.value_and_grad pattern (~10 ms/step vs ~300 ms/step, which is eager
    # nnx graph-traversal overhead, independent of batch/RK4 steps).
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    tx = optax.adam(optax.cosine_decay_schedule(LR, N_STEPS))
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, opt_state, key):
        x = gm.targets.sample_spiral(key, BATCH)

        def loss_fn(p):
            return -jnp.mean(nnx.merge(graphdef, p, rest).log_density(x))

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    losses = np.full(N_STEPS, np.nan)
    keys = jax.random.split(jax.random.key(SEED + 2), N_STEPS)
    t0 = time.perf_counter()
    for i in tqdm(range(N_STEPS), desc="train"):
        params, opt_state, loss = train_step(params, opt_state, keys[i])
        losses[i] = loss
    print(f"training wall-clock (incl. jit): {time.perf_counter() - t0:.1f} s")
    print(f"final NLL (mean of last 50): {losses[-50:].mean():.4f}")
    nnx.update(model, params)  # write trained params back onto the module

    x_model, _ = model.sample((N_EVAL,))
    key_ref, key_ref2 = jax.random.split(jax.random.key(SEED + 3))
    x_target = gm.targets.sample_spiral(key_ref, N_EVAL)
    ed_floor = gm.metrics.energy_distance(
        gm.targets.sample_spiral(key_ref2, N_EVAL), x_target
    )
    ed = gm.metrics.energy_distance(x_model, x_target)
    print(f"energy-distance noise floor (target vs target, n={N_EVAL}): {ed_floor:.4f}")
    print(f"energy distance (model vs target, n={N_EVAL}): {ed:.4f}")

    gm.checkpoints.save(model, CKPT)
    print(f"saved {CKPT}")

    def model_log_density(pts):
        flat = pts.reshape(-1, 2)
        return model.log_density(flat).reshape(pts.shape[:-1])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    gm.plotting.scatter2d(x_target, ax=axes[0])
    axes[0].set_title("target (spiral)")
    gm.plotting.scatter2d(x_model, ax=axes[1])
    axes[1].set_title(f"CNF samples (RK4, {RK4_STEPS} steps)")
    gm.plotting.density2d(model_log_density, ax=axes[2], n=100)
    axes[2].set_title("CNF exact density")
    fig.savefig(FIG, bbox_inches="tight")
    print(f"saved {FIG}")


if __name__ == "__main__":
    main()
