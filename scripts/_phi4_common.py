"""Shared reverse-KL training loop for the two phi4 flow scripts.

Validated recipe: ``bijx.ConvVF`` with a (5, 5) kernel + free-theory prior
(``bijx.FreeTheoryScaling``) + ``bijx.ContFlowRK4`` (16 steps), L = 6,
~300 Adam steps with a warmup-cosine LR schedule; loss = reverse KL
``mean(log_q + S)`` with ESS as auxiliary metric.  A (3, 3)-kernel /
white-noise-prior variant does NOT converge in 300 steps — do not change
kernel or prior without re-validating convergence and the seed-0 collapse.
"""

from __future__ import annotations

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

FREE_PRIOR_M2 = 1.0   # mass^2 of the free-theory prior (UV smoothing)


def build_model(vf, L: int, rk4_steps: int, seed: int) -> bijx.Transformed:
    flow = bijx.ContFlowRK4(vf, steps=rk4_steps)
    # Free-theory prior: takes care of damping the high-momentum modes so
    # the conv flow only has to build the double-well / ordering structure.
    # With a plain white-noise IndependentNormal prior the same flow gets
    # stuck near m=0 (validated).
    prior = bijx.Transformed(
        bijx.IndependentNormal((L, L), rngs=nnx.Rngs(sample=seed + 1)),
        bijx.FreeTheoryScaling(FREE_PRIOR_M2, (L, L), half=False),
    )
    return bijx.Transformed(prior, flow)


def train(
    model: bijx.Transformed,
    *,
    m2: float,
    lam: float,
    batch: int,
    n_steps: int,
    lr: float,
    clip: float | None,
    ckpt: Path,
    fig: Path,
    tag: str,
) -> None:
    # Warmup-cosine schedule: with only ~300 steps a flat small LR does not
    # converge; a hot flat LR eventually exploits RK4 integration error
    # (loss sinks below the free-energy bound while true ESS collapses).
    # Peak lr decayed fully to 0 by n_steps is the stable middle.
    schedule = optax.warmup_cosine_decay_schedule(0.0, lr, 20, n_steps)
    tx = optax.adam(schedule)
    if clip is not None:
        tx = optax.chain(optax.clip_by_global_norm(clip), tx)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # NOTE: no outer @nnx.jit here. bijx's ContFlowRK4 adjoint (odeint_rk4 with
    # closure_convert) fails under nnx.jit-of-grad with "No constant handler for
    # DynamicJaxprTracer" (verified on jax 0.9 and 0.11, bijx 1.3.1); the solver
    # is jitted internally, so plain nnx.value_and_grad is the working pattern.
    def train_step(model, optimizer):
        def loss_fn(model):
            phi, log_q = model.sample((batch,))
            target_ld = -gm.phi4.action(phi, m2, lam)
            # Reverse KL up to log Z (free-energy bound). Gradients flow
            # through the flow-generated samples (reparametrization) —
            # do NOT stop-gradient them.
            loss = jnp.mean(log_q - target_ld)
            ess = bijx.effective_sample_size(target_ld, log_q)
            return loss, ess
        (loss, ess), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(grads=grads, model=model)
        return loss, ess

    losses = np.full(n_steps, np.nan)
    ess_hist = np.full(n_steps, np.nan)
    for i in tqdm(range(n_steps), desc=f"train {tag}"):
        loss, ess = train_step(model, optimizer)
        losses[i], ess_hist[i] = loss, ess
    stride = max(n_steps // 12, 1)
    print("trajectory (step: loss, batch ESS/N):")
    for i in range(0, n_steps, stride):
        print(f"  {i:4d}: {losses[i]:8.3f}  {ess_hist[i]:.3f}")
    print(f"final loss {losses[-20:].mean():.3f}, final batch ESS/N {ess_hist[-20:].mean():.3f}")

    # Diagnostics on a large fresh batch. Batch-size-128 ESS estimates are
    # optimistic; report the 2048-sample value as the honest number.
    phi, log_q = model.sample((2048,))
    target_ld = -gm.phi4.action(phi, m2, lam)
    ess = float(bijx.effective_sample_size(target_ld, log_q))
    mag = np.asarray(gm.phi4.magnetization(phi))
    lw = np.array(target_ld - log_q, dtype=np.float64)
    lw -= lw.max()
    w = np.exp(lw)
    w /= w.sum()
    print(f"eval (2048 samples): ESS/N = {ess:.3f}")
    print(f"<m> = {mag.mean():+.4f}  (collapse flag: ~0 if both modes covered)")
    print(f"<|m|> raw = {np.abs(mag).mean():.4f}, reweighted = {np.sum(w * np.abs(mag)):.4f}")
    print(f"frac(m > 0) = {(mag > 0).mean():.3f}")

    gm.checkpoints.save(model, ckpt)
    print(f"saved {ckpt}")

    fig_obj, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(mag, bins=50)
    ax.set_xlabel("magnetization per site")
    ax.set_title(f"phi4 {tag}: m-histogram (2048 flow samples)")
    fig_obj.savefig(fig, bbox_inches="tight")
    print(f"saved {fig}")
