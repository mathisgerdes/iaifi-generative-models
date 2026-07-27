"""SU(3) single-variable flow demo (2b appendix, provided code).

Transport beyond R^n: a conjugation-invariant target p(U) ~ exp(beta Re tr U)
on SU(3), Haar prior, CNF from a scalar potential (gradient + divergence via
``bijx.lie.value_grad_divergence``), integrated with a Crouch-Grossmann
solver that stays on the group.  Follows bijx's ``su3-flow.ipynb`` tutorial.

Run from the repo root (~1 min laptop CPU):

    python scripts/su3_demo.py
"""

from functools import partial

import bijx
import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import optax
from jax_autovmap import autovmap
from tqdm import tqdm

BETA = 2.0
BATCH = 64
N_STEPS = 100
LR = 2e-3


def target_log_density(u):
    """log p(U) = beta * Re tr U (unnormalized); conjugation-invariant."""
    return BETA * jnp.trace(u, axis1=-2, axis2=-1).real


class Potential(nnx.Module):
    """Scalar potential on conjugation-invariant features Re/Im tr U^k."""

    def __init__(self, width=64, *, rngs):
        self.lin1 = nnx.Linear(5, width, rngs=rngs)
        self.lin2 = nnx.Linear(width, width, rngs=rngs)
        self.out = nnx.Linear(
            width, 1, kernel_init=nnx.initializers.normal(1e-3), rngs=rngs
        )

    def __call__(self, t, u):
        tr1 = jnp.trace(u, axis1=-2, axis2=-1)
        tr2 = jnp.trace(u @ u, axis1=-2, axis2=-1)
        feats = jnp.stack(
            [tr1.real, tr1.imag, tr2.real, tr2.imag, jnp.asarray(t, tr1.real.dtype)]
        )
        h = nnx.gelu(self.lin1(feats))
        h = h + nnx.gelu(self.lin2(h))
        return self.out(h).squeeze()


class PotentialVF(nnx.Module):
    """CNF vector field: (grad, -div) of the potential on the group."""

    def __init__(self, potential):
        self.potential = potential

    @autovmap(t=0, u=2)
    def __call__(self, t, u):
        _, vec, div = bijx.lie.value_grad_divergence(
            partial(self.potential, t), u, bijx.lie.SU3_GEN
        )
        return vec, -div


def main():
    potential = Potential(rngs=nnx.Rngs(params=0))
    flow = bijx.ContFlowCG(
        PotentialVF(potential),
        tableau=bijx.cg.CG2,
        steps=10,
        x_type=bijx.cg.Unitary(
            transport_adjoint=True,
            derivative=bijx.cg.UnitaryDeriv(project_step=True),
        ),
    )
    model = bijx.Transformed(bijx.lie.HaarDistribution(3, rngs=nnx.Rngs(sample=1)), flow)
    optimizer = nnx.Optimizer(
        model,
        optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LR)),
        wrt=nnx.Param,
    )

    @nnx.jit
    def train_step(model, optimizer):
        def loss_fn(model):
            u, log_q = model.sample((BATCH,))
            # reverse KL up to log Z; same free-energy bound as the phi4 flow
            return jnp.mean(log_q - target_log_density(u))

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads=grads, model=model)
        return loss

    losses = np.full(N_STEPS, np.nan)
    for i in tqdm(range(N_STEPS), desc="train su3"):
        losses[i] = train_step(model, optimizer)

    u, log_q = model.sample((512,))
    ess = bijx.effective_sample_size(target_log_density(u), log_q)
    print(f"loss {losses[0]:.3f} -> {losses[-10:].mean():.3f}, ESS/N = {float(ess):.3f}")


if __name__ == "__main__":
    main()
