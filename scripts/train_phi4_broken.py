"""Train the phi4 broken-Z2 flow (checkpoint manifest: phi4 collapsed).

The Z2 symmetry of the vector field is deliberately broken (bias + even
polynomial features), and the collapse is seeded by a constant push on the
conv bias at INITIALIZATION; training itself uses the same stable recipe
as the equivariant script.  Validated (diffrax solver, dt = 0.05):
BIAS_INIT = 0.3 with SEED = 1 collapses onto the -m mode (~96% of samples
with m < 0) with batch-stable eval ESS/N ~ 0.01 and healthy reweighted
Z2-even observables; BIAS_INIT = 0.1 does NOT tip it, BIAS_INIT = 0 usually
finds the symmetric solution; SEED = 0 at 0.3 collapses but its eval ESS is
wildly batch-dependent (0.001-0.7) — keep SEED = 1 for the shipped
checkpoint.  (The pre-diffrax recipe — noisy regime: batch 64, hot LR, no
clipping — no longer collapses cleanly with the more accurate solver.)

Run from the repo root (~2 min laptop CPU):

    python scripts/train_phi4_broken.py
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import bijx
import flax.nnx as nnx
import jax.numpy as jnp
from bijx.nn.features import FourierFeatures, PolynomialFeatures

import iaifi_gm as gm
from _phi4_common import build_model, train

# ----- hyperparameters (edit here) -----------------------------------------
SEED = 1            # validated with BIAS_INIT (see docstring); 0 gives unstable ESS
BIAS_INIT = 0.3     # constant init push on the conv bias — this is what tips it
L = 6
M2, LAM = gm.phi4.M2, gm.phi4.LAM   # pinned couplings (validated: -1.0, 0.5)
BATCH = 128
N_STEPS = 300
SOLVER_DT = 0.05  # fixed Tsit5 step size (20 steps, t = 0 -> 1)
LR = 3e-3           # peak of the warmup-cosine schedule
CLIP = 1.0          # same stable recipe as the equivariant script
KERNEL = (5, 5)
CKPT = Path("checkpoints/phi4_broken.msgpack")
FIG = Path("checkpoints/phi4_broken_mag.png")
# ----------------------------------------------------------------------------


def main():
    # Z2-breaking knobs vs the equivariant defaults: bias on, and even
    # polynomial powers (0, 2) added to the feature map.
    vf = bijx.ConvVF.build(
        KERNEL,
        (),
        use_bias=True,
        features=(
            partial(FourierFeatures, 49),
            partial(PolynomialFeatures, (0, 1, 2)),
        ),
        rngs=nnx.Rngs(params=SEED),
    )
    vf.conv.bias[...] = jnp.full_like(vf.conv.bias[...], BIAS_INIT)
    model = build_model(vf, L, SOLVER_DT, SEED)
    train(
        model,
        m2=M2, lam=LAM, batch=BATCH, n_steps=N_STEPS, lr=LR, clip=CLIP,
        ckpt=CKPT, fig=FIG, tag="broken-Z2",
    )


if __name__ == "__main__":
    main()
