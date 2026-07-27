"""Train the phi4 broken-Z2 flow (checkpoint manifest: phi4 collapsed).

The Z2 symmetry of the vector field is deliberately broken (bias + even
polynomial features).  Collapse needs the NOISY training regime: batch 64,
hotter peak LR, no gradient clipping — in the stable regime (batch 128 +
clipping) the symmetric solution is reachable and the Z2-breaking
parameters stay dormant, so the flow does NOT collapse.  At the pinned
couplings, SEED=0 collapses onto the +m mode (validated: raw <m> ~ +0.7,
~97% of samples with m > 0); seeds 1 and 2 do not collapse — collapse is
seed-dependent, keep SEED=0 for the shipped checkpoint.

Run from the repo root (~2 min laptop CPU):

    python scripts/train_phi4_broken.py
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import bijx
import flax.nnx as nnx
from bijx.nn.features import FourierFeatures, PolynomialFeatures

import iaifi_gm as gm
from _phi4_common import build_model, train

# ----- hyperparameters (edit here) -----------------------------------------
SEED = 0            # fixed seed; validated to collapse (1, 2 do not)
L = 6
M2, LAM = gm.phi4.M2, gm.phi4.LAM   # pinned couplings (validated: -1.0, 0.5)
BATCH = 64          # noisier gradients help tip the flow into one mode
N_STEPS = 300
RK4_STEPS = 16
LR = 5e-3           # peak of the warmup-cosine schedule
CLIP = None         # no clipping: instability seeds the symmetry breaking
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
    model = build_model(vf, L, RK4_STEPS, SEED)
    train(
        model,
        m2=M2, lam=LAM, batch=BATCH, n_steps=N_STEPS, lr=LR, clip=CLIP,
        ckpt=CKPT, fig=FIG, tag="broken-Z2",
    )


if __name__ == "__main__":
    main()
