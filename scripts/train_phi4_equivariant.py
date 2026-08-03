"""Train the phi4 equivariant flow (checkpoint manifest: phi4 equivariant).

`bijx.ConvVF.build` feature defaults (sin FourierFeatures,
PolynomialFeatures((1,)), use_bias=False) are Z2-equivariant — this is
2b §4's "fix" checkpoint.  Stable training regime: batch 128, peak LR 3e-3
(warmup-cosine), gradient clipping.  Laptop-CPU bounded (~3 min).

Run from the repo root:

    python scripts/train_phi4_equivariant.py
"""

from __future__ import annotations

from pathlib import Path

import bijx
import flax.nnx as nnx

import iaifi_gm as gm
from _phi4_common import build_model, train

# ----- hyperparameters (edit here) -----------------------------------------
SEED = 0
L = 6
M2, LAM = gm.phi4.M2, gm.phi4.LAM   # pinned couplings (validated: -1.0, 0.5)
BATCH = 128
N_STEPS = 300
SOLVER_DT = 0.05  # fixed Tsit5 step size (20 steps, t = 0 -> 1)
LR = 3e-3        # peak of the warmup-cosine schedule
CLIP = 1.0       # global-norm gradient clip (stability)
KERNEL = (5, 5)  # (3, 3) has too few D4 orbit parameters to converge here
CKPT = Path("checkpoints/phi4_equivariant.msgpack")
FIG = Path("checkpoints/phi4_equivariant_mag.png")
# ----------------------------------------------------------------------------


def main():
    vf = bijx.ConvVF.build(KERNEL, (), rngs=nnx.Rngs(params=SEED))
    model = build_model(vf, L, SOLVER_DT, SEED)
    train(
        model,
        m2=M2, lam=LAM, batch=BATCH, n_steps=N_STEPS, lr=LR, clip=CLIP,
        ckpt=CKPT, fig=FIG, tag="equivariant",
    )


if __name__ == "__main__":
    main()
