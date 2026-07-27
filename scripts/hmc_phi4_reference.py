"""High-precision HMC reference for phi4 observables at the pinned couplings.

Generates the reference numbers (with error bars) that go into
``iaifi_gm.phi4.REFERENCE`` — bijx ships no HMC (only IMH), so this ~50-line
jitted HMC lives here for reproducibility.  Linked from 2b §3; never run in
class.  Errors come from independent chains (mean over chains, standard error
across chains; jackknife over chains for the Binder cumulant).

Run from the repo root (a few minutes on a laptop CPU with the defaults):

    python scripts/hmc_phi4_reference.py
"""

from __future__ import annotations

from pathlib import Path

import flax.nnx as nnx  # noqa: F401  (keeps import order consistent with other scripts)
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

import iaifi_gm as gm

# ----- hyperparameters (edit here) -----------------------------------------
SEED = 0
L = 6
M2, LAM = gm.phi4.M2, gm.phi4.LAM   # pinned couplings (validated: -1.0, 0.5)
N_CHAINS = 64
N_THERM = 500        # discarded trajectories per chain
N_TRAJ = 4000        # kept trajectories per chain
LEAPFROG_STEPS = 10
STEP_SIZE = 0.1
HIST_NPZ = Path("checkpoints/phi4_hmc_reference.npz")   # m-histogram data for 2b
# ----------------------------------------------------------------------------

action = lambda phi: gm.phi4.action(phi, M2, LAM)   # batched via autovmap
grad_action = jax.vmap(jax.grad(lambda phi: gm.phi4.action(phi, M2, LAM)))


@jax.jit
def hmc_step(key, phi):
    """One HMC trajectory for a batch of chains; phi: (C, L, L)."""
    key_mom, key_acc = jax.random.split(key)
    pi = jax.random.normal(key_mom, phi.shape)
    h_old = action(phi) + 0.5 * jnp.sum(pi**2, axis=(-2, -1))

    # leapfrog
    phi_new = phi
    pi = pi - 0.5 * STEP_SIZE * grad_action(phi_new)
    for _ in range(LEAPFROG_STEPS - 1):
        phi_new = phi_new + STEP_SIZE * pi
        pi = pi - STEP_SIZE * grad_action(phi_new)
    phi_new = phi_new + STEP_SIZE * pi
    pi = pi - 0.5 * STEP_SIZE * grad_action(phi_new)

    h_new = action(phi_new) + 0.5 * jnp.sum(pi**2, axis=(-2, -1))
    accept = jax.random.uniform(key_acc, (phi.shape[0],)) < jnp.exp(h_old - h_new)
    phi = jnp.where(accept[:, None, None], phi_new, phi)
    return phi, jnp.mean(accept)


def main():
    key = jax.random.key(SEED)
    key, key_init = jax.random.split(key)
    phi = 0.1 * jax.random.normal(key_init, (N_CHAINS, L, L))

    for _ in tqdm(range(N_THERM), desc="thermalize"):
        key, key_step = jax.random.split(key)
        phi, _ = hmc_step(key_step, phi)

    mags = np.empty((N_TRAJ, N_CHAINS))
    acc = np.empty(N_TRAJ)
    for i in tqdm(range(N_TRAJ), desc="measure"):
        key, key_step = jax.random.split(key)
        phi, acc[i] = hmc_step(key_step, phi)
        mags[i] = np.asarray(gm.phi4.magnetization(phi))
    print(f"mean acceptance: {acc.mean():.3f} (tune STEP_SIZE toward ~0.7-0.9)")

    # Z2-even observables: chain means -> standard error across independent
    # chains (chains are independent; per-chain autocorrelation is absorbed
    # into the chain mean).
    abs_by_chain = np.abs(mags).mean(axis=0)
    abs_mag = abs_by_chain.mean()
    abs_mag_err = abs_by_chain.std(ddof=1) / np.sqrt(N_CHAINS)

    sq_by_chain = (mags**2).mean(axis=0)
    mag_sq = sq_by_chain.mean()
    mag_sq_err = sq_by_chain.std(ddof=1) / np.sqrt(N_CHAINS)

    # Binder U4: jackknife over chains.
    def u4(m):
        return 1.0 - np.mean(m**4) / (3.0 * np.mean(m**2) ** 2)

    u4_full = u4(mags)
    jack = np.array(
        [u4(np.delete(mags, c, axis=1)) for c in range(N_CHAINS)]
    )
    u4_err = np.sqrt((N_CHAINS - 1) * np.mean((jack - jack.mean()) ** 2))

    # m-histogram data for the notebook's reference figure (2b §1/§4).
    hist, bin_edges = np.histogram(mags.ravel(), bins=61, range=(-1.5, 1.5), density=True)
    np.savez(
        HIST_NPZ,
        bin_edges=bin_edges,
        density=hist,
        L=L, m2=M2, lam=LAM,
        n_samples=mags.size,
    )
    print(f"saved m-histogram data to {HIST_NPZ}")

    n_samples = N_TRAJ * N_CHAINS
    print(f"<|m|> = {abs_mag:.5f} +- {abs_mag_err:.5f}")
    print(f"<m^2> = {mag_sq:.5f} +- {mag_sq_err:.5f}")
    print(f"U4    = {u4_full:.5f} +- {u4_err:.5f}")
    print("\npaste into src/iaifi_gm/phi4.py REFERENCE:")
    print(
        f"REFERENCE[({L}, {M2}, {LAM})] = {{\n"
        f'    "abs_mag": ({abs_mag:.5f}, {abs_mag_err:.5f}),\n'
        f'    "mag_sq": ({mag_sq:.5f}, {mag_sq_err:.5f}),\n'
        f'    "binder_u4": ({u4_full:.5f}, {u4_err:.5f}),\n'
        f'    "n_samples": {n_samples},\n'
        f"}}"
    )


if __name__ == "__main__":
    main()
