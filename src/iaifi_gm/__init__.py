"""Helper package for the IAIFI Summer School 2026 generative-models tutorial.

Canonical alias in every notebook: ``import iaifi_gm as gm``.

Submodules: ``targets`` (spiral/GMM), ``phi4`` (lattice action +
reference numbers), ``data`` (fashion-MNIST), ``models`` (TimeMLP, SmallUNet),
``plotting``, ``checkpoints`` (msgpack nnx.state I/O), ``metrics``
(energy distance).  All pedagogical code lives in the notebooks — this
package holds only targets/data/models/plotting/checkpoints/metrics.
"""

from . import checkpoints, data, metrics, models, phi4, plotting, targets

__version__ = "0.1.0"

__all__ = [
    "checkpoints",
    "data",
    "install_check",
    "metrics",
    "models",
    "phi4",
    "plotting",
    "targets",
]


def install_check() -> None:
    """One-cell environment check (run this in advance of the session)."""
    import bijx
    import flax
    import jax
    import jax.numpy as jnp
    import optax

    print(f"jax    {jax.__version__}  devices: {jax.devices()}")
    print(f"flax   {flax.__version__}")
    print(f"optax  {optax.__version__}")
    print(f"bijx   {bijx.__version__}")
    print(f"iaifi_gm {__version__}")

    grad_sq = jax.jit(jax.grad(lambda x: jnp.sum(x**2)))
    assert jnp.allclose(grad_sq(jnp.arange(3.0)), 2 * jnp.arange(3.0))
    print("jit/grad smoke test: ok — you are ready.")
