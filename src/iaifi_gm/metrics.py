"""Sample-quality metrics.  Currently: energy distance (deterministic, no knobs)."""

from __future__ import annotations

import jax.numpy as jnp


def energy_distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Energy distance between two sample sets.

    ``E = 2 E||X - Y|| - E||X - X'|| - E||Y - Y'||`` estimated with V-statistics
    (all pairs, including the zero diagonal).  Deterministic given the samples;
    zero iff the distributions coincide (in the population limit).  Memory is
    O(n·m) — keep sample sizes at a few thousand.

    Args:
        x: Samples, shape ``(n, d)``.
        y: Samples, shape ``(m, d)``.

    Returns:
        Scalar energy distance.
    """

    def mean_pairwise(a, b):
        d2 = jnp.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return jnp.mean(jnp.sqrt(d2))

    return 2.0 * mean_pairwise(x, y) - mean_pairwise(x, x) - mean_pairwise(y, y)
