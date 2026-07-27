"""2D toy targets: the spiral (canonical), a GMM, and a Z2-symmetric posterior.

The spiral is ported from the web/notes figure math
(``src/figures/math/gaussian2d.ts`` + ``transforms2d.ts``): a two-blob
Gaussian mixture pushed through an area-preserving swirl (``det J = 1``, so
the pushforward density is exactly ``base_pdf(swirl^{-1}(x))``).
Domain of interest: ``[-3, 3]^2``.  This is the ONE spiral shared by the post
figures and all notebooks; bijx's ``gmm-spiral.ipynb`` spiral is a different
curve — do not mix them.

All functions are pure JAX and broadcast over leading batch dimensions.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

# --------------------------------------------------------------------------
# Spiral (canonical target)
# --------------------------------------------------------------------------

#: Plotting/evaluation domain used by figures and notebooks.
SPIRAL_DOMAIN = (-3.0, 3.0)

#: Two-blob mixture parameters (equal weights), ported from gaussian2d.ts.
SPIRAL_MEANS = jnp.array([[1.3, 0.0], [-1.3, 0.0]])
SPIRAL_VAR = 0.32  # isotropic variance of each blob

#: Swirl strength (transforms2d.ts: rotation angle = strength * radius).
SWIRL_STRENGTH = 1.2


def swirl(xy: jax.Array, strength: float = SWIRL_STRENGTH, sign: float = 1.0) -> jax.Array:
    """Area-preserving swirl about the origin.

    Rotates each point by an angle ``sign * strength * r`` where ``r`` is its
    radius (radius-dependent rotation, ``det J = 1``).  ``sign=-1`` gives the
    exact inverse.

    Args:
        xy: Points, shape ``(..., 2)``.

    Returns:
        Swirled points, shape ``(..., 2)``.
    """
    x, y = xy[..., 0], xy[..., 1]
    r = jnp.sqrt(x**2 + y**2)
    a = sign * strength * r
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.stack([c * x - s * y, s * x + c * y], axis=-1)


def sample_spiral(key: jax.Array, batch_size: int) -> jax.Array:
    """Draw spiral samples: mixture draw, then forward swirl.

    Args:
        key: PRNG key.
        batch_size: Number of samples.

    Returns:
        Samples of shape ``(batch_size, 2)``.
    """
    key_comp, key_norm = jax.random.split(key)
    comp = jax.random.bernoulli(key_comp, 0.5, (batch_size,))
    means = jnp.where(comp[:, None], SPIRAL_MEANS[0], SPIRAL_MEANS[1])
    base = means + jnp.sqrt(SPIRAL_VAR) * jax.random.normal(key_norm, (batch_size, 2))
    return swirl(base)


def _iso_normal_logpdf(x: jax.Array, mean: jax.Array, var: float) -> jax.Array:
    """Isotropic Gaussian log-pdf; ``x, mean: (..., d) -> (...)``."""
    d = x.shape[-1]
    quad = jnp.sum((x - mean) ** 2, axis=-1) / var
    return -0.5 * (quad + d * jnp.log(2 * jnp.pi * var))


def spiral_log_density(xy: jax.Array) -> jax.Array:
    """Exact spiral log-density (swirl has unit Jacobian: no correction term).

    Args:
        xy: Points, shape ``(..., 2)``.

    Returns:
        Log-density, shape ``(...,)``.
    """
    base = swirl(xy, sign=-1.0)
    logs = jnp.stack(
        [_iso_normal_logpdf(base, m, SPIRAL_VAR) for m in SPIRAL_MEANS], axis=-1
    )
    return logsumexp(logs, axis=-1) - jnp.log(2.0)


# --------------------------------------------------------------------------
# GMM (the only sanctioned secondary 2D target)
# --------------------------------------------------------------------------

#: Six equal-weight components on a ring of radius 2 (fits the [-3, 3] domain).
GMM_MEANS = 2.0 * jnp.stack(
    [
        jnp.cos(2 * jnp.pi * jnp.arange(6) / 6),
        jnp.sin(2 * jnp.pi * jnp.arange(6) / 6),
    ],
    axis=-1,
)
GMM_VAR = 0.09  # isotropic variance (std 0.3) of each component


def sample_gmm(key: jax.Array, batch_size: int) -> jax.Array:
    """Draw GMM samples; shape ``(batch_size, 2)``."""
    key_comp, key_norm = jax.random.split(key)
    idx = jax.random.randint(key_comp, (batch_size,), 0, GMM_MEANS.shape[0])
    means = GMM_MEANS[idx]
    return means + jnp.sqrt(GMM_VAR) * jax.random.normal(key_norm, (batch_size, 2))


def gmm_log_density(xy: jax.Array) -> jax.Array:
    """Exact GMM log-density; ``xy: (..., 2) -> (...,)``."""
    logs = jnp.stack(
        [_iso_normal_logpdf(xy, m, GMM_VAR) for m in GMM_MEANS], axis=-1
    )
    return logsumexp(logs, axis=-1) - jnp.log(GMM_MEANS.shape[0] * 1.0)


# --------------------------------------------------------------------------
# Z2-symmetric bimodal posterior (Notebook 2d)
# --------------------------------------------------------------------------
#
# Inference problem (outline-nb2d): parameter theta in R^2, prior
# N(0, tau^2 I), sign-symmetric likelihood
#   p(x | theta) = 1/2 N(x; theta, sigma^2 I) + 1/2 N(x; -theta, sigma^2 I)
# ("a detector that can't tell theta from -theta").  The posterior is exactly
# Z2-symmetric and bimodal for any dataset.

#: Pinned data recipe (verify the modes are cleanly separated before changing).
THETA_STAR = jnp.array([1.5, 0.8])
SIGMA_OBS = 0.5
TAU_PRIOR = 2.0
N_OBS = 4
OBS_SEED = 0


def sample_observations(
    key: jax.Array | None = None,
    n: int = N_OBS,
    theta_star: jax.Array = THETA_STAR,
    sigma: float = SIGMA_OBS,
) -> jax.Array:
    """Generate the observation set ``x_i ~ p(x | theta*)``.

    With no arguments this reproduces the fixed dataset used everywhere
    (seed ``OBS_SEED``).  Returns shape ``(n, 2)``.
    """
    if key is None:
        key = jax.random.key(OBS_SEED)
    key_sign, key_norm = jax.random.split(key)
    signs = jnp.where(jax.random.bernoulli(key_sign, 0.5, (n, 1)), 1.0, -1.0)
    return signs * theta_star + sigma * jax.random.normal(key_norm, (n, 2))


def log_posterior(
    theta: jax.Array,
    observations: jax.Array,
    sigma: float = SIGMA_OBS,
    tau: float = TAU_PRIOR,
) -> jax.Array:
    """Unnormalized log-posterior ``log p(theta) + sum_i log p(x_i | theta)``.

    Log-sum-exp stabilized.  ``theta: (..., 2) -> (...,)``.
    """
    log_prior = _iso_normal_logpdf(theta, jnp.zeros(2), tau**2)
    th = theta[..., None, :]  # (..., 1, 2) against observations (n, 2)
    log_plus = _iso_normal_logpdf(observations, th, sigma**2)
    log_minus = _iso_normal_logpdf(observations, -th, sigma**2)
    log_lik = logsumexp(jnp.stack([log_plus, log_minus], axis=-1), axis=-1) - jnp.log(2.0)
    return log_prior + jnp.sum(log_lik, axis=-1)


def posterior_gmm_params(
    observations: jax.Array,
    sigma: float = SIGMA_OBS,
    tau: float = TAU_PRIOR,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Exact normalized posterior as a ``2^n``-component Gaussian mixture.

    The product of ``n`` two-component likelihoods times the Gaussian prior
    expands over sign vectors ``s in {+-1}^n``; each term is Gaussian with
    precision ``lam = 1/tau^2 + n/sigma^2``, mean ``sum_i s_i x_i / (sigma^2 lam)``
    and mixture log-weight ``|sum_i s_i x_i|^2 / (2 sigma^4 lam)`` (up to the
    common normalization).

    Args:
        observations: Dataset of shape ``(n, 2)``; intended for small ``n``.

    Returns:
        ``(means, variances, weights)`` with shapes ``(2^n, 2)``, ``(2^n, 2)``,
        ``(2^n,)`` — ``weights`` are normalized probabilities;
        ``bijx.GaussianMixture`` takes logit weights (softmax applied
        internally), so pass ``jnp.log(weights)``.
    """
    n = observations.shape[0]
    signs = jnp.array(list(itertools.product((1.0, -1.0), repeat=n)))  # (2^n, n)
    lam = 1.0 / tau**2 + n / sigma**2
    b = jnp.einsum("sn,nd->sd", signs, observations) / sigma**2
    means = b / lam
    log_w = jnp.sum(b**2, axis=-1) / (2.0 * lam)
    weights = jax.nn.softmax(log_w)
    variances = jnp.full_like(means, 1.0 / lam)
    return means, variances, weights
