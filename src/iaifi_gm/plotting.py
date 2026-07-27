"""Matplotlib helpers for the notebooks: 2D scatter/density panels, image grids.

All helpers return the figure/axes and never call ``plt.show()`` (guarded
display: py-as-script must run headless).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def use_style() -> None:
    """Apply the shared plotting defaults (call once per notebook/script)."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "figure.figsize": (4.5, 4.0),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "scatter.marker": ".",
        }
    )


def scatter2d(
    samples,
    ax: plt.Axes | None = None,
    lim: float = 3.0,
    s: float = 4.0,
    alpha: float = 0.5,
    **kwargs,
) -> plt.Axes:
    """Scatter 2D samples on a square ``[-lim, lim]^2`` panel.

    Args:
        samples: Array of shape ``(N, 2)``.
    """
    if ax is None:
        _, ax = plt.subplots()
    samples = np.asarray(samples)
    ax.scatter(samples[:, 0], samples[:, 1], s=s, alpha=alpha, **kwargs)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    return ax


def density2d(
    log_density_fn,
    ax: plt.Axes | None = None,
    lim: float = 3.0,
    n: int = 200,
    **kwargs,
) -> plt.Axes:
    """Heatmap of ``exp(log_density_fn)`` on a ``[-lim, lim]^2`` grid.

    Args:
        log_density_fn: Callable ``(..., 2) -> (...)`` (broadcasting over the
            grid), e.g. ``gm.targets.spiral_log_density``.
    """
    if ax is None:
        _, ax = plt.subplots()
    grid = jnp.linspace(-lim, lim, n)
    xx, yy = jnp.meshgrid(grid, grid)
    pts = jnp.stack([xx, yy], axis=-1)
    dens = np.asarray(jnp.exp(log_density_fn(pts)))
    ax.imshow(
        dens, origin="lower", extent=(-lim, lim, -lim, lim), aspect="equal", **kwargs
    )
    return ax


def image_grid(
    images,
    nrow: int = 8,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot a grid of grayscale images.

    Args:
        images: Array of shape ``(N, H, W)`` or ``(N, H, W, 1)``, any range
            (each panel is shown with its own min/max).
        nrow: Images per row.
    """
    images = np.asarray(images)
    if images.ndim == 4:
        images = images[..., 0]
    n = images.shape[0]
    ncol = int(np.ceil(n / nrow))
    if figsize is None:
        figsize = (1.2 * nrow, 1.2 * ncol)
    fig, axes = plt.subplots(ncol, nrow, figsize=figsize, squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i], cmap="gray")
    fig.tight_layout()
    return fig


def device_report() -> None:
    """One-line 'where am I running' report (NB setup cells)."""
    print(f"jax {jax.__version__} — devices: {jax.devices()}")
