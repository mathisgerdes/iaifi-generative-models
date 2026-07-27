"""Network architectures: time-conditioned MLP (2D toys) and a small UNet (images).

Both models follow the call convention of the notebooks: ``model(x, t)`` where
``t`` is a scalar or a ``(B,)`` array in ``[0, 1]`` (t = 0 noise, t = 1 data).
The same classes serve as epsilon-networks, velocity fields, or CNF vector-field
bodies — only the training target differs.
"""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp


def sinusoidal_time_embedding(
    t: jax.Array, dim: int = 32, max_freq: float = 1000.0
) -> jax.Array:
    """Transformer-style sin/cos features of time; ``t: (...,) -> (..., dim)``.

    Frequencies are log-spaced in ``[1, max_freq]``; ``dim`` must be even.
    """
    half = dim // 2
    freqs = jnp.exp(jnp.linspace(0.0, jnp.log(max_freq), half))
    angles = t[..., None] * freqs
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


class TimeMLP(nnx.Module):
    """MLP with sinusoidal time embedding for 2D toys.

    Call: ``model(x, t)`` with ``x: (..., dim)``, ``t`` scalar or ``(...,)``
    → ``(..., dim)``.
    """

    def __init__(
        self,
        dim: int = 2,
        hidden: int = 128,
        depth: int = 3,
        time_dim: int = 32,
        *,
        rngs: nnx.Rngs,
    ):
        self.time_dim = time_dim
        widths = [dim + time_dim] + [hidden] * depth
        self.layers = nnx.List(
            nnx.Linear(n_in, n_out, rngs=rngs)
            for n_in, n_out in zip(widths[:-1], widths[1:])
        )
        self.out = nnx.Linear(hidden, dim, rngs=rngs)

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        t = jnp.broadcast_to(jnp.asarray(t, x.dtype), x.shape[:-1])
        emb = sinusoidal_time_embedding(t, self.time_dim)
        h = jnp.concatenate([x, emb], axis=-1)
        for layer in self.layers:
            h = jax.nn.silu(layer(h))
        return self.out(h)


class ResBlock(nnx.Module):
    """GroupNorm–SiLU–Conv residual block with additive time conditioning."""

    def __init__(self, c_in: int, c_out: int, time_dim: int, *, rngs: nnx.Rngs):
        self.norm1 = nnx.GroupNorm(c_in, num_groups=min(8, c_in), rngs=rngs)
        self.conv1 = nnx.Conv(c_in, c_out, (3, 3), rngs=rngs)
        self.time_proj = nnx.Linear(time_dim, c_out, rngs=rngs)
        self.norm2 = nnx.GroupNorm(c_out, num_groups=min(8, c_out), rngs=rngs)
        self.conv2 = nnx.Conv(c_out, c_out, (3, 3), rngs=rngs)
        self.skip = nnx.Conv(c_in, c_out, (1, 1), rngs=rngs) if c_in != c_out else None

    def __call__(self, x: jax.Array, emb: jax.Array) -> jax.Array:
        h = self.conv1(jax.nn.silu(self.norm1(x)))
        h = h + self.time_proj(jax.nn.silu(emb))[:, None, None, :]
        h = self.conv2(jax.nn.silu(self.norm2(h)))
        skip = x if self.skip is None else self.skip(x)
        return h + skip


class SmallUNet(nnx.Module):
    """Small UNet for 28x28 images (fashion-MNIST scale).

    Call: ``model(x, t)`` with ``x: (B, 28, 28, C)``, ``t`` scalar or ``(B,)``
    → ``(B, 28, 28, C)``.  Two downsamplings (28 → 14 → 7), skip connections,
    additive time conditioning in every residual block.
    """

    def __init__(
        self,
        channels: tuple[int, int, int] = (32, 64, 128),
        in_channels: int = 1,
        time_dim: int = 128,
        *,
        rngs: nnx.Rngs,
    ):
        c0, c1, c2 = channels
        self.time_dim = time_dim
        self.time_mlp1 = nnx.Linear(time_dim, time_dim, rngs=rngs)
        self.time_mlp2 = nnx.Linear(time_dim, time_dim, rngs=rngs)

        self.conv_in = nnx.Conv(in_channels, c0, (3, 3), rngs=rngs)
        self.enc0 = ResBlock(c0, c0, time_dim, rngs=rngs)
        self.pool0 = nnx.Conv(c0, c1, (3, 3), strides=2, rngs=rngs)  # 28 -> 14
        self.enc1 = ResBlock(c1, c1, time_dim, rngs=rngs)
        self.pool1 = nnx.Conv(c1, c2, (3, 3), strides=2, rngs=rngs)  # 14 -> 7
        self.mid1 = ResBlock(c2, c2, time_dim, rngs=rngs)
        self.mid2 = ResBlock(c2, c2, time_dim, rngs=rngs)
        self.up1 = nnx.ConvTranspose(c2, c1, (4, 4), strides=2, rngs=rngs)  # 7 -> 14
        self.dec1 = ResBlock(c1 + c1, c1, time_dim, rngs=rngs)
        self.up0 = nnx.ConvTranspose(c1, c0, (4, 4), strides=2, rngs=rngs)  # 14 -> 28
        self.dec0 = ResBlock(c0 + c0, c0, time_dim, rngs=rngs)
        self.norm_out = nnx.GroupNorm(c0, num_groups=min(8, c0), rngs=rngs)
        self.conv_out = nnx.Conv(c0, in_channels, (3, 3), rngs=rngs)

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        t = jnp.broadcast_to(jnp.asarray(t, x.dtype), x.shape[:1])
        emb = sinusoidal_time_embedding(t, self.time_dim)
        emb = self.time_mlp2(jax.nn.silu(self.time_mlp1(emb)))

        h0 = self.enc0(self.conv_in(x), emb)          # (B, 28, 28, c0)
        h1 = self.enc1(self.pool0(h0), emb)           # (B, 14, 14, c1)
        h2 = self.mid2(self.mid1(self.pool1(h1), emb), emb)  # (B, 7, 7, c2)
        u1 = self.dec1(jnp.concatenate([self.up1(h2), h1], axis=-1), emb)
        u0 = self.dec0(jnp.concatenate([self.up0(u1), h0], axis=-1), emb)
        return self.conv_out(jax.nn.silu(self.norm_out(u0)))
