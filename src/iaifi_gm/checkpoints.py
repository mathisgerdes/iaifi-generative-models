"""Checkpoint I/O: plain msgpack serialization of ``nnx.state`` — no orbax.

By default only ``nnx.Param`` leaves are saved/restored (RNG state and other
mutable variables are excluded, so bijx models with internal ``Rngs`` are
safe to checkpoint).  Loading is in-place onto an already-constructed model
of the same architecture; construction stays visible in notebooks/scripts.
"""

from __future__ import annotations

from pathlib import Path

import flax.nnx as nnx
import flax.serialization


def _stringify_keys(tree):
    """msgpack maps require string keys; nnx list indices are ints."""
    if isinstance(tree, dict):
        return {str(k): _stringify_keys(v) for k, v in tree.items()}
    return tree


def _destringify_keys(tree):
    if isinstance(tree, dict):
        return {
            (int(k) if k.isdigit() else k): _destringify_keys(v)
            for k, v in tree.items()
        }
    return tree


def save(model: nnx.Module, path: str | Path, *, state_filter=nnx.Param) -> Path:
    """Serialize the model's parameter state to ``path`` (msgpack).

    Args:
        model: Any ``nnx.Module`` (including bijx flows).
        path: Target file, conventionally ``checkpoints/<name>.msgpack``.
        state_filter: nnx filter for which variables to save.

    Returns:
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = nnx.state(model, state_filter)
    pure = _stringify_keys(state.to_pure_dict())
    path.write_bytes(flax.serialization.msgpack_serialize(pure))
    return path


def load(model: nnx.Module, path: str | Path, *, state_filter=nnx.Param) -> nnx.Module:
    """Restore parameters saved by :func:`save` into ``model`` (in place).

    ``model`` must be a freshly constructed instance of the same architecture
    (same shapes); returns the updated model for convenience.
    """
    pure = _destringify_keys(flax.serialization.msgpack_restore(Path(path).read_bytes()))
    state = nnx.state(model, state_filter)
    state.replace_by_pure_dict(pure)
    nnx.update(model, state)
    return model
