"""Fashion-MNIST loading (plain idx-gzip files, no dataset framework).

Files (~30 MB total for the training split) are downloaded on first use into
``data/`` (gitignored) and parsed with numpy.  Nothing is downloaded at
import time.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

#: Class names, index-aligned with the integer labels.
CLASS_NAMES = (
    "t-shirt/top", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "ankle boot",
)

_BASE_URLS = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/",
    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
)

_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}


def _download(filename: str, data_dir: Path) -> Path:
    path = data_dir / filename
    if path.exists():
        return path
    data_dir.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for base in _BASE_URLS:
        try:
            print(f"downloading {base + filename} -> {path}")
            urllib.request.urlretrieve(base + filename, path)
            return path
        except Exception as err:  # try the next mirror
            last_err = err
    raise RuntimeError(f"could not download {filename}") from last_err


def _parse_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">HBB", f.read(4))
        _, dtype_code, ndim = magic
        assert dtype_code == 0x08, f"unexpected idx dtype code {dtype_code}"
        shape = struct.unpack(f">{ndim}I", f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_fashion_mnist(
    split: str = "train",
    data_dir: str | Path = "data",
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a fashion-MNIST split, downloading it on first use.

    Args:
        split: ``"train"`` (60k) or ``"test"`` (10k).
        data_dir: Download/cache directory (default ``data/``, gitignored).
        normalize: If true, images are float32 scaled to ``[-1, 1]``
            (the convention used by all image scripts/notebooks);
            otherwise raw uint8 in ``[0, 255]``.

    Returns:
        ``(images, labels)`` with shapes ``(N, 28, 28, 1)`` and ``(N,)``.
    """
    img_file, lbl_file = _FILES[split]
    data_dir = Path(data_dir)
    images = _parse_idx(_download(img_file, data_dir))[..., None]
    labels = _parse_idx(_download(lbl_file, data_dir)).astype(np.int32)
    if normalize:
        images = images.astype(np.float32) / 127.5 - 1.0
    return images, labels
