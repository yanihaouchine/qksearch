"""Two Gaussian blobs, for smoke tests."""

from __future__ import annotations

import numpy as np

from .base import Dataset


def toy_dataset(n_samples: int = 20, n_features: int = 2, seed: int = 0) -> Dataset:
    """Two Gaussian blobs centred at ``-1`` and ``+1``, labels ``-1``/``+1``."""
    rng = np.random.default_rng(seed)
    half = n_samples // 2
    a = rng.normal(loc=-1.0, scale=0.5, size=(half, n_features))
    b = rng.normal(loc=+1.0, scale=0.5, size=(n_samples - half, n_features))
    X = np.vstack([a, b])
    y = np.concatenate([-np.ones(half), np.ones(n_samples - half)])
    order = rng.permutation(n_samples)
    return Dataset(X[order], y[order])
