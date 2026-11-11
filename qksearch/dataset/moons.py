"""Two interleaving half-moons, a non-linearly separable classification benchmark."""

from __future__ import annotations

import numpy as np

from .base import Dataset


def moons_dataset(n_samples: int = 40, noise: float = 0.15, seed: int = 0) -> Dataset:
    """Two noisy half-moons, labels ``-1``/``+1``."""
    rng = np.random.default_rng(seed)
    half = n_samples // 2
    t1 = rng.uniform(0.0, np.pi, half)
    t2 = rng.uniform(0.0, np.pi, n_samples - half)
    outer = np.column_stack([np.cos(t1), np.sin(t1)])
    inner = np.column_stack([1.0 - np.cos(t2), 0.5 - np.sin(t2)])
    X = np.vstack([outer, inner]) + rng.normal(scale=noise, size=(n_samples, 2))
    y = np.concatenate([-np.ones(half), np.ones(n_samples - half)])
    order = rng.permutation(n_samples)
    return Dataset(X[order], y[order])
