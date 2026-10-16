from __future__ import annotations

import numpy as np

from .base import RegressionDataset


def sine_target(X: np.ndarray) -> np.ndarray:
    return np.sin(np.asarray(X, dtype=float)).prod(axis=1)


def sine_dataset(
    n_samples: int = 60,
    n_features: int = 2,
    noise: float = 0.05,
    seed: int = 0,
) -> RegressionDataset:
    if n_features < 1:
        raise ValueError("n_features must be >= 1")
    if noise < 0:
        raise ValueError("noise must be >= 0")
    rng = np.random.default_rng(seed)
    X = rng.uniform(-np.pi / 2, np.pi / 2, size=(n_samples, n_features))
    y = sine_target(X) + rng.normal(scale=noise, size=n_samples)
    return RegressionDataset(X, y).standardized()
