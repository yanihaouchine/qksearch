"""Immutable, validated containers for features ``X`` and targets ``y``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BaseDataset:
    """Features ``X`` (``n_samples`` x ``n_features``) and targets ``y`` (``n_samples``)."""

    X: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2D, got ndim={self.X.ndim}")
        if self.y.ndim != 1:
            raise ValueError(f"y must be 1D, got ndim={self.y.ndim}")
        if len(self.X) != len(self.y):
            raise ValueError(f"X and y are misaligned: {len(self.X)} != {len(self.y)}")

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


@dataclass(frozen=True)
class Dataset(BaseDataset):
    """Binary classification dataset with labels in ``{-1, +1}``."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not set(np.unique(self.y)) <= {-1, 1}:
            raise ValueError("y must contain only -1/+1 labels")


@dataclass(frozen=True)
class RegressionDataset(BaseDataset):
    """Regression dataset with real, non-constant targets."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not np.issubdtype(self.y.dtype, np.floating):
            raise ValueError(f"y must be floating point for regression, got {self.y.dtype}")
        if float(self.y.std()) == 0.0:
            raise ValueError("y is constant: no kernel can be told apart")

    def standardized(self) -> RegressionDataset:
        """Copy with zero-mean, unit-variance targets."""
        y = np.asarray(self.y, dtype=float)
        return RegressionDataset(self.X, (y - y.mean()) / y.std())
