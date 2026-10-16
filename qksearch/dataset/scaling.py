from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from .base import BaseDataset

D = TypeVar("D", bound=BaseDataset)


@dataclass(frozen=True)
class AngleScaler:
    mean: np.ndarray
    std: np.ndarray
    scale: float = float(np.pi / 4)
    bound: float = float(np.pi / 2)

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        *,
        scale: float = float(np.pi / 4),
        bound: float = float(np.pi / 2),
    ) -> AngleScaler:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got ndim={X.ndim}")
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        if bound <= 0:
            raise ValueError(f"bound must be > 0, got {bound}")
        std = X.std(axis=0)
        constant_cols = np.flatnonzero(std == 0.0)
        if constant_cols.size:
            raise ValueError(
                f"constant columns, cannot standardize: {constant_cols.tolist()}"
            )
        return cls(X.mean(axis=0), std, float(scale), float(bound))

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got ndim={X.ndim}")
        if X.shape[1] != self.mean.size:
            raise ValueError(
                f"X has {X.shape[1]} features, the scaler expects {self.mean.size}"
            )
        return np.clip((X - self.mean) / self.std * self.scale, -self.bound, self.bound)

    def inverse_transform(self, angles: np.ndarray) -> np.ndarray:
        angles = np.asarray(angles, dtype=float)
        if angles.ndim != 2:
            raise ValueError(f"angles must be 2D, got ndim={angles.ndim}")
        if angles.shape[1] != self.mean.size:
            raise ValueError(
                f"angles has {angles.shape[1]} features, the scaler expects {self.mean.size}"
            )
        return angles / self.scale * self.std + self.mean

    def apply(self, data: D) -> D:
        return type(data)(self.transform(data.X), data.y)
