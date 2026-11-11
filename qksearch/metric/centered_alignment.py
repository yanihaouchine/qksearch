"""Centered kernel alignment with a Gaussian target kernel, for regression.

Both matrices are centered (``H K H`` with ``H = I - 11^T / n``, Cortes et al.,
2012) so a constant offset in the kernel does not inflate the score. The
target kernel is ``exp(-(y_i - y_j)^2 / 2 sigma^2)`` rather than ``y y^T``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _center(gram: np.ndarray) -> np.ndarray:
    row = gram.mean(axis=-1, keepdims=True)
    return gram - row - row.swapaxes(-1, -2) + row.mean(axis=-2, keepdims=True)


class CenteredAlignment:
    """Centered alignment between the Gram matrix and a Gaussian kernel on ``y``.

    Parameters
    ----------
    sigma
        Bandwidth of the target kernel; default: median of ``|y_i - y_j|``.
    """

    def __init__(self, sigma: Optional[float] = None) -> None:
        if sigma is not None and sigma <= 0:
            raise ValueError("sigma must be > 0")
        self.sigma = sigma
        self._y: Optional[np.ndarray] = None
        self._target: Optional[np.ndarray] = None
        self._target_norm: float = 0.0

    def bandwidth(self, y: np.ndarray) -> float:
        """``sigma`` if given, else the median pairwise distance of ``y``."""
        if self.sigma is not None:
            return float(self.sigma)
        yv = np.asarray(y, dtype=float).ravel()
        if yv.size < 2:
            return 1.0
        iu = np.triu_indices(yv.size, k=1)
        median = float(np.median(np.abs(yv[iu[0]] - yv[iu[1]])))
        if median > 0.0:
            return median
        std = float(yv.std())
        return std if std > 0.0 else 1.0

    def target_kernel(self, y: np.ndarray) -> np.ndarray:
        """Uncentered Gaussian kernel on the targets."""
        yv = np.asarray(y, dtype=float).ravel()
        sigma = self.bandwidth(yv)
        delta = yv[:, None] - yv[None, :]
        return np.exp(-(delta ** 2) / (2.0 * sigma ** 2))

    def _centered_target(self, y: np.ndarray) -> tuple[np.ndarray, float]:
        if self._y is not y or self._target is None:
            target = _center(self.target_kernel(y))
            self._y = y
            self._target = target
            self._target_norm = float(np.linalg.norm(target))
        return self._target, self._target_norm

    def __call__(self, gram: np.ndarray, y: np.ndarray) -> float:
        target, target_norm = self._centered_target(y)
        centered = _center(gram)
        denom = float(np.linalg.norm(centered)) * target_norm
        if denom == 0.0:
            return 0.0
        return float((centered * target).sum() / denom)

    def batch(self, grams: np.ndarray, y: np.ndarray) -> np.ndarray:
        target, target_norm = self._centered_target(y)
        centered = _center(grams)
        num = np.einsum("bij,ij->b", centered, target)
        denom = np.sqrt(np.einsum("bij,bij->b", centered, centered)) * target_norm
        return np.divide(num, denom, out=np.zeros_like(num), where=denom != 0.0)
