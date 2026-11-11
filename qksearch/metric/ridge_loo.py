"""Leave-one-out error of kernel ridge regression, in closed form.

For ``A = (K + alpha I)^-1``, the leave-one-out residual of sample ``i`` is
``(A y)_i / A_ii``, so all ``n`` refits cost a single inversion (or one
eigendecomposition shared by a whole grid of ``alpha``).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class RidgeLooMse:
    """Negative leave-one-out MSE of kernel ridge, minimized over ``alphas``.

    Parameters
    ----------
    alphas
        Ridge regularization grid; the best value is kept per Gram matrix.
    normalize
        Divide the MSE by ``Var(y)``, so the score equals ``LOO R^2 - 1``.
    chunk_size
        Number of Gram matrices processed at once in :meth:`batch`.
    """

    def __init__(self, alphas: Sequence[float] = (1e-3,), normalize: bool = True,
                 chunk_size: int = 256) -> None:
        alphas = np.asarray(alphas, dtype=float).ravel()
        if alphas.size == 0 or np.any(alphas <= 0):
            raise ValueError("alphas must be a non-empty sequence of values > 0")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self.alphas = alphas
        self.normalize = normalize
        self.chunk_size = chunk_size

    def loo_errors(self, grams: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Leave-one-out residuals, shape ``(..., n_alphas, n)``."""
        yv = np.asarray(y, dtype=float).ravel()
        if self.alphas.size == 1:
            inv = np.linalg.inv(grams + self.alphas[0] * np.eye(yv.size))
            coef = inv @ yv
            diag = np.diagonal(inv, axis1=-2, axis2=-1)
            return (coef / diag)[..., None, :]
        w, V = np.linalg.eigh(grams)
        w = np.clip(w, 0.0, None)
        inv = 1.0 / (w[..., None, :] + self.alphas[:, None])
        proj = np.einsum("...ji,j->...i", V, yv)
        coef = np.einsum("...ij,...aj->...ai", V, inv * proj[..., None, :])
        diag = np.einsum("...ij,...aj->...ai", V * V, inv)
        return coef / diag

    def loo_mse(self, grams: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Leave-one-out MSE per alpha, shape ``(..., n_alphas)``."""
        grams = np.asarray(grams, dtype=float)
        out = np.empty(grams.shape[:-2] + (self.alphas.size,))
        flat_in = grams.reshape((-1,) + grams.shape[-2:])
        flat_out = out.reshape(-1, self.alphas.size)
        for start in range(0, flat_in.shape[0], self.chunk_size):
            stop = start + self.chunk_size
            errors = self.loo_errors(flat_in[start:stop], y)
            flat_out[start:stop] = (errors ** 2).mean(axis=-1)
        return out

    def _scale(self, y: np.ndarray) -> float:
        if not self.normalize:
            return 1.0
        var = float(np.var(y))
        return var if var > 0.0 else 1.0

    def best_alpha(self, gram: np.ndarray, y: np.ndarray) -> float:
        """Grid value of ``alpha`` with the lowest leave-one-out MSE."""
        return float(self.alphas[int(np.argmin(self.loo_mse(gram, y)))])

    def __call__(self, gram: np.ndarray, y: np.ndarray) -> float:
        return float(-self.loo_mse(gram, y).min() / self._scale(y))

    def batch(self, grams: np.ndarray, y: np.ndarray) -> np.ndarray:
        return -self.loo_mse(grams, y).min(axis=-1) / self._scale(y)
