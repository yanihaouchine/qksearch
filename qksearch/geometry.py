from __future__ import annotations

import numpy as np


def normalize_trace(K: np.ndarray) -> np.ndarray:
    trace = float(np.trace(K))
    if trace <= 0.0:
        return np.asarray(K, dtype=float)
    return K.shape[0] * np.asarray(K, dtype=float) / trace


def _sqrtm_psd(K: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh((K + K.T) / 2.0)
    return (vecs * np.sqrt(np.clip(vals, 0.0, None))) @ vecs.T


def geometric_difference(K_q: np.ndarray, K_c: np.ndarray, *,
                         lam: float = 1e-6) -> float:
    Kq = normalize_trace(K_q)
    Kc = normalize_trace(K_c)
    S = _sqrtm_psd(Kq)
    M = S @ np.linalg.solve(Kc + lam * np.eye(Kc.shape[0]), S)
    top = float(np.linalg.eigvalsh((M + M.T) / 2.0).max())
    return float(np.sqrt(max(top, 0.0)))
