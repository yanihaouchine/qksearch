"""Expressibility of a feature map (Sim et al., 2019).

KL divergence between the distribution of fidelities ``|<psi(x)|psi(x')>|^2``
over random input pairs and the Haar distribution ``(d-1)(1-F)^(d-2)``.
Lower means closer to Haar-random states.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from qksearch.gates import State

BOUND = float(np.pi / 2)

def n_features_of(state: State) -> int:
    """Number of input features read by the circuit."""
    indices = [i for _, _, f in state if f is not None for i in f]
    return 1 + max(indices) if indices else 0

def haar_bin_probabilities(edges: np.ndarray, dim: int) -> np.ndarray:
    """Haar probability of each fidelity bin, for states of dimension ``dim``."""
    if dim < 2:
        raise ValueError(f"dim must be >= 2, got {dim}")
    e = np.clip(np.asarray(edges, dtype=float), 0.0, 1.0)
    tail = (1.0 - e) ** (dim - 1)
    return tail[:-1] - tail[1:]

def sample_fidelities(backend, state: State, n_features: Optional[int] = None, *,
                      n_pairs: int = 5000, bound: float = BOUND,
                      seed: int = 0) -> np.ndarray:
    """Fidelities of ``n_pairs`` pairs of inputs drawn uniformly in ``[-bound, bound]^d``."""
    if n_pairs < 1:
        raise ValueError(f"n_pairs must be >= 1, got {n_pairs}")
    if bound <= 0.0:
        raise ValueError(f"bound must be > 0, got {bound}")
    d = n_features_of(state) if n_features is None else int(n_features)
    if d < 0:
        raise ValueError(f"n_features must be >= 0, got {d}")

    rng = np.random.default_rng(seed)
    X1 = rng.uniform(-bound, bound, size=(n_pairs, d))
    X2 = rng.uniform(-bound, bound, size=(n_pairs, d))
    psi1 = backend.states(state, X1)
    psi2 = backend.states(state, X2)
    overlap = np.einsum("ij,ij->i", psi1.conj(), psi2, optimize=True)
    return np.clip(np.abs(overlap) ** 2, 0.0, 1.0)

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """``KL(p || q)`` of two histograms, ignoring empty bins of ``p``."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError(f"p and q must have the same shape, {p.shape} vs {q.shape}")
    mask = p > 0.0
    if not mask.any():
        return 0.0
    ratio = np.divide(p, q, out=np.ones_like(p), where=(mask & (q > 0.0)))
    return float(np.sum(p[mask] * np.log(ratio[mask])))


def expressibility(backend, state: State, n_features: Optional[int] = None, *,
                   n_pairs: int = 5000, n_bins: int = 75,
                   bound: float = BOUND, seed: int = 0) -> float:
    """KL divergence to Haar of the fidelity histogram (``n_bins`` bins)."""
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    fidelities = sample_fidelities(backend, state, n_features,
                                   n_pairs=n_pairs, bound=bound, seed=seed)
    counts, edges = np.histogram(fidelities, bins=n_bins, range=(0.0, 1.0))
    p = counts / float(len(fidelities))
    q = haar_bin_probabilities(edges, backend.dim)
    return kl_divergence(p, q)
