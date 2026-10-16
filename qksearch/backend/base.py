from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from qksearch.gates import ANGLE_ENCODING, State


class Backend(Protocol):
    def gram(self, state: State, X: np.ndarray) -> np.ndarray:
        ...

    def cross_gram(self, state: State, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        ...


ANGLE_ENCODERS = {
    "prod": lambda cols: np.prod(cols, axis=-1),
    "havlicek": lambda cols: np.prod(np.pi - cols, axis=-1),
    "havlicek_scaled": lambda cols: np.prod(np.pi - cols, axis=-1) / (2 * np.pi),
}


def _encoder(name: Optional[str]):
    return ANGLE_ENCODERS[ANGLE_ENCODING.get(name, "prod")]


def feature_angles(X: np.ndarray, f: tuple[int, ...],
                   name: Optional[str] = None) -> np.ndarray:
    return np.asarray(_encoder(name)(X[:, list(f)]), dtype=float)


def feature_angle(x: np.ndarray, f: tuple[int, ...],
                  name: Optional[str] = None) -> float:
    return float(_encoder(name)(x[list(f)]))


def _gram_from_states(vecs: list[np.ndarray]) -> np.ndarray:
    A = np.asarray(vecs)
    K = np.abs(A.conj() @ A.T) ** 2
    return _unit_diagonal(K)


def _unit_diagonal(K: np.ndarray) -> np.ndarray:
    n = K.shape[-1]
    K[..., np.arange(n), np.arange(n)] = 1.0
    return K


def _cross_gram_from_states(vecs1: list[np.ndarray], vecs2: list[np.ndarray]) -> np.ndarray:
    A = np.asarray(vecs1)
    B = np.asarray(vecs2)
    return np.abs(A.conj() @ B.T) ** 2
