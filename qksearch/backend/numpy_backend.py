from __future__ import annotations

from typing import Optional

import numpy as np

from qksearch.gates import HADAMARD_LAYER_NAME, IDENTITY_NAME, PHASE_POWERS, GateOp, State

from .base import _unit_diagonal, feature_angles

_DIAGONAL = (frozenset({IDENTITY_NAME, "rz", "rzz", "rzzh", "rzzhn", "cz"})
             | frozenset(PHASE_POWERS))

_SQRT2_INV = 1.0 / np.sqrt(2.0)
_H = np.array([[_SQRT2_INV, _SQRT2_INV], [_SQRT2_INV, -_SQRT2_INV]], dtype=complex)


class NumpyBackend:
    def __init__(self, n_qubits: int) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        self.n_qubits = n_qubits
        self.dim = 1 << n_qubits
        idx = np.arange(self.dim)
        self._idx = idx
        self._signs = np.array([1 - 2 * ((idx >> q) & 1) for q in range(n_qubits)],
                               dtype=float)
        self._const_phase: dict[GateOp, np.ndarray] = {}
        self._perm: dict[tuple[int, int], np.ndarray] = {}


    def _z_pattern(self, name: str, qubits: tuple[int, ...]) -> np.ndarray:
        if name == "rz":
            return self._signs[qubits[0]]
        if name in ("rzz", "rzzh", "rzzhn"):
            return self._signs[qubits[0]] * self._signs[qubits[1]]
        raise ValueError(f"unknown parameterized diagonal gate: {name!r}")

    def _constant_phase(self, name: str, qubits: tuple[int, ...]) -> np.ndarray:
        key: GateOp = (name, qubits, None)
        phase = self._const_phase.get(key)
        if phase is not None:
            return phase
        if name == "cz":
            theta, on = np.pi, qubits
        elif name in PHASE_POWERS:
            arity, k = PHASE_POWERS[name]
            theta, on = k * np.pi / 4, qubits[:arity]
        else:
            raise ValueError(f"unknown diagonal gate: {name!r}")
        active = np.ones(self.dim, dtype=bool)
        for q in on:
            active &= ((self._idx >> q) & 1).astype(bool)
        phase = np.where(active, np.exp(1j * theta), 1.0 + 0j)
        self._const_phase[key] = phase
        return phase

    def _permutation(self, control: int, target: int) -> np.ndarray:
        key = (control, target)
        perm = self._perm.get(key)
        if perm is None:
            flip = ((self._idx >> control) & 1).astype(bool)
            perm = np.where(flip, self._idx ^ (1 << target), self._idx)
            self._perm[key] = perm
        return perm

    def _apply_1q(self, psi: np.ndarray, U: np.ndarray, q: int) -> np.ndarray:
        m = psi.shape[0]
        high, low = 1 << (self.n_qubits - 1 - q), 1 << q
        t = psi.reshape(m, high, 2, low)
        sub = "ij,majb->maib" if U.ndim == 2 else "mij,majb->maib"
        return np.einsum(sub, U, t, optimize=True).reshape(m, self.dim)

    def diagonal_phase(self, op: GateOp, X: np.ndarray) -> Optional[np.ndarray]:
        name, qubits, f = op
        if name == IDENTITY_NAME or name not in _DIAGONAL:
            return None
        if f is None:
            return self._constant_phase(name, qubits)
        angles = feature_angles(X, f, name)
        return np.exp(-0.5j * angles[:, None] * self._z_pattern(name, qubits)[None, :])

    def apply(self, psi: np.ndarray, op: GateOp, X: np.ndarray) -> np.ndarray:
        name, qubits, f = op
        if name == IDENTITY_NAME:
            return psi
        phase = self.diagonal_phase(op, X)
        if phase is not None:
            return psi * phase
        angles = None if f is None else feature_angles(X, f, name)

        if name == "h":
            return self._apply_1q(psi, _H, qubits[0])
        if name == HADAMARD_LAYER_NAME:
            for q in range(self.n_qubits):
                psi = self._apply_1q(psi, _H, q)
            return psi
        if name == "cx":
            return psi[:, self._permutation(qubits[0], qubits[1])]
        if name in ("rx", "ry"):
            if angles is None:
                raise ValueError(f"{name} requires an angle")
            half = 0.5 * angles
            c, s = np.cos(half), np.sin(half)
            U = np.empty((len(angles), 2, 2), dtype=complex)
            if name == "rx":
                U[:, 0, 0] = U[:, 1, 1] = c
                U[:, 0, 1] = U[:, 1, 0] = -1j * s
            else:
                U[:, 0, 0] = U[:, 1, 1] = c
                U[:, 0, 1], U[:, 1, 0] = -s, s
            return self._apply_1q(psi, U, qubits[0])
        raise ValueError(f"unknown gate: {name!r}")


    def initial_states(self, n_samples: int) -> np.ndarray:
        psi = np.zeros((n_samples, self.dim), dtype=complex)
        psi[:, 0] = 1.0
        return psi

    def states(self, state: State, X: np.ndarray) -> np.ndarray:
        psi = self.initial_states(len(X))
        for op in state:
            psi = self.apply(psi, op, X)
        return psi


    @staticmethod
    def gram_from_states(psi: np.ndarray) -> np.ndarray:
        return _unit_diagonal(np.abs(psi.conj() @ psi.T) ** 2)

    @staticmethod
    def gram_batch(psis: np.ndarray) -> np.ndarray:
        return _unit_diagonal(np.abs(psis.conj() @ np.swapaxes(psis, 1, 2)) ** 2)

    def gram(self, state: State, X: np.ndarray) -> np.ndarray:
        return self.gram_from_states(self.states(state, X))

    def cross_gram(self, state: State, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        A, B = self.states(state, X1), self.states(state, X2)
        return np.abs(A.conj() @ B.T) ** 2
