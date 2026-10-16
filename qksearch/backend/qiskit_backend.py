from __future__ import annotations

from typing import Optional

import numpy as np

from qksearch.gates import IDENTITY_NAME, PHASE_POWERS, State

from .base import _cross_gram_from_states, _gram_from_states, feature_angle


class QiskitBackend:
    def __init__(self, n_qubits: int) -> None:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector

        self.n_qubits = n_qubits
        self._QuantumCircuit = QuantumCircuit
        self._Statevector = Statevector

    def _apply(self, qc, name: str, qubits: tuple[int, ...], angle: Optional[float]) -> None:
        if name == IDENTITY_NAME:
            return
        if name == "h":
            qc.h(qubits[0])
        elif name == "rx":
            qc.rx(angle, qubits[0])
        elif name == "ry":
            qc.ry(angle, qubits[0])
        elif name == "rz":
            qc.rz(angle, qubits[0])
        elif name == "cx":
            qc.cx(*qubits)
        elif name == "cz":
            qc.cz(*qubits)
        elif name in ("rzz", "rzzh", "rzzhn"):
            qc.rzz(angle, *qubits)
        elif name in PHASE_POWERS:
            arity, k = PHASE_POWERS[name]
            theta = k * np.pi / 4
            (qc.p if arity == 1 else qc.cp)(theta, *qubits)
        else:
            raise ValueError(f"porte inconnue : {name!r}")

    def _circuit(self, state: State, x: np.ndarray):
        qc = self._QuantumCircuit(self.n_qubits)
        for name, qubits, f in state:
            self._apply(qc, name, qubits, None if f is None else feature_angle(x, f, name))
        return qc

    def _states(self, state: State, X: np.ndarray) -> list[np.ndarray]:
        return [self._Statevector.from_instruction(self._circuit(state, x)).data
                for x in X]

    def gram(self, state: State, X: np.ndarray) -> np.ndarray:
        return _gram_from_states(self._states(state, X))

    def cross_gram(self, state: State, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return _cross_gram_from_states(self._states(state, X1),
                                       self._states(state, X2))
