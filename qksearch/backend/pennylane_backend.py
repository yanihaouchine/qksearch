from __future__ import annotations

from typing import Optional

import numpy as np

from qksearch.gates import IDENTITY_NAME, PHASE_POWERS, State

from .base import _cross_gram_from_states, _gram_from_states, feature_angle


class PennyLaneBackend:
    def __init__(self, n_qubits: int, device: str = "default.qubit") -> None:
        import pennylane as qml

        self.n_qubits = n_qubits
        self._qml = qml
        self._dev = qml.device(device, wires=n_qubits)

    def _apply(self, name: str, qubits: tuple[int, ...], angle: Optional[float]) -> None:
        qml = self._qml
        if name == IDENTITY_NAME:
            return
        if name == "h":
            qml.Hadamard(wires=qubits[0])
        elif name == "rx":
            qml.RX(angle, wires=qubits[0])
        elif name == "ry":
            qml.RY(angle, wires=qubits[0])
        elif name == "rz":
            qml.RZ(angle, wires=qubits[0])
        elif name == "cx":
            qml.CNOT(wires=list(qubits))
        elif name == "cz":
            qml.CZ(wires=list(qubits))
        elif name in ("rzz", "rzzh", "rzzhn"):
            qml.IsingZZ(angle, wires=list(qubits))
        elif name in PHASE_POWERS:
            arity, k = PHASE_POWERS[name]
            theta = k * np.pi / 4
            if arity == 1:
                qml.PhaseShift(theta, wires=qubits[0])
            else:
                qml.ControlledPhaseShift(theta, wires=list(qubits))
        else:
            raise ValueError(f"porte inconnue : {name!r}")

    def _states(self, state: State, X: np.ndarray) -> list[np.ndarray]:
        qml = self._qml

        @qml.qnode(self._dev)
        def psi(x):
            for name, qubits, f in state:
                self._apply(name, qubits, None if f is None else feature_angle(x, f, name))
            return qml.state()

        return [np.asarray(psi(x)) for x in X]

    def gram(self, state: State, X: np.ndarray) -> np.ndarray:
        return _gram_from_states(self._states(state, X))

    def cross_gram(self, state: State, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return _cross_gram_from_states(self._states(state, X1),
                                       self._states(state, X2))
