"""Statevector simulators that turn a circuit and a dataset into a Gram matrix.

:class:`NumpyBackend` is the fast, vectorized default; the Qiskit and
PennyLane backends simulate one sample at a time and serve as references
(``tests/check_equivalence.py`` checks NumPy against Qiskit).
"""

from __future__ import annotations

from .base import Backend
from .numpy_backend import NumpyBackend
from .pennylane_backend import PennyLaneBackend
from .qiskit_backend import QiskitBackend

__all__ = [
    "Backend",
    "NumpyBackend",
    "QiskitBackend",
    "PennyLaneBackend",
]
