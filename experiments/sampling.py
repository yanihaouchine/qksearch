from __future__ import annotations

import numpy as np

from qksearch.gates import State, iqp_gate_set
from qksearch.iqp import iqp_circuit


def random_diagonal_layer(n_qubits: int, n_gates: int,
                          rng: np.random.Generator) -> State:
    catalog = list(iqp_gate_set(n_qubits, n_qubits).placements(()))
    idx = rng.integers(0, len(catalog), size=n_gates)
    return tuple(catalog[i] for i in idx)


def random_iqp_circuit(n_qubits: int, n_gates: int,
                       rng: np.random.Generator) -> State:
    return iqp_circuit(random_diagonal_layer(n_qubits, n_gates, rng), n_qubits)
