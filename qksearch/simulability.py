"""Certificates that a circuit's kernel can be computed classically.

A circuit is first normalized: Hadamard frames are tracked per qubit, gates
that cannot act (Z rotations on untouched ``|0>`` qubits, trailing constant
phases) are dropped, and the qubits are split into non-interacting blocks.
Each block is then certified by a structural rule:

- ``product``: a single qubit;
- ``iqp``: diagonal gates conjugated by Hadamards; the overlap is an average
  of ``2**k`` unit phases, exact or estimated by sampling bitstrings;
- ``matchgate``: Z and XX rotations on a path, i.e. free fermions, whose
  overlaps are determinants (Valiant; Terhal & DiVincenzo).

:func:`rff_gram` adds an empirical test: random Fourier features drawn
from the circuit's integer frequency spectrum (Rahimi & Recht, 2007).
"""

from __future__ import annotations

from functools import reduce
from typing import Optional

import numpy as np

from qksearch.backend import NumpyBackend
from qksearch.backend.base import ANGLE_ENCODERS, feature_angles
from qksearch.gates import (
    ANGLE_ENCODING,
    HADAMARD_LAYER_NAME,
    IDENTITY_NAME,
    PHASE_POWERS,
    GateOp,
    State,
)

ZZ_GATES = frozenset({"rzz", "rzzh", "rzzhn"})
ROTATIONS = ZZ_GATES | {"rz", "rx"}
CONSTANT_PHASES = frozenset(PHASE_POWERS) | {"cz"}
RULES = ("product", "iqp", "matchgate")

Gate = tuple[str, int, GateOp]


def _axis(name: str, parity: int) -> str:
    if name == "rx":
        base = "X"
    elif name in ROTATIONS or name in CONSTANT_PHASES:
        base = "Z"
    else:
        return "Y"
    if base == "X":
        return "Z" if parity else "X"
    return "X" if parity else "Z"


def _normalize(circuit: State, n_qubits: int) -> Optional[list[Gate]]:
    parity = [0] * n_qubits
    gates: list[Gate] = []
    for op in circuit:
        name, on, _ = op
        if name == IDENTITY_NAME:
            continue
        if name == HADAMARD_LAYER_NAME:
            parity = [p ^ 1 for p in parity]
        elif name == "h":
            parity[on[0]] ^= 1
        elif len({parity[q] for q in on}) != 1:
            return None
        else:
            gates.append((_axis(name, parity[on[0]]), parity[on[0]], op))
    return gates


def _drop_inert(gates: list[Gate]) -> list[Gate]:
    touched: set[int] = set()
    kept = []
    for gate in gates:
        axis, parity, (name, on, f) = gate
        if axis == "Z":
            live = tuple(q for q in on if q in touched)
            if not live or (f is None and len(live) < len(on)):
                continue
            if len(live) < len(on) and name == "rzz":
                gate = (axis, parity, ("rz", live, f))
        kept.append(gate)
        touched.update(gate[2][1])
    return kept


def _drop_trailing_constants(gates: list[Gate]) -> list[Gate]:
    if not gates:
        return gates
    axis = gates[-1][0]
    kept = []
    tail = True
    for gate in reversed(gates):
        tail = tail and gate[0] == axis
        if not (tail and gate[2][2] is None):
            kept.append(gate)
    return kept[::-1]


def _simplify(gates: list[Gate]) -> list[Gate]:
    while True:
        reduced = _drop_inert(_drop_trailing_constants(gates))
        if len(reduced) == len(gates):
            return reduced
        gates = reduced


def _components(gates: list[Gate], n_qubits: int) -> list[tuple[int, ...]]:
    parent = list(range(n_qubits))

    def root(q: int) -> int:
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    for _, _, (_, on, _) in gates:
        for q in on[1:]:
            parent[root(q)] = root(on[0])
    groups: dict[int, list[int]] = {}
    for q in range(n_qubits):
        groups.setdefault(root(q), []).append(q)
    return [tuple(group) for group in groups.values()]


def _restrict(gates: list[Gate], qubits: tuple[int, ...]) -> list[Gate]:
    position = {q: i for i, q in enumerate(qubits)}
    return [(axis, parity, (name, tuple(position[q] for q in on), f))
            for axis, parity, (name, on, f) in gates
            if all(q in position for q in on)]


def _blocks(circuit: State, n_qubits: int) -> Optional[list[tuple[tuple[int, ...], list[Gate]]]]:
    gates = _normalize(circuit, n_qubits)
    if gates is None:
        return None
    gates = _simplify(gates)
    return [(qubits, _restrict(gates, qubits)) for qubits in _components(gates, n_qubits)]


def components(circuit: State, n_qubits: int) -> list[tuple[int, ...]]:
    """Groups of qubits that never interact after simplification."""
    blocks = _blocks(circuit, n_qubits)
    if blocks is None:
        return [tuple(range(n_qubits))]
    return [qubits for qubits, _ in blocks]


def _path_order(gates: list[Gate], k: int) -> Optional[list[int]]:
    edges = {frozenset(on) for _, _, (_, on, _) in gates if len(on) == 2}
    neighbours: dict[int, set[int]] = {q: set() for q in range(k)}
    for a, b in map(tuple, edges):
        neighbours[a].add(b)
        neighbours[b].add(a)
    ends = [q for q, adj in neighbours.items() if len(adj) == 1]
    if len(edges) != k - 1 or any(len(adj) > 2 for adj in neighbours.values()) or not ends:
        return None
    order = [ends[0]]
    while len(order) < k:
        step = neighbours[order[-1]] - set(order[-2:-1])
        if len(step) != 1:
            return None
        order.append(step.pop())
    return order


def _is_matchgate(gates: list[Gate], k: int) -> bool:
    for axis, _, (name, on, _) in gates:
        if name not in ROTATIONS or axis != ("Z" if len(on) == 1 else "X"):
            return False
    return _path_order(gates, k) is not None


def _rule(gates: list[Gate], k: int) -> Optional[str]:
    if k == 1:
        return "product"
    if all(axis == "X" for axis, _, _ in gates):
        return "iqp"
    if _is_matchgate(gates, k):
        return "matchgate"
    return None


def certify(circuit: State, n_qubits: int) -> frozenset[str]:
    """Rules certifying every block of ``circuit``; empty if one block has none.

    Examples: ``frozenset({"iqp"})``, ``frozenset({"product", "matchgate"})``.
    """
    blocks = _blocks(circuit, n_qubits)
    if blocks is None:
        return frozenset()
    rules = set()
    for qubits, gates in blocks:
        rule = _rule(gates, len(qubits))
        if rule is None:
            return frozenset()
        rules.add(rule)
    return frozenset(rules)


def _product_gram(gates: list[Gate], X: np.ndarray) -> np.ndarray:
    h = ("h", (0,), None)
    ops = tuple(step for _, parity, op in gates
                for step in (h,) * parity + (op,) + (h,) * parity)
    psi = NumpyBackend(1).states(ops, X)
    return np.abs(psi.conj() @ psi.T) ** 2


def _phase(op: GateOp, X: np.ndarray, bits: np.ndarray) -> np.ndarray:
    name, on, f = op
    if f is None:
        arity, k = (2, 4) if name == "cz" else PHASE_POWERS[name]
        return np.outer(np.full(len(X), k * np.pi / 4), bits[:, list(on[:arity])].prod(axis=1))
    signs = 1 - 2 * bits[:, list(on)]
    return np.outer(-0.5 * feature_angles(X, f, name), signs.prod(axis=1))


def _iqp_gram(gates: list[Gate], X: np.ndarray, k: int, samples: Optional[int],
              rng: np.random.Generator) -> np.ndarray:
    if samples is None:
        bits = (np.arange(1 << k)[:, None] >> np.arange(k)) & 1
    else:
        bits = rng.integers(0, 2, size=(samples, k))
    phases = np.zeros((len(X), len(bits)))
    for _, _, op in gates:
        phases += _phase(op, X, bits)
    E = np.exp(1j * phases)
    return np.abs(E.conj() @ E.T / len(bits)) ** 2


def _rotate(M: np.ndarray, a: int, b: int, theta: np.ndarray) -> None:
    c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
    for axis in (1, 2):
        index_a = (slice(None), a) if axis == 1 else (slice(None), slice(None), a)
        index_b = (slice(None), b) if axis == 1 else (slice(None), slice(None), b)
        row_a, row_b = M[index_a].copy(), M[index_b].copy()
        M[index_a] = c * row_a - s * row_b
        M[index_b] = s * row_a + c * row_b


def _matchgate_gram(gates: list[Gate], X: np.ndarray, k: int) -> np.ndarray:
    position = {q: i for i, q in enumerate(_path_order(gates, k))}
    M = np.zeros((len(X), 2 * k, 2 * k))
    modes = np.arange(k)
    M[:, 2 * modes, 2 * modes + 1] = -1.0
    M[:, 2 * modes + 1, 2 * modes] = 1.0
    for _, _, (name, on, f) in gates:
        if len(on) == 1:
            a = 2 * position[on[0]]
        else:
            a = 2 * min(position[q] for q in on) + 1
        _rotate(M, a, a + 1, feature_angles(X, f, name))
    det = np.linalg.det(M[:, None] + M[None, :])
    return np.sqrt(np.abs(det)) / 2 ** k


def classical_gram(circuit: State, X: np.ndarray, n_qubits: int,
                   samples: Optional[int] = None, seed: int = 0) -> np.ndarray:
    """Gram matrix computed by the certificates, without a statevector.

    The kernel of a block-separable circuit is the product of the block
    kernels. ``samples`` switches IQP blocks from the exact sum over all
    bitstrings to a Monte Carlo estimate. Raises ``ValueError`` if a block is
    not certified.
    """
    blocks = _blocks(circuit, n_qubits)
    if blocks is None:
        raise ValueError("circuit mixes Hadamard frames inside a gate")
    rng = np.random.default_rng(seed)
    K = np.ones((len(X), len(X)))
    for qubits, gates in blocks:
        k = len(qubits)
        rule = _rule(gates, k)
        if rule is None:
            raise ValueError(f"no simulability certificate for qubits {qubits}")
        if rule == "product":
            K *= _product_gram(gates, X)
        elif rule == "iqp":
            K *= _iqp_gram(gates, X, k, samples, rng)
        else:
            K *= _matchgate_gram(gates, X, k)
    return K


Variable = tuple[str, tuple[int, ...]]


def encoding_spectrum(circuit: State) -> dict[Variable, int]:
    """How many gates read each encoded variable.

    A variable read by ``c`` gates enters the kernel with integer frequencies
    in ``[-c, c]``.
    """
    spectrum: dict[Variable, int] = {}
    for name, _, f in circuit:
        if f is not None:
            key = (ANGLE_ENCODING.get(name, "prod"), tuple(f))
            spectrum[key] = spectrum.get(key, 0) + 1
    return spectrum


def rff_frequencies(circuit: State, n_features: int,
                    seed: int = 0) -> tuple[list[Variable], np.ndarray]:
    """Random integer frequencies of shape ``(n_features, n_variables)``.

    The frequency of a variable read by ``c`` gates is drawn from
    ``Binomial(2c, 1/2) - c``.
    """
    spectrum = encoding_spectrum(circuit)
    counts = np.array(list(spectrum.values()), dtype=int)
    rng = np.random.default_rng(seed)
    omegas = rng.binomial(2 * counts, 0.5, size=(n_features, counts.size)) - counts
    return list(spectrum), omegas


def _encoded(X: np.ndarray, variables: list[Variable]) -> np.ndarray:
    columns = [ANGLE_ENCODERS[encoder](X[:, list(f)]) for encoder, f in variables]
    return np.stack(columns, axis=1) if columns else np.zeros((len(X), 0))


def rff_gram(circuit: State, X1: np.ndarray, X2: Optional[np.ndarray] = None,
             n_features: int = 1024, seed: int = 0) -> np.ndarray:
    """Random-Fourier-feature kernel ``mean_w cos(w . (phi(x) - phi(x')))``.

    ``phi`` are the encoded variables of the circuit and ``w`` its sampled
    frequencies (see :func:`rff_frequencies`).
    """
    variables, omegas = rff_frequencies(circuit, n_features, seed)
    phase1 = _encoded(X1, variables) @ omegas.T
    phase2 = phase1 if X2 is None else _encoded(X2, variables) @ omegas.T
    return (np.cos(phase1) @ np.cos(phase2).T
            + np.sin(phase1) @ np.sin(phase2).T) / n_features


def stabilizer_renyi_entropy(psis: np.ndarray) -> np.ndarray:
    """Stabilizer 2-Rényi entropy ``M_2`` ("magic") of each state (Leone et al., 2022).

    ``M_2 = 0`` exactly for stabilizer states; it grows with non-Clifford content.
    """
    psis = np.atleast_2d(psis)
    dim = psis.shape[-1]
    walsh = reduce(np.kron, [np.array([[1.0, 1.0], [1.0, -1.0]])] * (dim.bit_length() - 1))
    index = np.arange(dim)
    total = np.zeros(psis.shape[0])
    for a in range(dim):
        expectations = (psis[:, index ^ a].conj() * psis) @ walsh
        total += (np.abs(expectations) ** 4).sum(axis=1)
    return np.log2(dim / total)


class SimulabilityPenalty:
    """Penalty ``-weight`` on certified circuits, to steer the search away from them."""

    needs_gram = False

    def __init__(self, n_qubits: int, weight: float = 1.0) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        if weight < 0:
            raise ValueError("weight must be >= 0")
        self.n_qubits = n_qubits
        self.weight = weight

    def __call__(self, circuit: State) -> float:
        return -self.weight if certify(circuit, self.n_qubits) else 0.0
