from __future__ import annotations

from itertools import combinations

import numpy as np

from qksearch.gates import State

_AXES = {"rx": "X", "ry": "Y", "rz": "Z",
         "rzz": "ZZ", "rzzh": "ZZ", "rzzhn": "ZZ"}

Generator = tuple[tuple[str, int], ...]


def state_generators(state: State) -> frozenset[Generator]:
    gens: set[Generator] = set()
    for name, qubits, f in state:
        axes = _AXES.get(name)
        if axes is None:
            if f is not None:
                raise ValueError(f"parameterized gate with no known generator: {name!r}")
            continue
        gens.add(tuple(zip(axes, qubits)))
    return frozenset(gens)


def commute(g1: Generator, g2: Generator) -> bool:
    axes2 = {q: a for a, q in g2}
    anti = sum(1 for a, q in g1 if q in axes2 and axes2[q] != a)
    return anti % 2 == 0


def is_abelian(gens: frozenset[Generator]) -> bool:
    return all(commute(g1, g2) for g1, g2 in combinations(gens, 2))


_SYMPLECTIC = {"X": (1, 0), "Y": (1, 1), "Z": (0, 1)}


def symplectic(gen: Generator) -> tuple[int, int]:
    x = z = 0
    for axis, qubit in gen:
        bx, bz = _SYMPLECTIC[axis]
        x ^= bx << qubit
        z ^= bz << qubit
    return x, z


def pauli_closure(gens: frozenset[Generator]) -> set[tuple[int, int]]:
    codes = np.array(sorted({symplectic(g) for g in gens}), dtype=np.int64)
    n_qubits = max(int(codes.max()).bit_length(), 1)
    seen = np.zeros(1 << (2 * n_qubits), dtype=bool)
    seen[codes[:, 0] | (codes[:, 1] << n_qubits)] = True
    closure, frontier = codes, codes
    while len(frontier):
        anti = np.bitwise_count((frontier[:, None, 0] & closure[None, :, 1])
                                ^ (frontier[:, None, 1] & closure[None, :, 0])) & 1
        f, c = np.nonzero(anti)
        products = frontier[f] ^ closure[c]
        keys = np.unique(products[:, 0] | (products[:, 1] << n_qubits))
        keys = keys[~seen[keys]]
        seen[keys] = True
        frontier = np.stack([keys & ((1 << n_qubits) - 1), keys >> n_qubits], axis=1)
        closure = np.concatenate([closure, frontier])
    return {(int(x), int(z)) for x, z in closure}


class DlaCalculator:
    def __init__(self) -> None:
        self._cache: dict[frozenset[Generator], int] = {}

    def dimension(self, state: State) -> int:
        return self.dimension_of(state_generators(state))

    def dimension_of(self, gens: frozenset[Generator]) -> int:
        if not gens:
            return 0
        cached = self._cache.get(gens)
        if cached is not None:
            return cached
        dim = len(gens) if is_abelian(gens) else len(pauli_closure(gens))
        self._cache[gens] = dim
        return dim
