from __future__ import annotations

from itertools import combinations

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


class DlaCalculator:
    def __init__(self) -> None:
        self._qml = None
        self._cache: dict[frozenset[Generator], int] = {}

    def _pennylane(self):
        if self._qml is None:
            import pennylane as qml

            self._qml = qml
        return self._qml

    def _operator(self, gen: Generator):
        qml = self._pennylane()
        paulis = {"X": qml.X, "Y": qml.Y, "Z": qml.Z}
        op = paulis[gen[0][0]](gen[0][1])
        for axis, qubit in gen[1:]:
            op = op @ paulis[axis](qubit)
        return op

    def dimension(self, state: State) -> int:
        return self.dimension_of(state_generators(state))

    def dimension_of(self, gens: frozenset[Generator]) -> int:
        if not gens:
            return 0
        cached = self._cache.get(gens)
        if cached is not None:
            return cached
        dim = (len(gens) if is_abelian(gens)
               else len(self._pennylane().lie_closure([self._operator(g) for g in gens])))
        self._cache[gens] = dim
        return dim
