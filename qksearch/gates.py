"""Gate vocabulary and gate sets that define the circuit search space.

A gate is a ``GateOp`` triple ``(name, qubits, features)``: ``features`` lists
the input coordinates whose product is the rotation angle, or ``None`` for a
constant gate. A circuit (``State``) is a tuple of such triples.

Diagonal gates commute, so a set of diagonal gates is a multiset: a
*canonical* :class:`ProductGateSet` only appends gates in catalogue order,
which visits each multiset exactly once.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Optional, Protocol, Sequence, Union

GateOp = tuple[str, tuple[int, ...], Optional[tuple[int, ...]]]
State = tuple[GateOp, ...]

IDENTITY_NAME = "id"
IDENTITY_OP: GateOp = (IDENTITY_NAME, (), None)

HADAMARD_LAYER_NAME = "hl"
HADAMARD_LAYER_OP: GateOp = (HADAMARD_LAYER_NAME, (), None)

PHASE_POWERS: dict[str, tuple[int, int]] = {
    "t": (1, 1), "s": (1, 2), "t3": (1, 3), "z": (1, 4),
    "t5": (1, 5), "sdg": (1, 6), "tdg": (1, 7),
    "ct": (2, 1), "cs": (2, 2), "ct3": (2, 3), "cz": (2, 4),
    "ct5": (2, 5), "csdg": (2, 6), "ctdg": (2, 7),
}

DIAGONAL_GATES: frozenset[str] = (frozenset({"rz", "rzz", "rzzh", "rzzhn",
                                             IDENTITY_NAME})
                                  | frozenset(PHASE_POWERS))

ANGLE_ENCODING: dict[str, str] = {"rzzh": "havlicek", "rzzhn": "havlicek_scaled"}


OrderSeed = Union[int, str, None]
RANDOM_ORDER = "random"


class GateSet(Protocol):
    """Which gates may be added to a circuit, and where."""

    def placements(self, state: State) -> Sequence[GateOp]:
        """Gates that may be added to ``state``."""
        ...

    def insert(self, state: State, op: GateOp) -> State:
        """Return ``state`` with ``op`` added."""
        ...


@dataclass(frozen=True)
class GateSpec:
    """A gate type: name, number of qubits, and whether its angle reads the data.

    ``symmetric`` gates act the same on ``(a, b)`` and ``(b, a)``, so only
    one ordering of their qubits is enumerated.
    """

    name: str
    arity: int
    uses_feature: bool
    symmetric: bool = False

    @property
    def diagonal(self) -> bool:
        return self.name in DIAGONAL_GATES


IDENTITY_SPEC = GateSpec(IDENTITY_NAME, arity=0, uses_feature=False)

ENTANGLER_SPECS: dict[str, GateSpec] = {
    name: GateSpec(name, arity=2, uses_feature=True, symmetric=True)
    for name in ("rzz", "rzzh", "rzzhn")
}


def resolve_order_seed(order_seed: OrderSeed) -> Optional[int]:
    """Turn ``order_seed`` into the seed of the catalogue shuffle.

    ``None`` keeps the declaration order, ``"random"`` draws a fresh seed.
    """
    if order_seed is None:
        return None
    if order_seed == RANDOM_ORDER:
        return random.SystemRandom().getrandbits(32)
    if isinstance(order_seed, bool) or not isinstance(order_seed, int):
        raise ValueError(
            f"order_seed must be None, an int or {RANDOM_ORDER!r}, "
            f"got {order_seed!r}")
    return order_seed


class ProductGateSet:
    """Every placement of every :class:`GateSpec` on the active qubits.

    Feature ``i`` is mapped to qubit ``i``: gates that read the data are only
    placed on the first ``n_features`` qubits.

    Parameters
    ----------
    with_identity
        Add an identity gate, so circuits shorter than the depth are reachable.
    canonical
        Restrict appends to catalogue order (default: when all gates are
        diagonal). Each multiset of commuting gates is then visited once.
    order_seed
        Shuffle of the catalogue order; see :func:`resolve_order_seed`.
    """

    def __init__(self, specs: Sequence[GateSpec], n_qubits: int, n_features: int,
                 with_identity: bool = False,
                 canonical: Optional[bool] = None,
                 order_seed: OrderSeed = None) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        specs = tuple(specs)
        if with_identity and IDENTITY_SPEC not in specs:
            specs += (IDENTITY_SPEC,)
        uses_features = any(s.uses_feature for s in specs)
        if uses_features and not 1 <= n_features <= n_qubits:
            raise ValueError(
                f"feature→qubit mapping: n_features ({n_features}) must lie "
                f"between 1 and n_qubits ({n_qubits})")
        self.specs = specs
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.n_active = n_features if uses_features else n_qubits
        self.canonical = (all(s.diagonal for s in specs) if canonical is None
                          else canonical)
        self.order_seed = resolve_order_seed(order_seed)
        catalog = self._enumerate()
        if self.order_seed is not None:
            random.Random(self.order_seed).shuffle(catalog)
        self._catalog: tuple[GateOp, ...] = tuple(catalog)
        self._rank: dict[GateOp, int] = {op: i for i, op in enumerate(self._catalog)}
        if self.canonical:
            self._suffixes: tuple[tuple[GateOp, ...], ...] = tuple(
                self._catalog[i:] for i in range(len(self._catalog)))
        else:
            self._without: dict[GateOp, tuple[GateOp, ...]] = {
                op: tuple(o for o in self._catalog if o != op) for op in self._catalog}

    @property
    def catalog(self) -> tuple[GateOp, ...]:
        return self._catalog

    def _qubit_tuples(self, spec: GateSpec) -> Sequence[tuple[int, ...]]:
        qubits = range(self.n_active)
        picker = combinations if spec.symmetric else permutations
        return list(picker(qubits, spec.arity))

    def _enumerate(self) -> list[GateOp]:
        return [(spec.name, qubits, qubits if spec.uses_feature else None)
                for spec in self.specs
                for qubits in self._qubit_tuples(spec)]

    def placements(self, state: State) -> Sequence[GateOp]:
        """Gates that may follow ``state``.

        Canonical: the catalogue from the last gate onwards. Otherwise: every
        gate except an immediate repeat of the last one.
        """
        last = state[-1] if state else None
        if self.canonical:
            rank = self._rank.get(last) if last is not None else None
            return self._catalog if rank is None else self._suffixes[rank]
        if last is None or last == IDENTITY_OP:
            return self._catalog
        return self._without.get(last, self._catalog)

    def free_placements(self, state: State) -> Sequence[GateOp]:
        """Whole catalogue, ignoring the canonical order (used by path-following descents)."""
        return self._catalog

    def insert(self, state: State, op: GateOp) -> State:
        """Insert ``op`` at its catalogue position (canonical) or append it."""
        if not self.canonical:
            return state + (op,)
        rank = self._rank.get(op)
        if rank is None:
            return state + (op,)
        i = len(state)
        while i > 0 and self._rank.get(state[i - 1], -1) > rank:
            i -= 1
        return state[:i] + (op,) + state[i:]


class LayeredGateSet:
    """Diagonal layers separated by Hadamard layers, up to ``max_layers`` layers.

    Within a layer, placements follow the canonical order of ``base``;
    the Hadamard-layer pseudo-gate ``hl`` opens the next layer.
    """

    def __init__(self, base: ProductGateSet, max_layers: int) -> None:
        if max_layers < 1:
            raise ValueError("max_layers must be >= 1")
        if not base.canonical:
            raise ValueError("base gate set must be canonical (diagonal)")
        self.base = base
        self.max_layers = max_layers
        self.n_qubits = base.n_qubits
        self.n_features = base.n_features
        self.n_active = base.n_active
        self._catalog = base.catalog + ((HADAMARD_LAYER_OP,) if max_layers > 1 else ())

    @property
    def catalog(self) -> tuple[GateOp, ...]:
        return self._catalog

    def placements(self, state: State) -> Sequence[GateOp]:
        """Canonical successors within the current layer, plus ``hl`` if a layer is left."""
        if not state or state[-1] == HADAMARD_LAYER_OP:
            return self.base.catalog
        ops = tuple(self.base.placements(state))
        if state.count(HADAMARD_LAYER_OP) < self.max_layers - 1:
            ops += (HADAMARD_LAYER_OP,)
        return ops

    def insert(self, state: State, op: GateOp) -> State:
        return state + (op,)


def layer_count(state: State) -> int:
    """Number of diagonal layers in ``state``."""
    return 1 + sum(op == HADAMARD_LAYER_OP for op in state)


def phase_power_specs() -> list[GateSpec]:
    """Constant phase gates ``diag(1, e^{i k pi/4})`` and their controlled versions."""
    return [GateSpec(name, arity, uses_feature=False, symmetric=True)
            for name, (arity, _k) in PHASE_POWERS.items()]


def iqp_gate_set(n_qubits: int, n_features: int,
                 with_phase_powers: bool = False,
                 canonical: Optional[bool] = None,
                 order_seed: OrderSeed = None,
                 extra_entanglers: Sequence[str] = ()) -> ProductGateSet:
    """Gate set of the diagonal part of an IQP circuit: ``rz`` and ``rzz``.

    ``extra_entanglers`` adds Havlíček-style ``rzzh``/``rzzhn`` gates whose
    angle is ``(pi - x_i)(pi - x_j)``.
    """
    specs = [
        GateSpec("rz", 1, uses_feature=True),
        ENTANGLER_SPECS["rzz"],
    ]
    for name in extra_entanglers:
        if name not in ENTANGLER_SPECS:
            raise ValueError(f"unknown entangler: {name!r} "
                             f"(expected: {', '.join(ENTANGLER_SPECS)})")
        specs.append(ENTANGLER_SPECS[name])
    if with_phase_powers:
        specs += phase_power_specs()
    return ProductGateSet(specs, n_qubits, n_features, canonical=canonical,
                          order_seed=order_seed)


def pauli_gate_set(n_qubits: int, n_features: int,
                   with_phase_powers: bool = False,
                   canonical: Optional[bool] = None,
                   order_seed: OrderSeed = None) -> ProductGateSet:
    """Non-diagonal gate set ``rx``, ``ry``, ``rz``, ``rzz``, beyond the IQP family."""
    specs = [
        GateSpec("rx", 1, uses_feature=True),
        GateSpec("ry", 1, uses_feature=True),
        GateSpec("rz", 1, uses_feature=True),
        GateSpec("rzz", 2, uses_feature=True, symmetric=True),
    ]
    if with_phase_powers:
        specs += phase_power_specs()
    return ProductGateSet(specs, n_qubits, n_features, canonical=canonical,
                          order_seed=order_seed)


def phase_power_gate_set(n_qubits: int, n_features: int = 0,
                         order_seed: OrderSeed = None) -> ProductGateSet:
    """Gate set made only of constant phase gates."""
    return ProductGateSet(phase_power_specs(), n_qubits, n_features,
                          order_seed=order_seed)
