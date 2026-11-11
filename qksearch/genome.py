"""Fixed-length integer encoding of layered IQP circuits, for the genetic algorithm.

A genome stores one multiplicity per (layer, gate) gene: how many times an
``rz``/``rzz`` data gate is repeated, or which power of a constant phase gate
(``t``, ``cs``...) is applied. Diagonal gates commute within a layer, so this
loses nothing; :func:`decode` puts the Hadamard layers back between layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Iterator, Optional, Sequence

import numpy as np

from qksearch.gates import (
    IDENTITY_OP,
    PHASE_POWERS,
    GateOp,
    GateSpec,
    ProductGateSet,
    State,
)

Genome = tuple[int, ...]

PHASE_BY_POWER: dict[int, dict[int, str]] = {}
for _name, (_arity, _k) in PHASE_POWERS.items():
    PHASE_BY_POWER.setdefault(_arity, {})[_k] = _name

DEFAULT_DATA_CAP = 7


def phase_order(name: str) -> int:
    """Order of a phase gate in the group it generates (``t``: 8, ``s``: 4, ``z``: 2)."""
    _arity, k = PHASE_POWERS[name]
    return 8 // gcd(8, k)


@dataclass(frozen=True)
class GeneSpec:
    """One gene: a gate, its maximum multiplicity ``cap`` (``2**w - 1``) and its layer."""

    op: GateOp
    cap: int
    layer: int

    def __post_init__(self) -> None:
        if self.cap < 1:
            raise ValueError(f"cap must be >= 1, got {self.cap}")
        if (self.cap + 1) & self.cap:
            raise ValueError(
                f"cap + 1 must be a power of two so that every bit pattern is a "
                f"valid multiplicity, got cap={self.cap}")

    @property
    def width(self) -> int:
        """Number of bits used by the gene."""
        return (self.cap + 1).bit_length() - 1

    @property
    def is_phase(self) -> bool:
        return self.op[0] in PHASE_POWERS


@dataclass(frozen=True)
class GenomeSpec:
    """Ordered list of genes, grouped by layer."""

    genes: tuple[GeneSpec, ...]
    n_qubits: int
    n_layers: int

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")
        if not self.genes:
            raise ValueError("a genome spec needs at least one gene")
        if any(g.layer >= self.n_layers for g in self.genes):
            raise ValueError("a gene sits in a layer beyond n_layers")
        layers = [g.layer for g in self.genes]
        if layers != sorted(layers):
            raise ValueError("genes must be grouped by increasing layer")

    def __len__(self) -> int:
        return len(self.genes)

    @property
    def n_bits(self) -> int:
        """Total number of bits, i.e. ``log2`` of the size of the search space."""
        return sum(g.width for g in self.genes)

    @property
    def caps(self) -> tuple[int, ...]:
        return tuple(g.cap for g in self.genes)

    def layer_slice(self, layer: int) -> slice:
        """Slice of the genes that belong to ``layer``."""
        lo = next(i for i, g in enumerate(self.genes) if g.layer == layer)
        hi = lo + sum(1 for g in self.genes if g.layer == layer)
        return slice(lo, hi)

    def zero(self) -> Genome:
        return (0,) * len(self.genes)

    def describe(self) -> str:
        """Summary of the genes and bits per layer."""
        parts = []
        for layer in range(self.n_layers):
            sl = self.layer_slice(layer)
            sub = self.genes[sl]
            bits = sum(g.width for g in sub)
            parts.append(f"layer {layer}: {len(sub)} genes, {bits} bits")
        return f"{len(self.genes)} genes, {self.n_bits} bits — " + "; ".join(parts)


def layer_gate_set(n_qubits: int, n_features: int,
                   phases: Sequence[str] = ()) -> ProductGateSet:
    """Diagonal gates of one layer: ``rz``, ``rzz`` and the given phase gates."""
    specs = [GateSpec("rz", 1, uses_feature=True),
             GateSpec("rzz", 2, uses_feature=True, symmetric=True)]
    for name in phases:
        if name not in PHASE_POWERS:
            raise ValueError(f"unknown phase gate: {name!r} "
                             f"(expected one of {', '.join(sorted(PHASE_POWERS))})")
        arity, _k = PHASE_POWERS[name]
        specs.append(GateSpec(name, arity, uses_feature=False, symmetric=True))
    return ProductGateSet(specs, n_qubits=n_qubits, n_features=n_features,
                          with_identity=True)


def iqp_genome_spec(n_qubits: int, n_features: int, *, n_layers: int = 2,
                    phases: Sequence[str] = ("t", "cs"),
                    data_cap: int = DEFAULT_DATA_CAP) -> GenomeSpec:
    """Genome spec of an ``n_layers``-layer IQP circuit.

    Constant phase gates are left out of the last layer, where they act on
    the measured basis and cannot change the kernel.
    """
    if phases and n_layers > 1 and n_features != n_qubits:
        raise ValueError(
            f"phase gates need n_features == n_qubits (got {n_features} and "
            f"{n_qubits}): ProductGateSet would confine them to the feature "
            f"qubits. Pass phases=() or widen the feature map.")
    genes: list[GeneSpec] = []
    for layer in range(n_layers):
        last = layer == n_layers - 1
        gs = layer_gate_set(n_qubits, n_features, () if last else phases)
        for op in gs.catalog:
            if op == IDENTITY_OP:
                continue
            name = op[0]
            cap = phase_order(name) - 1 if name in PHASE_POWERS else data_cap
            genes.append(GeneSpec(op, cap, layer))
    return GenomeSpec(tuple(genes), n_qubits=n_qubits, n_layers=n_layers)


def _hadamard_layer(n_qubits: int) -> State:
    return tuple(("h", (q,), None) for q in range(n_qubits))


def expand(gene: GeneSpec, k: int) -> State:
    """Gates encoded by multiplicity ``k`` of ``gene``."""
    if k == 0:
        return ()
    if k > gene.cap:
        raise ValueError(f"multiplicity {k} exceeds cap {gene.cap} for {gene.op}")
    name, qubits, f = gene.op
    if not gene.is_phase:
        return (gene.op,) * k
    _arity, base = PHASE_POWERS[name]
    power = (base * k) % 8
    if power == 0:
        return ()
    return ((PHASE_BY_POWER[len(qubits)][power], qubits, f),)


def decode(g: Genome, spec: GenomeSpec) -> State:
    """Genome to circuit, Hadamard layers included between layers."""
    if len(g) != len(spec.genes):
        raise ValueError(f"genome of length {len(g)}, spec expects {len(spec.genes)}")
    h = _hadamard_layer(spec.n_qubits)
    out: list[GateOp] = []
    for layer in range(spec.n_layers):
        if layer:
            out.extend(h)
        sl = spec.layer_slice(layer)
        for gene, k in zip(spec.genes[sl], g[sl]):
            out.extend(expand(gene, k))
    return tuple(out)


def encode(state: State, spec: GenomeSpec) -> Genome:
    """Inverse of :func:`decode`; raises ``ValueError`` if ``state`` does not fit ``spec``."""
    index = {(gene.layer, gene.op): i for i, gene in enumerate(spec.genes)}
    phases = {(gene.layer, gene.op[1]): i
              for i, gene in enumerate(spec.genes) if gene.is_phase}
    counts = [0] * len(spec.genes)
    layer = 0
    i = 0
    ops = list(state)
    h = _hadamard_layer(spec.n_qubits)
    while i < len(ops):
        if tuple(ops[i:i + len(h)]) == h:
            layer += 1
            if layer >= spec.n_layers:
                raise ValueError("more Hadamard layers than the spec allows")
            i += len(h)
            continue
        name, qubits, f = ops[i]
        i += 1
        if name == IDENTITY_OP[0]:
            continue
        if name in PHASE_POWERS:
            _arity, power = PHASE_POWERS[name]
            j = phases.get((layer, qubits))
            if j is None:
                raise ValueError(f"phase gate {name}{qubits} is not a gene of "
                                 f"layer {layer}")
            _a, base = PHASE_POWERS[spec.genes[j].op[0]]
            k = next((m for m in range(1, phase_order(spec.genes[j].op[0]))
                      if (base * m) % 8 == power), None)
            if k is None:
                raise ValueError(f"{name} is not a power of {spec.genes[j].op[0]}")
            counts[j] = k
            continue
        key = (layer, (name, qubits, f))
        if key not in index:
            raise ValueError(f"op {key[1]} is not a gene of layer {layer}")
        j = index[key]
        counts[j] += 1
        if counts[j] > spec.genes[j].cap:
            raise ValueError(f"op {key} repeated more than its cap {spec.genes[j].cap}")
    return tuple(counts)


def to_int(g: Genome, spec: GenomeSpec) -> int:
    """Pack a genome into an integer of ``spec.n_bits`` bits."""
    value = 0
    shift = 0
    for gene, k in zip(spec.genes, g):
        if not 0 <= k <= gene.cap:
            raise ValueError(f"multiplicity {k} out of [0, {gene.cap}] for {gene.op}")
        value |= k << shift
        shift += gene.width
    return value


def from_int(value: int, spec: GenomeSpec) -> Genome:
    """Inverse of :func:`to_int`."""
    if not 0 <= value < (1 << spec.n_bits):
        raise ValueError(f"value out of [0, 2**{spec.n_bits})")
    out: list[int] = []
    for gene in spec.genes:
        out.append(value & gene.cap)
        value >>= gene.width
    return tuple(out)


def weight(g: Genome, spec: GenomeSpec) -> int:
    """Number of gates: data multiplicities, plus one per active phase gene."""
    return sum(1 if gene.is_phase and k else k for gene, k in zip(spec.genes, g))


def support(g: Genome) -> int:
    """Number of non-zero genes."""
    return sum(1 for k in g if k)


def l1_distance(a: Genome, b: Genome) -> int:
    """Sum of absolute differences of multiplicities."""
    return sum(abs(x - y) for x, y in zip(a, b))


def n_phase_gates(g: Genome, spec: GenomeSpec) -> int:
    """Number of active phase genes."""
    return sum(1 for gene, k in zip(spec.genes, g) if k and gene.is_phase)


def repair(g: Genome, spec: GenomeSpec, rng: np.random.Generator, *,
           max_support: Optional[int] = None,
           max_weight: Optional[int] = None,
           protect: Sequence[int] = ()) -> Genome:
    """Randomly zero or decrement genes until the support and weight limits hold.

    Genes in ``protect`` are touched last.
    """
    out = list(g)
    kept = frozenset(protect)

    def eligible() -> list[int]:
        free = [i for i, k in enumerate(out) if k and (i not in kept or k > 1)]
        return free or [i for i, k in enumerate(out) if k]

    if max_support is not None:
        nz = [i for i, k in enumerate(out) if k]
        excess = len(nz) - max_support
        if excess > 0:
            free = [i for i in nz if i not in kept] or nz
            drop = rng.choice(free, size=min(excess, len(free)), replace=False)
            for i in drop:
                out[int(i)] = 0
    if max_weight is not None:
        while weight(tuple(out), spec) > max_weight:
            i = int(rng.choice(eligible()))
            out[i] = 0 if spec.genes[i].is_phase else out[i] - 1
    return tuple(out)


def random_genome(spec: GenomeSpec, rng: np.random.Generator, *,
                  max_support: int = 6, max_weight: Optional[int] = None,
                  require_last_layer: bool = True) -> Genome:
    """Random genome with between 1 and ``max_support`` active genes.

    With ``require_last_layer``, at least one gene of the last layer is
    active, so the circuit does not collapse to fewer layers.
    """
    if not 1 <= max_support <= len(spec.genes):
        raise ValueError(f"max_support must lie in [1, {len(spec.genes)}]")
    size = int(rng.integers(1, max_support + 1))
    out = [0] * len(spec.genes)

    picks = list(rng.choice(len(spec.genes), size=size, replace=False))
    if require_last_layer and spec.n_layers > 1:
        last = spec.layer_slice(spec.n_layers - 1)
        if not any(last.start <= int(i) < last.stop for i in picks):
            picks[-1] = int(rng.integers(last.start, last.stop))
    for i in picks:
        i = int(i)
        out[i] = int(rng.integers(1, spec.genes[i].cap + 1))

    protect = ()
    if require_last_layer and spec.n_layers > 1:
        last = spec.layer_slice(spec.n_layers - 1)
        protect = tuple(i for i in range(last.start, last.stop) if out[i])
    return repair(tuple(out), spec, rng, max_weight=max_weight, protect=protect)


def enumerate_genomes(spec: GenomeSpec, *, max_weight: int) -> Iterator[Genome]:
    """Yield every genome whose :func:`weight` is at most ``max_weight``."""
    caps = spec.caps
    out = [0] * len(caps)

    def rec(i: int, left: int) -> Iterator[Genome]:
        if i == len(caps):
            yield tuple(out)
            return
        span = (min(caps[i], 1) if spec.genes[i].is_phase else min(caps[i], left))
        for k in range(span + 1):
            out[i] = k
            cost = 1 if (spec.genes[i].is_phase and k) else k
            if spec.genes[i].is_phase and k:
                for power in range(1, caps[i] + 1):
                    out[i] = power
                    yield from rec(i + 1, left - cost)
                out[i] = 0
                break
            yield from rec(i + 1, left - cost)
        out[i] = 0

    return rec(0, max_weight)
