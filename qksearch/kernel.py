from __future__ import annotations

from collections import OrderedDict
from typing import ClassVar, Optional, Protocol, Sequence, Union

import numpy as np

from qksearch.backend import Backend
from qksearch.dataset import BaseDataset
from qksearch.dla import DlaCalculator
from qksearch.gates import GateOp, GateSet, State
from qksearch.metric import Metric


class StateRegularizer(Protocol):
    needs_gram: ClassVar[bool] = False

    def __call__(self, state: State) -> float:
        ...


class GramRegularizer(Protocol):
    needs_gram: ClassVar[bool] = True

    def __call__(self, gram: np.ndarray) -> float:
        ...

    def batch(self, grams: np.ndarray) -> np.ndarray:
        ...


Regularizer = Union[StateRegularizer, GramRegularizer]


class DlaPenalty:
    needs_gram = False

    def __init__(self, dla: DlaCalculator, n_qubits: int, weight: float = 1.0) -> None:
        if weight < 0:
            raise ValueError("weight must be >= 0")
        self.dla = dla
        self.weight = weight
        self.dim_max = 4**n_qubits - 1

    def dimension(self, state: State) -> int:
        return self.dla.dimension(state)

    def __call__(self, state: State) -> float:
        return -self.weight * self.dla.dimension(state) / self.dim_max


class GramVarianceBonus:
    needs_gram = True

    def __init__(self, weight: float = 0.5) -> None:
        if weight < 0:
            raise ValueError("weight must be >= 0")
        self.weight = weight
        self._m: int = -1
        self._iu: Optional[tuple[np.ndarray, np.ndarray]] = None

    def _indices(self, m: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if m < 2:
            return None
        if m != self._m:
            self._m = m
            self._iu = np.triu_indices(m, k=1)
        return self._iu

    def __call__(self, gram: np.ndarray) -> float:
        iu = self._indices(gram.shape[-1])
        if iu is None:
            return 0.0
        return 4.0 * self.weight * float(gram[iu].var())

    def batch(self, grams: np.ndarray) -> np.ndarray:
        iu = self._indices(grams.shape[-1])
        if iu is None:
            return np.zeros(grams.shape[0], dtype=float)
        return 4.0 * self.weight * grams[:, iu[0], iu[1]].var(axis=1)


class CircuitEvaluator:
    def __init__(self, backend: Backend, data: BaseDataset, metric: Metric,
                 prefix: State = (), suffix: State = (),
                 psi_cache_size: int = 4096,
                 gram_feature: Optional[GramRegularizer] = None) -> None:
        self.backend = backend
        self.data = data
        self.metric = metric
        self.prefix = prefix
        self.suffix = suffix
        self.psi_cache_size = psi_cache_size
        self.gram_feature = gram_feature
        self._incremental = hasattr(backend, "apply") and hasattr(backend, "states")
        self._batched = hasattr(backend, "gram_batch") and hasattr(metric, "batch")
        self._scores: dict[State, float] = {}
        self._features: dict[State, float] = {}
        self._psi: OrderedDict[State, np.ndarray] = OrderedDict()
        self._root: Optional[np.ndarray] = None
        self._suffix_matrix: Optional[np.ndarray] = None
        self._op_phase: dict[GateOp, Optional[np.ndarray]] = {}


    def _apply(self, psi: np.ndarray, op: GateOp) -> np.ndarray:
        if op not in self._op_phase:
            self._op_phase[op] = (self.backend.diagonal_phase(op, self.data.X)
                                  if hasattr(self.backend, "diagonal_phase") else None)
        phase = self._op_phase[op]
        if phase is not None:
            return psi * phase
        return self.backend.apply(psi, op, self.data.X)

    def _prefix_states(self) -> np.ndarray:
        if self._root is None:
            self._root = self.backend.states(self.prefix, self.data.X)
        return self._root

    def _compiled_suffix(self) -> Optional[np.ndarray]:
        if self._suffix_matrix is not None:
            return self._suffix_matrix
        if not self._incremental or any(op[2] is not None for op in self.suffix):
            return None
        dim = self.backend.dim
        basis = np.eye(dim, dtype=complex)
        x = np.zeros((dim, self.data.X.shape[1]))
        for op in self.suffix:
            basis = self.backend.apply(basis, op, x)
        self._suffix_matrix = basis
        return self._suffix_matrix

    def states(self, state: State) -> np.ndarray:
        cached = self._psi.get(state)
        if cached is not None:
            self._psi.move_to_end(state)
            return cached
        if not state:
            return self._prefix_states()

        parent = self._psi.get(state[:-1])
        if parent is not None:
            self._psi.move_to_end(state[:-1])
            psi = self._apply(parent, state[-1])
        elif len(state) == 1:
            psi = self._apply(self._prefix_states(), state[0])
        else:
            psi = self._prefix_states()
            for op in state:
                psi = self._apply(psi, op)

        self._psi[state] = psi
        if len(self._psi) > self.psi_cache_size:
            self._psi.popitem(last=False)
        return psi

    def final_states(self, state: State) -> np.ndarray:
        psi = self.states(state)
        if not self.suffix:
            return psi
        matrix = self._compiled_suffix()
        if matrix is not None:
            return psi @ matrix
        for op in self.suffix:
            psi = self.backend.apply(psi, op, self.data.X)
        return psi


    def circuit(self, state: State) -> State:
        return self.prefix + state + self.suffix

    def gram(self, state: State) -> np.ndarray:
        if self._incremental:
            return self.backend.gram_from_states(self.final_states(state))
        return self.backend.gram(self.circuit(state), self.data.X)

    def kta(self, state: State) -> float:
        cached = self._scores.get(state)
        if cached is not None:
            return cached
        gram = self.gram(state)
        if self.gram_feature is not None:
            self._features[state] = float(self.gram_feature(gram))
        value = float(self.metric(gram, self.data.y))
        self._scores[state] = value
        return value

    def feature(self, state: State) -> float:
        cached = self._features.get(state)
        if cached is not None:
            return cached
        if self.gram_feature is None:
            return 0.0
        value = self._features[state] = float(self.gram_feature(self.gram(state)))
        return value

    def kta_many(self, states: Sequence[State]) -> np.ndarray:
        out = np.empty(len(states), dtype=float)
        pending: list[int] = []
        for i, s in enumerate(states):
            cached = self._scores.get(s)
            if cached is not None:
                out[i] = cached
            else:
                pending.append(i)
        if not pending:
            return out
        if not (self._incremental and self._batched):
            for i in pending:
                out[i] = self.kta(states[i])
            return out

        psis = np.stack([self.final_states(states[i]) for i in pending])
        grams = self.backend.gram_batch(psis)
        values = self.metric.batch(grams, self.data.y)
        feats = None if self.gram_feature is None else self.gram_feature.batch(grams)
        for k, i in enumerate(pending):
            out[i] = self._scores[states[i]] = float(values[k])
            if feats is not None:
                self._features[states[i]] = float(feats[k])
        return out


class QuantumKernelHeuristic:
    prefix: State = ()
    suffix: State = ()

    def __init__(
        self,
        gate_set: GateSet,
        backend: Backend,
        data: BaseDataset,
        metric: Metric,
        max_depth: int,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        gs_features = getattr(gate_set, "n_features", None)
        if gs_features is not None and gs_features != data.n_features:
            raise ValueError(
                f"gate_set encodes {gs_features} features, data has {data.n_features}")
        gs_qubits = getattr(gate_set, "n_qubits", None)
        be_qubits = getattr(backend, "n_qubits", None)
        if gs_qubits is not None and be_qubits is not None and gs_qubits != be_qubits:
            raise ValueError(
                f"gate_set on {gs_qubits} qubits, backend on {be_qubits}")
        self.gate_set = gate_set
        self.backend = backend
        self.data = data
        self.metric = metric
        self.max_depth = max_depth
        self.regularizer = regularizer
        self._evaluator: Optional[CircuitEvaluator] = None
        self._regularizations: dict[State, float] = {}

    @property
    def evaluator(self) -> CircuitEvaluator:
        if self._evaluator is None:
            reg = self.regularizer
            self._evaluator = CircuitEvaluator(
                self.backend, self.data, self.metric, self.prefix, self.suffix,
                gram_feature=reg if reg is not None and reg.needs_gram else None)
        return self._evaluator

    def candidates(self, state: State) -> Sequence[GateOp]:
        if len(state) >= self.max_depth:
            return []
        return self.gate_set.placements(state)

    def free_candidates(self, state: State) -> Sequence[GateOp]:
        if len(state) >= self.max_depth:
            return []
        free = getattr(self.gate_set, "free_placements", None)
        return free(state) if free is not None else self.gate_set.placements(state)

    def extend(self, state: State, op: GateOp) -> State:
        insert = getattr(self.gate_set, "insert", None)
        return insert(state, op) if insert is not None else state + (op,)

    def circuit(self, state: State) -> State:
        return self.prefix + state + self.suffix

    def kta(self, state: State) -> float:
        return self.evaluator.kta(state)

    def regularization(self, state: State) -> float:
        reg = self.regularizer
        if reg is None:
            return 0.0
        if reg.needs_gram:
            return self.evaluator.feature(state)
        cached = self._regularizations.get(state)
        if cached is None:
            cached = self._regularizations[state] = float(reg(self.circuit(state)))
        return cached

    def score(self, state: State) -> float:
        return self.kta(state) + self.regularization(state)

    def scores(self, states: Sequence[State]) -> np.ndarray:
        values = self.evaluator.kta_many(states)
        if self.regularizer is None:
            return values
        return values + np.array([self.regularization(s) for s in states])

    def is_complete(self, state: State) -> bool:
        return not self.candidates(state)


class KtaDlaHeuristic(QuantumKernelHeuristic):
    def __init__(
        self,
        gate_set: GateSet,
        backend: Backend,
        data: BaseDataset,
        metric: Metric,
        max_depth: int,
        dla: DlaCalculator,
        dla_weight: float = 0.2,
    ) -> None:
        n_qubits = (getattr(gate_set, "n_active", None)
                    or getattr(backend, "n_qubits", None)
                    or getattr(gate_set, "n_qubits", None))
        if n_qubits is None:
            raise ValueError("cannot infer n_qubits from the backend or the gate_set")
        super().__init__(gate_set, backend, data, metric, max_depth,
                         regularizer=DlaPenalty(dla, n_qubits, dla_weight))
