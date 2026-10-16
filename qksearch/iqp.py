from __future__ import annotations

from typing import Callable, Optional, Sequence

from qksearch.backend import QiskitBackend
from qksearch.dataset import BaseDataset, toy_dataset
from qksearch.dla import DlaCalculator
from qksearch.gates import (
    ANGLE_ENCODING,
    RANDOM_ORDER,
    GateOp,
    OrderSeed,
    State,
    iqp_gate_set,
)
from qksearch.kernel import (
    DlaPenalty,
    GramVarianceBonus,
    QuantumKernelHeuristic,
    Regularizer,
)
from qksearch.metric import Metric, TargetAlignment
from qksearch.search import (
    GreedySearch,
    SearchResult,
    SearchStrategy,
    search,
)


def hadamard_layer(n_qubits: int) -> State:
    return tuple(("h", (q,), None) for q in range(n_qubits))


def iqp_circuit(diagonal: State, n_qubits: int) -> State:
    h = hadamard_layer(n_qubits)
    return h + diagonal + h


def angle_label(name: str, f: tuple[int, ...]) -> str:
    if ANGLE_ENCODING.get(name, "").startswith("havlicek"):
        scaled = "/2π" if ANGLE_ENCODING[name].endswith("scaled") else ""
        return "".join(f"(π−x{i})" for i in f) + scaled
    return "".join(f"x{i}" for i in f)


def describe(state: State) -> str:
    parts = []
    for name, qubits, f in state:
        q = ",".join(map(str, qubits))
        angle = "" if f is None else "·" + angle_label(name, f)
        parts.append(f"{name}({q})" + angle)
    return " · ".join(parts)


def show(name: str, result: SearchResult[State], heuristic: IqpKernelHeuristic) -> None:
    best = result.best_state
    full = iqp_circuit(best, heuristic.n_qubits) if best is not None else None
    detail = ""
    reg = heuristic.regularizer
    if best is not None and reg is not None:
        extra = (f"  dim(DLA)={reg.dimension(full)}/{reg.dim_max}"
                 if hasattr(reg, "dimension") else "")
        detail = (f"  [KTA={heuristic.kta(best):+.4f}  "
                  f"reg.={heuristic.regularization(best):+.4f}{extra}]")
    print(f"{name:12s} score={result.best_score:+.4f}{detail}  "
          f"(expanded={result.n_expanded}, evaluated={result.n_evaluated})")
    print(f"{'':12s} circuit = {describe(full) if full else '—'}")


class IqpKernelHeuristic(QuantumKernelHeuristic):
    def __init__(
        self,
        gate_set,
        backend,
        data: BaseDataset,
        metric,
        max_depth: int,
        n_qubits: int,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        super().__init__(gate_set, backend, data, metric, max_depth,
                         regularizer=regularizer)
        self.n_qubits = n_qubits
        self.prefix = self.suffix = hadamard_layer(n_qubits)


REGULARIZERS: dict[str, Callable[[int, float], Optional[Regularizer]]] = {
    "dla": lambda n_active, w: DlaPenalty(DlaCalculator(), n_active, w),
    "variance": lambda n_active, w: GramVarianceBonus(w),
    "none": lambda n_active, w: None,
}

DEFAULT_WEIGHTS: dict[str, float] = {"dla": 1.0, "variance": 0.5, "none": 0.0}


def default_metric(data: BaseDataset) -> Metric:
    return TargetAlignment()


def build_heuristic(
    data: BaseDataset,
    n_qubits: int,
    n_data_gates: int,
    regularization: str = "variance",
    weight: float = 0.0,
    backend: Optional[object] = None,
    metric: Optional[Metric] = None,
    order_seed: OrderSeed = None,
    extra_entanglers: Sequence[str] = (),
) -> IqpKernelHeuristic:
    if regularization not in REGULARIZERS:
        raise ValueError(f"unknown regularization: {regularization!r} "
                         f"(expected: {', '.join(REGULARIZERS)})")
    gate_set = iqp_gate_set(n_qubits=n_qubits, n_features=data.n_features,
                            with_phase_powers=False, order_seed=order_seed,
                            extra_entanglers=extra_entanglers)
    backend = backend if backend is not None else QiskitBackend(n_qubits=n_qubits)
    metric = metric if metric is not None else default_metric(data)
    regularizer = (REGULARIZERS[regularization](gate_set.n_active, weight)
                   if weight > 0 else None)
    return IqpKernelHeuristic(
        gate_set=gate_set,
        backend=backend,
        data=data,
        metric=metric,
        max_depth=n_data_gates,
        n_qubits=n_qubits,
        regularizer=regularizer,
    )


def main(order_seed: OrderSeed = RANDOM_ORDER) -> None:
    n_qubits = 2
    n_data_gates = 6

    data = toy_dataset(n_samples=12, n_features=2, seed=0)
    initial: State = ()
    strategies: dict[str, SearchStrategy[State, GateOp]] = {
        "greedy": GreedySearch(),
    }
    for regularization in REGULARIZERS:
        weight = DEFAULT_WEIGHTS[regularization]
        heuristic = build_heuristic(data, n_qubits=n_qubits, n_data_gates=n_data_gates,
                                    regularization=regularization, weight=weight,
                                    order_seed=order_seed)
        print(f"regularization = {regularization} (weight {weight:g}, "
              f"canonical order seed {heuristic.gate_set.order_seed})")
        for name, strategy in strategies.items():
            result = search(initial, heuristic, strategy)
            show(name, result, heuristic)


if __name__ == "__main__":
    main()
