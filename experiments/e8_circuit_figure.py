from __future__ import annotations

import sys
from functools import reduce
from operator import mul

from experiments import common

SEED = 0
N_QUBITS = 4
N_DATA_GATES = 6
MCTS_ITERATIONS = 1000


def run_and_plot() -> None:
    import matplotlib.pyplot as plt
    from qiskit.circuit import Parameter, QuantumCircuit

    from qksearch.backend import NumpyBackend, QiskitBackend
    from qksearch.enumerate_circuits import strip_identity
    from qksearch.iqp import build_heuristic, describe
    from qksearch.search import MctsSearch, search

    common.report_style()
    train, _ = common.mof_split(SEED)
    heuristic = build_heuristic(train, n_qubits=N_QUBITS,
                                n_data_gates=N_DATA_GATES,
                                regularization="none", weight=0.0,
                                backend=NumpyBackend(N_QUBITS))
    res = search((), heuristic,
                 MctsSearch(n_iterations=MCTS_ITERATIONS, seed=SEED))
    layer = strip_identity(res.best_state)
    kta = float(heuristic.kta(res.best_state))

    df = common.read_csv("e5_benchmark.csv")
    row = df[(df["dataset"] == "mof") & (df["method"] == "quantum-kta")
             & (df["seed"] == SEED)].iloc[0]
    assert row["detail"] == describe(layer), (row["detail"], describe(layer))
    assert abs(row["kta_train"] - kta) < 1e-9

    drawer = QiskitBackend(n_qubits=N_QUBITS)
    params = [Parameter(f"x{f}") for f in range(N_QUBITS)]
    qc = QuantumCircuit(N_QUBITS)
    full = (tuple(("h", (q,), None) for q in range(N_QUBITS)) + layer
            + tuple(("h", (q,), None) for q in range(N_QUBITS)))
    for name, qubits, f in full:
        angle = None if f is None else reduce(mul, (params[i] for i in f))
        drawer._apply(qc, name, qubits, angle)

    style = {
        "backgroundcolor": "#ffffff",
        "linecolor": common.INK2,
        "textcolor": common.INK,
        "gatetextcolor": "#ffffff",
        "displaycolor": {
            "h": [common.MUTED, "#ffffff"],
            "rz": [common.BLUE, "#ffffff"],
            "rzz": [common.BLUE, "#ffffff"],
            "rzzh": [common.BLUE, "#ffffff"],
            "rzzhn": [common.BLUE, "#ffffff"],
        },
    }
    fig, ax = plt.subplots(figsize=(6.4, 2.3), constrained_layout=True)
    qc.draw(output="mpl", style=style, ax=ax, fold=-1)
    common.save_fig(fig, "fig_e8_circuit_mof")
    plt.close(fig)
    print(f"circuit : {describe(layer)}  KTA={kta:+.4f}  R²={row['r2_test']:+.3f}")


if __name__ == "__main__":
    _ = sys.argv
    run_and_plot()
