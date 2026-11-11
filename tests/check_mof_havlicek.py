from __future__ import annotations

import argparse
import os
import time
from itertools import groupby

import numpy as np

from experiments import common
from experiments.e5_benchmark import _r2, _rbf_gamma_cv
from experiments.e10_rzz import (
    ENTANGLERS,
    N_QUBITS,
    _heuristic,
)
from qksearch.compare_kernels import draw_circuit
from qksearch.enumerate_circuits import strip_identity
from qksearch.gates import State
from qksearch.iqp import angle_label, describe, iqp_circuit
from qksearch.search import MctsSearch, search

CATALOGS = ENTANGLERS
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "figures")


def draw(state: State, n_qubits: int = N_QUBITS) -> str:
    from qiskit.circuit import Parameter, QuantumCircuit

    from qksearch.backend import QiskitBackend

    drawer = QiskitBackend(n_qubits=n_qubits)
    qc = QuantumCircuit(n_qubits)
    params: dict[str, Parameter] = {}
    for op, run in groupby(iqp_circuit(state, n_qubits)):
        name, qubits, f = op
        k = sum(1 for _ in run)
        if f is None:
            for _ in range(k):
                drawer._apply(qc, name, qubits, None)
            continue
        label = angle_label(name, f)
        angle = params.setdefault(label, Parameter(label))
        drawer._apply(qc, name, qubits, k * angle)
    return str(qc.draw(output="text", fold=-1))


def select(seed: int, entanglers: tuple[str, ...], budget: int):
    train, test = common.mof_split(seed)
    heuristic, gate_set = _heuristic(train, entanglers)
    result = search((), heuristic, MctsSearch(n_iterations=budget, seed=seed))
    state = strip_identity(result.best_state)
    circuit = iqp_circuit(state, N_QUBITS)
    be = heuristic.backend
    r2 = _r2(be.gram(circuit, train.X),
             be.cross_gram(circuit, test.X, train.X), train, test)
    return state, float(heuristic.kta(state)), r2, len(gate_set.catalog), result


def rbf_baseline(seed: int) -> float:
    from sklearn.metrics.pairwise import rbf_kernel

    train, test = common.mof_split(seed)
    gamma = _rbf_gamma_cv(train)
    return _r2(rbf_kernel(train.X, gamma=gamma),
               rbf_kernel(test.X, train.X, gamma=gamma), train, test)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0,
                   help="split shown in detail (default: 0)")
    p.add_argument("--seeds", type=int, default=0,
                   help="splits in the summary table, plus the RBF reference "
                        "(default: 0, detailed circuit only)")
    p.add_argument("--budget", type=int, default=1000,
                   help="MCTS iterations, as in E5 (default: 1000)")
    p.add_argument("--no-fig", action="store_true",
                   help="do not write the circuit PNG")
    p.add_argument("--fig-dir", default=FIG_DIR,
                   help=f"PNG output directory (default: {FIG_DIR})")
    p.add_argument("--catalogs", default="all",
                   help="catalogues to compare, among "
                        f"{', '.join(CATALOGS)} (default: all)")
    args = p.parse_args()

    names = [n.strip() for n in args.catalogs.split(",") if n.strip()]
    unknown = [n for n in names if n not in CATALOGS]
    if unknown:
        p.error(f"unknown catalogue: {', '.join(unknown)} "
                f"(expected: {', '.join(CATALOGS)})")
    catalogs = {n: CATALOGS[n] for n in names}

    t0 = time.time()
    for name, entanglers in catalogs.items():
        state, kta, r2, size, res = select(args.seed, entanglers, args.budget)
        print(f"\n=== split {args.seed} — catalogue {name} "
              f"({size} gates: {', '.join(('rz',) + entanglers)})")
        print(f"  circuit  : {describe(state) or 'empty'}")
        print(f"  KTA train = {kta:+.4f}   R² test = {r2:+.4f}   "
              f"({res.n_evaluated} circuits evaluated)")
        print("\n" + draw(state))
        if not args.no_fig:
            os.makedirs(args.fig_dir, exist_ok=True)
            path = os.path.join(args.fig_dir,
                                f"fig_mof_{name}_seed{args.seed}.png")
            draw_circuit(iqp_circuit(state, N_QUBITS), N_QUBITS, N_QUBITS, path,
                         title=f"MOF split {args.seed} — catalogue {name}"
                               f"   (KTA = {kta:+.3f}, test R² = {r2:+.3f})")
            print(f"  figure → {path}")
    if args.seeds < 1:
        return 0

    print(f"\n  RBF reference (γ by 5-fold CV): test R² = "
          f"{rbf_baseline(args.seed):+.4f}")

    print(f"\n=== {args.seeds} splits, KTA / R² / number of entanglers")
    print(f"{'seed':>5} | " + " | ".join(n.center(26) for n in catalogs))
    table = {name: [] for name in catalogs}
    for seed in range(args.seeds):
        cells = []
        for name, entanglers in catalogs.items():
            state, kta, r2, _, _ = select(seed, entanglers, args.budget)
            n_ent = sum(1 for op in state if len(op[1]) == 2)
            table[name].append((kta, r2, n_ent))
            cells.append(f"{kta:+.4f} / {r2:+.4f} / {n_ent}")
        print(f"{seed:>5} | " + " | ".join(c.rjust(26) for c in cells))
    print(f"{'med.':>5} | " + " | ".join(
        f"{np.median([v[0] for v in table[n]]):+.4f} / "
        f"{np.median([v[1] for v in table[n]]):+.4f} / "
        f"{np.median([v[2] for v in table[n]]):.1f}".rjust(26)
        for n in catalogs))
    print()
    for name in catalogs:
        used = sum(1 for v in table[name] if v[2] > 0)
        print(f"  catalogue {name:5s}: an entangler on {used}/{args.seeds} splits")
    print(f"  ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
