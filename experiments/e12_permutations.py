from __future__ import annotations

import sys
import time
from itertools import permutations

import numpy as np

from experiments import common

CSV = "e12_permutations.csv"
FIG = "fig_e12_permutations"
SEED = 0
N_QUBITS = 4

IQP_CIRCUITS: dict[str, tuple] = {
    "C1": (("rz", (0,), (0,)),
           ("rzz", (1, 2), (1, 2)),
           ("rz", (3,), (3,))),
    "C2": (("rz", (1,), (1,)),
           ("rzz", (0, 2), (0, 2)),
           ("rzzh", (1, 3), (1, 3))),
    "C3": (("rzz", (0, 1), (0, 1)),
           ("rzzh", (2, 3), (2, 3)),
           ("rz", (0,), (0,))),
    "C4": (("rz", (0,), (0,)),
           ("rzz", (0, 1), (0, 1)),
           ("rzzh", (2, 3), (2, 3)),
           ("rz", (2,), (2,))),
    "C5": (("rz", (0,), (0,)),
           ("rz", (2,), (2,)),
           ("rzz", (0, 1), (0, 1)),
           ("rzzh", (1, 3), (1, 3)),
           ("rzz", (2, 3), (2, 3))),
}

CONTROL_CIRCUITS: dict[str, tuple] = {
    "K1": (("rx", (1,), (1,)),
           ("rzz", (1, 2), (1, 2)),
           ("rz", (3,), (3,))),
    "K2": (("rx", (1,), (1,)),
           ("rzz", (0, 2), (0, 2)),
           ("rzzh", (1, 3), (1, 3))),
    "K3": (("rzz", (0, 1), (0, 1)),
           ("rzzh", (2, 3), (2, 3)),
           ("rx", (0,), (0,))),
    "K4": (("rx", (0,), (0,)),
           ("rzz", (0, 1), (0, 1)),
           ("rzzh", (2, 3), (2, 3)),
           ("rz", (2,), (2,))),
    "K5": (("rx", (0,), (0,)),
           ("rz", (2,), (2,)),
           ("rzz", (0, 1), (0, 1)),
           ("rzzh", (1, 3), (1, 3)),
           ("rzz", (2, 3), (2, 3))),
}

TABLE_CIRCUITS = ("C1", "C2", "C3", "K1", "K2", "K3")

EPS = float(np.finfo(float).eps)


def _heuristic(train, with_rx: bool):
    from qksearch.backend import NumpyBackend
    from qksearch.gates import ENTANGLER_SPECS, GateSpec, ProductGateSet
    from qksearch.iqp import IqpKernelHeuristic
    from qksearch.metric import TargetAlignment

    specs = [GateSpec("rz", 1, uses_feature=True),
             ENTANGLER_SPECS["rzz"], ENTANGLER_SPECS["rzzh"]]
    if with_rx:
        specs.append(GateSpec("rx", 1, uses_feature=True))
    gs = ProductGateSet(specs, n_qubits=N_QUBITS, n_features=N_QUBITS,
                        canonical=False)
    depth = max(len(c) for c in IQP_CIRCUITS.values())
    return IqpKernelHeuristic(
        gate_set=gs, backend=NumpyBackend(N_QUBITS), data=train,
        metric=TargetAlignment(), max_depth=depth, n_qubits=N_QUBITS,
        regularizer=None)


def _distinct_permutations(circuit: tuple) -> list[tuple]:
    seen = {circuit: None}
    for p in permutations(circuit):
        seen.setdefault(p, None)
    return list(seen)


def run() -> None:
    from qksearch.iqp import describe

    t0 = time.time()
    train, _ = common.mof_split(SEED)
    rows: list[dict] = []

    for family, circuits in (("iqp", IQP_CIRCUITS), ("control", CONTROL_CIRCUITS)):
        heuristic = _heuristic(train, with_rx=(family == "control"))
        for name, circuit in circuits.items():
            perms = _distinct_permutations(circuit)
            ref_kta = float(heuristic.kta(circuit))
            ref_gram = heuristic.evaluator.gram(circuit)
            for i, perm in enumerate(perms):
                gram = heuristic.evaluator.gram(perm)
                kta = float(heuristic.kta(perm))
                rows.append({
                    "family": family,
                    "circuit": name,
                    "n_gates": len(circuit),
                    "perm_index": i,
                    "order": "".join(str(circuit.index(op)) for op in perm),
                    "kta": kta,
                    "delta_kta": kta - ref_kta,
                    "gram_max_diff": float(np.abs(gram - ref_gram).max()),
                    "detail": describe(perm),
                })
            block = rows[-len(perms):]
            spread = max(r["kta"] for r in block) - min(r["kta"] for r in block)
            print(f"{family:8s} {name}  {len(perms):3d} permutations  "
                  f"KTA={ref_kta:+.6f}  spread={spread:.3e}")

    common.write_csv(CSV, rows)
    common.write_meta("e12", time.time() - t0, seed=SEED, n_qubits=N_QUBITS,
                      n_iqp_circuits=len(IQP_CIRCUITS),
                      n_control_circuits=len(CONTROL_CIRCUITS),
                      n_evaluations=len(rows), eps=EPS)


def plot() -> None:

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    common.report_style()
    df = common.read_csv(CSV, float_precision="round_trip", dtype={"order": str})

    families = (
        ("iqp", common.BLUE, "IQP — fully diagonal layer",
         "the gates commute: one value per multiset"),
        ("control", common.RED, "Control — one rx inside the layer",
         "the gates no longer commute: order changes the value"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.6), constrained_layout=True)

    for ax, (family, color, title, subtitle) in zip(axes, families):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        sub = df[df["family"] == family]
        shown = [c for c in TABLE_CIRCUITS if c in set(sub["circuit"])]

        n_rows = sum(len(sub[sub["circuit"] == c]) + 2 for c in shown) + 3
        h = 1.0 / n_rows
        x_perm, x_circ, x_val = 0.02, 0.11, 0.80

        y = 1.0 - 0.5 * h
        ax.text(0.0, y + 1.3 * h, title, fontsize=11, color=color,
                weight="bold", va="center")
        ax.text(0.0, y + 0.3 * h, subtitle, fontsize=8, color=common.INK2,
                va="center")
        y -= 1.6 * h
        for label, x, align in (("order", x_perm, "left"),
                                ("circuit (diagonal layer, Hadamards omitted)",
                                 x_circ, "left"),
                                ("KTA", x_val, "left")):
            ax.text(x, y, label, fontsize=8, color=common.MUTED, va="center",
                    ha=align)
        y -= 0.4 * h
        ax.plot([0.0, 1.0], [y, y], color=common.BASELINE, lw=0.8)
        y -= 0.9 * h

        for name in shown:
            g = sub[sub["circuit"] == name].sort_values("perm_index")
            ref = float(g["kta"].iloc[0])
            ax.text(x_perm, y, name, fontsize=8.5, color=color, weight="bold",
                    va="center")
            ax.text(x_circ, y,
                    f"{len(g)} permutations of the same multiset of "
                    f"{int(g['n_gates'].iloc[0])} gates",
                    fontsize=7.5, color=common.INK2, va="center")
            y -= h
            for _, r in g.iterrows():
                same = abs(float(r["kta"]) - ref) <= 8 * EPS
                ax.add_patch(Rectangle((0.0, y - 0.5 * h), 1.0, h,
                                       color=common.GRID, alpha=0.45, lw=0,
                                       zorder=0))
                ax.text(x_perm, y, str(r["order"]), fontsize=7.5,
                        family="monospace", color=common.MUTED, va="center")
                ax.text(x_circ, y, str(r["detail"]), fontsize=7.5,
                        family="monospace", color=common.INK, va="center")
                ax.text(x_val, y, f"{float(r['kta']):.10f}", fontsize=7.5,
                        family="monospace", va="center",
                        color=common.INK if same else common.RED,
                        weight="normal" if same else "bold")
                y -= h
            y -= h

    fig.suptitle("E12 — one circuit per row, its objective alongside: "
                 "permutations of an IQP layer give the same value, "
                 "the non-diagonal control does not", fontsize=12)
    common.save_fig(fig, FIG, formats=("png",))
    plt.close(fig)

    for family, _, _, _ in families:
        sub = df[df["family"] == family]
        worst = max(g["kta"].max() - g["kta"].min()
                    for _, g in sub.groupby("circuit"))
        print(f"{family:8s}: max spread over {len(sub)} evaluations = "
              f"{worst:.3e} ({'≤' if worst <= 8 * EPS else '>'} 8ε = "
              f"{8 * EPS:.1e})")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot()
