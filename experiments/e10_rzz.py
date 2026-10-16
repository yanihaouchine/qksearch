from __future__ import annotations

import io
import sys
import time

import numpy as np

from experiments import common

CSV = "e10_rzz.csv"
SEEDS = list(range(10))
N_QUBITS = 4
N_DATA_GATES = 6
MCTS_ITERATIONS = 1000

ENTANGLERS = {"rzz": ("rzz",), "rzzh": ("rzzh",), "rzzhn": ("rzzhn",),
              "all": ("rzz", "rzzh", "rzzhn")}


def _heuristic(train, entanglers):
    from qksearch.backend import NumpyBackend
    from qksearch.gates import ENTANGLER_SPECS, GateSpec, ProductGateSet
    from qksearch.iqp import IqpKernelHeuristic
    from qksearch.metric import TargetAlignment

    specs = [GateSpec("rz", 1, uses_feature=True)]
    specs += [ENTANGLER_SPECS[name] for name in entanglers]
    gs = ProductGateSet(specs, n_qubits=N_QUBITS, n_features=N_QUBITS,
                        with_identity=True)
    heuristic = IqpKernelHeuristic(
        gate_set=gs, backend=NumpyBackend(N_QUBITS), data=train,
        metric=TargetAlignment(), max_depth=N_DATA_GATES, n_qubits=N_QUBITS,
        regularizer=None)
    return heuristic, gs


def _row(catalog, protocol, seed, state, heuristic, train, test, n_evaluated):
    from qksearch.enumerate_circuits import strip_identity
    from experiments.e5_benchmark import _r2
    from qksearch.iqp import describe, iqp_circuit

    circuit = iqp_circuit(state, N_QUBITS)
    be = heuristic.backend
    stripped = strip_identity(state)
    return {
        "catalog": catalog, "protocol": protocol, "seed": seed,
        "kta_train": float(heuristic.kta(state)),
        "r2_test": _r2(be.gram(circuit, train.X),
                       be.cross_gram(circuit, test.X, train.X), train, test),
        "n_gates": len(stripped),
        "n_entanglers": sum(1 for op in stripped if len(op[1]) == 2),
        "n_evaluated": n_evaluated,
        "detail": describe(stripped) or "empty",
    }


def run() -> None:
    from qksearch.search import MctsSearch, search

    t0 = time.time()
    rows: list[dict] = []
    sizes: dict[str, int] = {}

    for catalog, entanglers in ENTANGLERS.items():
        for seed in SEEDS:
            train, test = common.mof_split(seed)
            heuristic, gs = _heuristic(train, entanglers)
            sizes[catalog] = len(gs.catalog)

            if len(entanglers) == 1:
                states = [()]
                for _ in range(N_DATA_GATES):
                    states = [s + (op,) for s in states for op in gs.placements(s)]
                ktas = heuristic.evaluator.kta_many(states)
                best = states[int(np.argmax(ktas))]
                rows.append(_row(catalog, "exhaustive", seed, best, heuristic,
                                 train, test, len(states)))

            res = search((), heuristic,
                         MctsSearch(n_iterations=MCTS_ITERATIONS, seed=seed))
            rows.append(_row(catalog, "mcts", seed, res.best_state, heuristic,
                             train, test, res.n_evaluated))
        print(f"{catalog:9s} ({sizes[catalog]:2d} portes) fait "
              f"({time.time() - t0:.1f}s)")

    common.write_csv(CSV, rows)

    import pandas as pd

    df = pd.DataFrame(rows)
    medians: dict[str, dict] = {}
    print(f"\n{'catalogue':>10} {'protocole':>11} {'KTA méd.':>9} {'R² méd.':>8} "
          f"{'circuits à ≥1 entangleur':>26}")
    for (catalog, protocol), sub in df.groupby(["catalog", "protocol"]):
        n_ent = int((sub["n_entanglers"] > 0).sum())
        medians[f"{catalog}/{protocol}"] = {
            "kta": float(sub["kta_train"].median()),
            "r2": float(sub["r2_test"].median()),
            "with_entangler": n_ent,
        }
        print(f"{catalog:>10} {protocol:>11} {sub['kta_train'].median():>+9.4f} "
              f"{sub['r2_test'].median():>+8.4f} {n_ent:>20d}/{len(sub)}")

    common.write_meta("e10_rzz", time.time() - t0, seeds=SEEDS,
                      n_qubits=N_QUBITS, n_data_gates=N_DATA_GATES,
                      mcts_iterations=MCTS_ITERATIONS,
                      catalog_sizes=sizes, medians=medians)


CATALOGS = ("rzz", "rzzh", "rzzhn", "all")
LABELS = {"rzz": "rzz", "rzzh": "rzzh", "rzzhn": "rzzhn", "all": "union"}
ANGLES = (r"angle de l'entangleur — rzz : $x_i x_j$    "
          r"rzzh : $(\pi - x_i)(\pi - x_j)$    "
          r"rzzhn : $(\pi - x_i)(\pi - x_j) / 2\pi$    "
          r"union : les trois au même catalogue")
PROTOCOLS = (("exhaustive", common.BLUE, "optimum exact (8008 circuits)"),
             ("mcts", common.RED, f"MCTS ({MCTS_ITERATIONS} itérations)"))


def plot() -> None:
    import matplotlib.pyplot as plt

    common.report_style()
    df = common.read_csv(CSV)
    present = [c for c in CATALOGS if c in set(df["catalog"])]
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))

    for ax, column, title, ylabel, floor in (
            (axes[0], "kta_train", "(a) alignement optimisé",
             "KTA (train)", None),
            (axes[1], "r2_test", "(b) ce qu'il rapporte vraiment",
             "$R^2$ test", -1.0)):
        below = 0
        for offset, (protocol, colour, label) in zip((-0.17, 0.17), PROTOCOLS):
            xs, ys, xs_out = [], [], []
            for i, catalog in enumerate(present):
                sub = df[(df["catalog"] == catalog)
                         & (df["protocol"] == protocol)]
                if sub.empty:
                    continue
                vals = sub[column].to_numpy(dtype=float)
                jitter = i + offset + rng.uniform(-0.07, 0.07, len(vals))
                if floor is not None:
                    out = vals < floor
                    below += int(out.sum())
                    xs_out.extend(jitter[out])
                    jitter, vals = jitter[~out], vals[~out]
                xs.extend(jitter)
                ys.extend(vals)
                med = float(np.median(sub[column].to_numpy(dtype=float)))
                ax.plot([i + offset - 0.14, i + offset + 0.14], [med, med],
                        color=common.INK, lw=1.6, zorder=3,
                        solid_capstyle="butt")
            ax.scatter(xs, ys, s=10, color=colour, alpha=0.55, lw=0,
                       label=label)
            if xs_out:
                ax.scatter(xs_out, [floor] * len(xs_out), s=16, marker="v",
                           color=colour, lw=0, clip_on=False)
        if floor is not None:
            ax.set_ylim(bottom=floor)
            if below:
                ax.text(0.5, 0.02, f"▾ {below} point"
                        f"{'s' if below > 1 else ''} sous {floor:+.0f}",
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=6.5, color=common.MUTED)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([LABELS[c] for c in present], fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.5)
    axes[0].legend(loc="upper left", fontsize=6.5)

    ax = axes[2]
    width = 0.34
    for offset, (protocol, colour, label) in zip((-width / 2, width / 2),
                                                 PROTOCOLS):
        for i, catalog in enumerate(present):
            sub = df[(df["catalog"] == catalog) & (df["protocol"] == protocol)]
            if sub.empty:
                continue
            kept = int((sub["n_entanglers"] > 0).sum())
            ax.bar(i + offset, kept, width=width, color=colour)
            ax.text(i + offset, kept + 0.15, str(kept), ha="center",
                    va="bottom", fontsize=7, color=common.INK2)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([LABELS[c] for c in present], fontsize=7.5)
    ax.set_ylim(0, len(SEEDS) * 1.25)
    ax.set_ylabel(f"circuits sur {len(SEEDS)} gardant "
                  "$\\geq 1$ entangleur")
    ax.set_title("(c) l'entangleur est-il retenu ?")
    ax.grid(True, axis="y", alpha=0.5)

    fig.tight_layout()
    fig.text(0.5, -0.03, ANGLES, ha="center", va="top", fontsize=7,
             color=common.MUTED)
    common.save_fig(fig, "fig_e10_rzz", formats=("pdf",))
    plt.close(fig)

    plot_circuits(df)


CIRCUIT_SEED = 0
CIRCUIT_PROTOCOL = "exhaustive"
CIRCUIT_CATALOGS = ("rzz", "rzzh", "rzzhn")


def _parse_detail(detail: str):
    state = []
    for part in detail.split(" · "):
        head, _, angle = part.partition(")·")
        name, _, qubits = head.partition("(")
        qs = tuple(int(q) for q in qubits.rstrip(")").split(",") if q != "")
        state.append((name, qs, qs if angle else None))
    return tuple(state)


def _angle(name: str, f, params):
    from qksearch.gates import ANGLE_ENCODING

    if f is None:
        return None
    encoding = ANGLE_ENCODING.get(name, "")
    angle = 1.0
    for i in f:
        angle = angle * (np.pi - params[i] if encoding.startswith("havlicek")
                         else params[i])
    return angle / (2 * np.pi) if encoding.endswith("scaled") else angle


def plot_circuits(df=None) -> None:
    import matplotlib.pyplot as plt
    from qiskit.circuit import Parameter, QuantumCircuit

    from qksearch.backend import QiskitBackend

    common.report_style()
    if df is None:
        df = common.read_csv(CSV)
    rows = []
    for catalog in CIRCUIT_CATALOGS:
        sub = df[(df["catalog"] == catalog)
                 & (df["protocol"] == CIRCUIT_PROTOCOL)
                 & (df["seed"] == CIRCUIT_SEED)]
        if not sub.empty:
            rows.append((catalog, sub.iloc[0]))
    if not rows:
        return

    drawer = QiskitBackend(n_qubits=N_QUBITS)
    params = [Parameter(f"x{f}") for f in range(N_QUBITS)]
    style = {
        "backgroundcolor": "#ffffff",
        "linecolor": common.INK2,
        "textcolor": common.INK,
        "gatetextcolor": "#ffffff",
        "displaycolor": {
            "h": [common.MUTED, "#ffffff"],
            "rz": [common.BLUE, "#ffffff"],
            "rzz": [common.RED, "#ffffff"],
            "rzzh": [common.RED, "#ffffff"],
            "rzzhn": [common.RED, "#ffffff"],
        },
    }

    images = []
    for (catalog, row) in rows:
        layer = _parse_detail(str(row["detail"]))
        qc = QuantumCircuit(N_QUBITS)
        full = (tuple(("h", (q,), None) for q in range(N_QUBITS)) + layer
                + tuple(("h", (q,), None) for q in range(N_QUBITS)))
        for name, qubits, f in full:
            drawer._apply(qc, name, qubits, _angle(name, f, params))
        drawing = qc.draw(output="mpl", style=style, fold=-1)
        buffer = io.BytesIO()
        drawing.savefig(buffer, format="png", dpi=common.FIG_DPI,
                        bbox_inches="tight")
        plt.close(drawing)
        buffer.seek(0)
        images.append(plt.imread(buffer))

    width = max(image.shape[1] for image in images)
    images = [np.pad(image, ((0, 0), (0, width - image.shape[1]), (0, 0)),
                     constant_values=1.0) for image in images]

    heights = [image.shape[0] / width for image in images]
    fig, axes = plt.subplots(
        len(images), 1, figsize=(7.2, 7.2 * sum(heights) + 0.5 * len(images)),
        height_ratios=heights, constrained_layout=True)
    for ax, image, (catalog, row) in zip(np.atleast_1d(axes), images, rows):
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(f"{LABELS[catalog]} — KTA {row['kta_train']:+.4f}   "
                     f"$R^2$ {row['r2_test']:+.3f}   "
                     f"{int(row['n_entanglers'])} entangleur"
                     f"{'s' if int(row['n_entanglers']) > 1 else ''}",
                     fontsize=8, loc="left")
    fig.suptitle(f"optimum exact du KTA sur MOF, graine {CIRCUIT_SEED} "
                 f"({N_DATA_GATES} portes de données au plus)", fontsize=9)
    common.save_fig(fig, "fig_e10_rzz", formats=("png",))
    plt.close(fig)


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot()
