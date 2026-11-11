from __future__ import annotations

import json
import sys
import time
from functools import reduce
from operator import mul

import numpy as np

from experiments import common

CSV = "e13_bandwidth.csv"

N_QUBITS = 4
N_DATA_GATES = 6
MCTS_ITERATIONS = 1000
SEED = 0
SCALES = tuple(float(np.pi * 2.0 ** k) for k in range(-7, 2))
REFERENCE = common.REFERENCE_SCALE


def _r2(K_train, K_test, train, test) -> float:
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.metrics import r2_score

    krr = KernelRidge(alpha=common.RIDGE_ALPHA,
                      kernel="precomputed").fit(K_train, train.y)
    return float(r2_score(test.y, krr.predict(K_test)))


def _search_at(scale: float) -> dict:
    from qksearch.backend import NumpyBackend
    from qksearch.enumerate_circuits import strip_identity
    from qksearch.iqp import build_heuristic, describe, iqp_circuit
    from qksearch.search import MctsSearch, search

    train, test = common.mof_split(SEED, scale=scale)
    backend = NumpyBackend(N_QUBITS)
    heuristic = build_heuristic(train, n_qubits=N_QUBITS,
                                n_data_gates=N_DATA_GATES,
                                regularization="none", weight=0.0,
                                backend=backend)
    res = search((), heuristic,
                 MctsSearch(n_iterations=MCTS_ITERATIONS, seed=SEED))
    layer = strip_identity(res.best_state)
    kta = float(heuristic.kta(res.best_state))

    circuit = iqp_circuit(layer, N_QUBITS)
    K = backend.gram(circuit, train.X)
    K_test = backend.cross_gram(circuit, test.X, train.X)
    return {
        "scale": scale,
        "scale_over_pi": scale / np.pi,
        "kta_train": kta,
        "r2_test": _r2(K, K_test, train, test),
        "cv_r2": common.cv_r2(K, train.y, common.cv_folds(len(train.y),
                                                          seed=SEED)),
        "n_gates": len(layer),
        "n_entanglers": sum(1 for op in layer if len(op[1]) == 2),
        "n_evaluated": res.n_evaluated,
        "detail": describe(layer),
        "layer_json": json.dumps([[name, list(qubits),
                                   None if f is None else list(f)]
                                  for name, qubits, f in layer]),
    }


def _check_reference(row: dict) -> None:
    df = common.read_csv("e5_benchmark.csv")
    ref = df[(df["dataset"] == "mof") & (df["method"] == "quantum-kta")
             & (df["seed"] == SEED)].iloc[0]
    assert ref["detail"] == row["detail"], (ref["detail"], row["detail"])
    assert abs(ref["kta_train"] - row["kta_train"]) < 1e-9
    print(f"  cross-check with E5/E8 (s = π/4): {row['detail']}")


def run() -> None:
    t0 = time.time()
    rows: list[dict] = []
    for scale in SCALES:
        row = _search_at(scale)
        rows.append(row)
        print(f"s = {row['scale_over_pi']:.5f}π  KTA {row['kta_train']:+.4f}  "
              f"R² {row['r2_test']:+.4f}  {row['n_gates']} gates "
              f"({row['n_entanglers']} rzz)  {row['detail']}  "
              f"({time.time() - t0:.1f}s)")
        if scale == REFERENCE:
            _check_reference(row)

    common.write_csv(CSV, rows)

    distinct = len({r["detail"] for r in rows})
    print(f"\n{distinct} distinct circuits for {len(rows)} bandwidths")
    common.write_meta("e13_bandwidth", time.time() - t0, seed=SEED,
                      n_qubits=N_QUBITS, n_data_gates=N_DATA_GATES,
                      mcts_iterations=MCTS_ITERATIONS,
                      scales_over_pi=[s / np.pi for s in SCALES],
                      reference_scale_over_pi=REFERENCE / np.pi,
                      bound_sigmas=common.BOUND_SIGMAS,
                      n_distinct_circuits=distinct,
                      summary={f"{r['scale_over_pi']:.5f}": {
                          "detail": r["detail"], "kta_train": r["kta_train"],
                          "r2_test": r["r2_test"]} for r in rows})


def _scale_label(scale_over_pi: float) -> str:
    k = int(round(np.log2(scale_over_pi)))
    if k == 0:
        return "$\\pi$"
    if k > 0:
        return f"${2 ** k}\\pi$"
    return f"$\\pi/{2 ** -k}$"


DRAW_DPI = 300


def _render(layer, style):
    from io import BytesIO

    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from qiskit.circuit import Parameter, QuantumCircuit

    from qksearch.backend import QiskitBackend

    drawer = QiskitBackend(n_qubits=N_QUBITS)
    params = [Parameter(f"x{f}") for f in range(N_QUBITS)]
    qc = QuantumCircuit(N_QUBITS)
    full = (tuple(("h", (q,), None) for q in range(N_QUBITS)) + tuple(layer)
            + tuple(("h", (q,), None) for q in range(N_QUBITS)))
    for name, qubits, f in full:
        angle = None if f is None else reduce(mul, (params[i] for i in f))
        drawer._apply(qc, name, tuple(qubits), angle)

    fig = qc.draw(output="mpl", style=style, fold=-1)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=DRAW_DPI, bbox_inches="tight",
                pad_inches=0.03, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return mpimg.imread(buf)


def _pad(images: list) -> list:
    h = max(im.shape[0] for im in images)
    w = max(im.shape[1] for im in images)
    out = []
    for im in images:
        canvas = np.ones((h, w, 4), dtype=im.dtype)
        top = (h - im.shape[0]) // 2
        canvas[top:top + im.shape[0], :im.shape[1]] = im
        out.append(canvas)
    return out


def plot() -> None:
    import matplotlib.pyplot as plt

    common.report_style()
    df = common.read_csv(CSV).sort_values("scale_over_pi").reset_index(drop=True)

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

    images = _pad([_render([(name, tuple(q), None if f is None else tuple(f))
                            for name, q, f in json.loads(row["layer_json"])],
                           style)
                   for _, row in df.iterrows()])

    n = len(df)
    ncols = 2
    nrows = -(-n // ncols)
    panel_w = 6.0
    panel_h = panel_w * images[0].shape[0] / images[0].shape[1]
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * panel_w + 0.4,
                                      nrows * (panel_h + 0.42) + 0.7))
    flat = np.asarray(axes).ravel()

    for ax, img, (_, row) in zip(flat, images, df.iterrows()):
        ax.imshow(img, interpolation="antialiased")
        ax.set_axis_off()
        ax.set_title(f"$s = ${_scale_label(row['scale_over_pi'])}   ·   "
                     f"KTA {row['kta_train']:+.3f}   ·   "
                     f"$R^2$ test {row['r2_test']:+.3f}   ·   "
                     f"{int(row['n_gates'])} gates",
                     fontsize=9, loc="left", pad=5, color=common.INK)

    for ax in flat[n:]:
        ax.set_axis_off()

    fig.suptitle("Circuit selected by the search, one bandwidth at a time\n"
                 f"(MCTS on the KTA, {MCTS_ITERATIONS} iterations, MOF, "
                 f"{N_QUBITS} qubits, {N_DATA_GATES} gates, seed {SEED})",
                 fontsize=11, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.2, w_pad=1.0)
    common.save_fig(fig, "fig_e13_bandwidth")
    plt.close(fig)


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot()
