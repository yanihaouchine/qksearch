from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from qksearch.compare_kernels import BASELINE, CMAP_DIV, FIG_DIR, INK, INK2, RED, draw_circuit
from qksearch.dataset.mof import CSV_PATH, FEATURES, _read_csv

FIG_PATH = os.path.join(FIG_DIR, "fig_corr_bet_volume.png")
IQP_EXAMPLE_PATH = os.path.join(FIG_DIR, "fig_iqp_example.png")

IQP_EXAMPLE_STATE = (
    ("h", (0,), None), ("h", (1,), None),
    ("rz", (0,), (0,)), ("rz", (1,), (1,)),
    ("rzz", (0, 1), (0, 1)),
    ("rz", (0,), (0,)), ("rz", (1,), (1,)),
    ("rzz", (0, 1), (0, 1)),
    ("h", (0,), None), ("h", (1,), None),
)


def iqp_example() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    draw_circuit(IQP_EXAMPLE_STATE, n_qubits=2, n_features=2,
                 path=IQP_EXAMPLE_PATH)
    print(f"figure → {IQP_EXAMPLE_PATH}")


def main() -> None:
    X, y = _read_csv(CSV_PATH)
    bet, volume = X[:, FEATURES.index("bet_m2_g")], X[:, FEATURES.index("volume_m3_g")]
    rho = float(np.corrcoef(bet, volume)[0, 1])

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    y_c = (y - y.mean()) / y.std()
    lim = float(np.abs(y_c).max())
    sc = ax.scatter(bet, volume, c=y_c, cmap=CMAP_DIV, vmin=-lim, vmax=lim,
                    s=42, edgecolors=INK2, linewidths=0.5, zorder=3)

    slope, intercept = np.polyfit(bet, volume, 1)
    xs = np.linspace(bet.min(), bet.max(), 2)
    ax.plot(xs, slope * xs + intercept, color=RED, linewidth=1.8,
            linestyle="--", zorder=2)

    ax.annotate(f"correlation ρ = {rho:+.2f}", xy=(0.04, 0.92),
                xycoords="axes fraction", fontsize=13, color=INK)
    ax.set_xlabel("x₀ — BET surface area (m²/g)")
    ax.set_ylabel("x₁ — pore volume (m³/g)")
    ax.set_title("MOF: the first two variables are redundant",
                 loc="left", pad=10)
    ax.grid(True)
    ax.set_axisbelow(True)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("target y (standardized)", color=INK2)
    cbar.outline.set_edgecolor(BASELINE)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(FIG_PATH)
    plt.close(fig)
    print(f"figure → {FIG_PATH}  (ρ = {rho:+.3f})")


if __name__ == "__main__":
    main()
    iqp_example()
