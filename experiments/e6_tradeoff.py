from __future__ import annotations

import sys
import time

import numpy as np

from experiments import common

CSV = "e6_tradeoff.csv"
CSV_CORR = "e6_correlations.csv"
LAMBDAS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9, 1.0, 1.3, 1.6, 2.0)
CS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)
DIM_MAX = {"sine": 4 ** 2 - 1, "mof": 4 ** 4 - 1}


def run() -> None:
    from scipy.stats import spearmanr

    t0 = time.time()
    df = common.read_csv("e3_expressivity.csv")

    rows: list[dict] = []
    for space, sub in df.groupby("space"):
        kta = sub["kta"].to_numpy()
        dla_norm = sub["dla"].to_numpy() / DIM_MAX[space]
        var4 = sub["var4"].to_numpy()
        r2 = sub["r2"].to_numpy()
        dla = sub["dla"].to_numpy()

        for lam in LAMBDAS:
            i = int(np.argmax(kta - lam * dla_norm))
            rows.append({"space": space, "regularizer": "dla", "weight": lam,
                         "winner_id": int(sub["circuit_id"].iloc[i]),
                         "winner_kta": float(kta[i]), "winner_r2": float(r2[i]),
                         "winner_dla": int(dla[i]),
                         "winner_var4": float(var4[i])})
        for c in CS:
            i = int(np.argmax(kta + c * var4))
            rows.append({"space": space, "regularizer": "variance", "weight": c,
                         "winner_id": int(sub["circuit_id"].iloc[i]),
                         "winner_kta": float(kta[i]), "winner_r2": float(r2[i]),
                         "winner_dla": int(dla[i]),
                         "winner_var4": float(var4[i])})
    common.write_csv(CSV, rows)

    e2 = common.read_csv("e2_depth_trends.csv")
    corr_rows = []
    for (config, seed), sub in e2.groupby(["config", "seed"]):
        corr_rows.append({
            "config": config, "seed": seed,
            "rho_kta_dla": float(spearmanr(sub["kta"], sub["dla"]).statistic),
            "rho_kta_var": float(spearmanr(sub["kta"], sub["var4"]).statistic),
        })
    for space, sub in df.groupby("space"):
        corr_rows.append({
            "config": f"{space}-r2", "seed": 0,
            "rho_kta_dla": float(spearmanr(sub["dla"], sub["r2"]).statistic),
            "rho_kta_var": float(spearmanr(sub["var4"], sub["r2"]).statistic),
        })
    common.write_csv(CSV_CORR, corr_rows)
    common.write_meta("e6_tradeoff", time.time() - t0,
                      lambdas=list(LAMBDAS), cs=list(CS))


def plot() -> None:
    import matplotlib.pyplot as plt

    common.report_style()
    df = common.read_csv(CSV)

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.4))
    colors = {"sine": common.BLUE, "mof": common.RED}
    labels = {"sine": "synthetic (35 circuits)", "mof": "MOF (1001 circuits)"}

    for ax, (reg, xlabel, marks) in zip(axes, (
            ("dla", r"DLA penalty weight $\lambda$",
             (common.PRODUCTION_WEIGHTS["dla"], common.BALANCE_WEIGHTS["dla"])),
            ("variance", r"variance bonus weight $c$",
             (common.PRODUCTION_WEIGHTS["variance"],
              common.BALANCE_WEIGHTS["variance"])))):
        for space in ("sine", "mof"):
            sub = df[(df["space"] == space) & (df["regularizer"] == reg)]
            ax.plot(sub["weight"], sub["winner_r2"], color=colors[space],
                    lw=1.6, marker="o", ms=3, label=labels[space])
        prod, bal = marks
        ax.axvline(prod, color=common.INK2, lw=0.9, ls=":")
        ax.axvline(bal, color=common.INK2, lw=0.9, ls="--")
        ax.text(prod, ax.get_ylim()[0], " tie-break", fontsize=6.5,
                color=common.INK2, ha="left", va="bottom", rotation=90)
        ax.text(bal, ax.get_ylim()[0], " balance", fontsize=6.5,
                color=common.INK2, ha="left", va="bottom", rotation=90)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.5)
    axes[0].set_ylabel("test $R^2$ of selected circuit")
    axes[0].legend(loc="center left", fontsize=7)
    fig.tight_layout()
    common.save_fig(fig, "fig_e6_tradeoff")
    plt.close(fig)


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot()
