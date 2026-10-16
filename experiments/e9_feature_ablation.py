from __future__ import annotations

import sys
import time

import numpy as np

from experiments import common

CSV = "e9_ablation.csv"
CSV_CORR = "e9_correlation.csv"
CSV_COV = "e9_covariance.csv"
CSV_MULTI = "e9_multiple_corr.csv"
SEEDS = list(range(10))
VARIANTS = {"all": None, "no_x0": 0, "no_x1": 1, "no_x2": 2, "no_x3": 3}

TRIPLES = {"x0_x2_x3": (0, 2, 3), "x1_x2_x3": (1, 2, 3)}


def _multiple_corr(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((residual ** 2).sum()) / ss_tot
    return float(np.sqrt(max(r2, 0.0))), r2


def run() -> None:
    import pandas as pd
    from sklearn.metrics.pairwise import rbf_kernel

    from qksearch.dataset import RegressionDataset
    from qksearch.dataset.mof import CSV_PATH, FEATURES, TARGET
    from experiments.e5_benchmark import _r2, _rbf_gamma_cv

    t0 = time.time()

    columns = [*FEATURES, TARGET]
    raw = pd.read_csv(CSV_PATH)
    corr = raw[columns].corr()
    corr_bet_vol = float(corr.loc["bet_m2_g", "volume_m3_g"])

    print("correlation matrix (Pearson, raw units):")
    print(f"{'':>15s}" + "".join(f"{c[:13]:>15s}" for c in columns))
    for a in columns:
        print(f"{a:>15s}"
              + "".join(f"{corr.loc[a, b]:+15.3f}" for b in columns))
    common.write_csv(CSV_CORR, [
        {"variable": a, **{b: float(corr.loc[a, b]) for b in columns}}
        for a in columns
    ])

    cov = raw[columns].cov()
    print("\ncovariance matrix (raw units):")
    print(f"{'':>15s}" + "".join(f"{c[:13]:>15s}" for c in columns))
    for a in columns:
        print(f"{a:>15s}"
              + "".join(f"{cov.loc[a, b]:+15.4g}" for b in columns))
    common.write_csv(CSV_COV, [
        {"variable": a, **{b: float(cov.loc[a, b]) for b in columns}}
        for a in columns
    ])

    y = raw[TARGET].to_numpy(dtype=float)
    multi_rows: list[dict] = []
    print("\nmultiple correlation with x4 (%s):" % TARGET)
    for name, idx in TRIPLES.items():
        names = [FEATURES[i] for i in idx]
        r, r2 = _multiple_corr(raw[names].to_numpy(dtype=float), y)
        multi_rows.append({"triple": name, "features": "+".join(names),
                           "R": r, "R2": r2})
        print(f"  ({', '.join(f'x{i}' for i in idx)}) → R = {r:+.3f}"
              f"  R² = {r2:+.3f}   [{', '.join(names)}]")
    common.write_csv(CSV_MULTI, multi_rows)

    e5 = common.read_csv("e5_benchmark.csv")
    mof = e5[(e5["dataset"] == "mof") & e5["method"].str.startswith("quantum")]
    usage = {f"x{i}": int(mof["detail"].str.contains(f"x{i}").sum())
             for i in range(4)}

    def drop(data, idx):
        keep = [i for i in range(data.X.shape[1]) if i != idx]
        return RegressionDataset(data.X[:, keep], data.y)

    rows: list[dict] = []
    for variant, idx in VARIANTS.items():
        for seed in SEEDS:
            train, test = common.mof_split(seed)
            if idx is not None:
                train, test = drop(train, idx), drop(test, idx)
            gamma = _rbf_gamma_cv(train)
            rows.append({
                "variant": variant, "seed": seed,
                "r2_cv": _r2(rbf_kernel(train.X, gamma=gamma),
                             rbf_kernel(test.X, train.X, gamma=gamma),
                             train, test),
            })
        med = np.median([r["r2_cv"] for r in rows if r["variant"] == variant])
        print(f"{variant:6s} median {med:+.3f} ({time.time() - t0:.1f}s)")

    common.write_csv(CSV, rows)
    common.write_meta("e9_feature_ablation", time.time() - t0,
                      seeds=SEEDS, corr_bet_vol=corr_bet_vol,
                      corr_matrix={a: {b: float(corr.loc[a, b])
                                       for b in columns} for a in columns},
                      cov_matrix={a: {b: float(cov.loc[a, b])
                                      for b in columns} for a in columns},
                      multiple_corr={r["triple"]: {"R": r["R"], "R2": r["R2"]}
                                     for r in multi_rows},
                      quantum_searches=len(mof), feature_usage=usage)


SHORT = {"bet_m2_g": "$x_0$ BET", "volume_m3_g": "$x_1$ volume",
         "pressure_bar": "$x_2$ pression", "temperature_k": "$x_3$ temp.",
         "adsorption_pct": "$y$ adsorption"}


def plot() -> None:
    import matplotlib.pyplot as plt

    common.report_style()
    corr = common.read_csv(CSV_CORR).set_index("variable")
    multi = common.read_csv(CSV_MULTI)
    df = common.read_csv(CSV)

    e5 = common.read_csv("e5_benchmark.csv")
    mof = e5[(e5["dataset"] == "mof") & e5["method"].str.startswith("quantum")]
    usage = [int(mof["detail"].str.contains(f"x{i}").sum()) for i in range(4)]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))

    ax = axes[0]
    colors = [common.RED if u == min(usage) else common.BLUE for u in usage]
    ax.bar(range(4), usage, color=colors, width=0.68)
    for i, u in enumerate(usage):
        ax.text(i, u + 0.6, f"{u}/{len(mof)}", ha="center", va="bottom",
                fontsize=7.5, color=common.INK2)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"$x_{i}$" for i in range(4)])
    ax.set_ylim(0, len(mof) * 1.18)
    ax.set_ylabel("circuits MOF gardant la feature")
    ax.set_title("(a) $x_1$ est presque toujours jeté")
    ax.grid(True, axis="y", alpha=0.5)

    ax = axes[1]
    labels = list(corr.columns)
    M = corr.loc[labels, labels].to_numpy(dtype=float)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if abs(M[i, j]) > 0.6 else common.INK)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([SHORT[c] for c in labels], rotation=45, ha="right",
                       fontsize=6.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([SHORT[c] for c in labels], fontsize=6.5)
    ax.tick_params(length=0)
    bet_vol = float(corr.loc["bet_m2_g", "volume_m3_g"])
    ax.set_title(f"(b) $x_0$–$x_1$ colinéaires ($r = {bet_vol:.2f}$)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(
        labelsize=6.5, length=0)

    ax = axes[2]
    variants = list(VARIANTS)
    rng = np.random.default_rng(0)
    for i, variant in enumerate(variants):
        vals = df[df["variant"] == variant]["r2_cv"].to_numpy(dtype=float)
        med = float(np.median(vals))
        colour = common.RED if variant == "no_x1" else common.BLUE
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(vals)), vals, s=9,
                   color=colour, alpha=0.55, lw=0)
        ax.plot([i - 0.3, i + 0.3], [med, med], color=common.INK, lw=1.6,
                zorder=3, solid_capstyle="butt")
    baseline = float(np.median(df[df["variant"] == "all"]["r2_cv"]))
    ax.axhline(baseline, color=common.INK2, lw=0.9, ls=":")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(["toutes"] + [f"sans $x_{v[-1]}$" for v in variants[1:]],
                       fontsize=7.5)
    ax.set_ylabel("$R^2$ test (RBF, $\\gamma$ par CV)")
    ax.set_title("(c) retirer $x_1$ n'abîme rien")
    ax.grid(True, axis="y", alpha=0.5)
    triples = ", ".join(
        f"$R^2({t.replace('_', ',').replace('x', 'x_')}) = {r:.2f}$"
        for t, r in zip(multi["triple"], multi["R2"]))
    ax.text(0.5, -0.34, f"corrélation multiple linéaire : {triples}",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.5,
            color=common.MUTED)

    fig.tight_layout()
    common.save_fig(fig, "fig_e9_ablation")
    plt.close(fig)


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot()
