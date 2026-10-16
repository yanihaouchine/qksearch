from __future__ import annotations

import os
import sys
import time

import numpy as np

from experiments import common

SMOKE = bool(os.environ.get("E14_SMOKE"))
_SUFFIX = "_smoke" if SMOKE else ""

CSV_RANK = f"e14_rank{_SUFFIX}.csv"
CSV_NOISE = f"e14_noise{_SUFFIX}.csv"

SEEDS = [0] if SMOKE else list(range(10))
SHOTS = (10, 1000) if SMOKE else (1, 3, 10, 30, 100, 1000, 10000)
N_REPS = 2 if SMOKE else 5
DATASETS = {"sine": 2} if SMOKE else {"sine": 2, "mof": 4}
SPACE_DEPTH = 4
REFERENCE = common.REFERENCE_SCALE
TOP_K = 10
RHO_FLOOR = 0.99
AGREEMENT_FLOOR = 0.8


def _split(dataset: str, seed: int):
    if dataset == "sine":
        return common.sine_split(seed, scale=REFERENCE)
    return common.mof_split(seed, scale=REFERENCE)


def _gate_set(n: int):
    from qksearch.gates import GateSpec, ProductGateSet

    return ProductGateSet(
        [GateSpec("rz", 1, uses_feature=True),
         GateSpec("rzz", 2, uses_feature=True, symmetric=True)],
        n_qubits=n, n_features=n, with_identity=True)


def _enumerate(gs, depth: int) -> list:
    states = [()]
    for _ in range(depth):
        states = [s + (op,) for s in states for op in gs.placements(s)]
    return states


def _heuristic(train, gs, n: int, depth: int):
    from qksearch.backend import NumpyBackend
    from qksearch.iqp import IqpKernelHeuristic
    from qksearch.metric import TargetAlignment

    return IqpKernelHeuristic(
        gate_set=gs, backend=NumpyBackend(n), data=train,
        metric=TargetAlignment(), max_depth=depth, n_qubits=n,
        regularizer=None)


def _sample_upper(rng, upper: np.ndarray, shots) -> np.ndarray:
    if shots is None:
        return upper
    return rng.binomial(shots, upper) / float(shots)


def _rebuild(upper: np.ndarray, n: int) -> np.ndarray:
    iu = np.triu_indices(n, k=1)
    K = np.empty((n, n))
    K[iu] = upper
    K.T[iu] = upper
    np.fill_diagonal(K, 1.0)
    return K


def _kta_batch(uppers: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(y)
    iu = np.triu_indices(n, k=1)
    target = np.outer(y, y)
    num = 2.0 * (uppers @ target[iu]) + float((y ** 2).sum())
    den = (np.sqrt(2.0 * (uppers ** 2).sum(axis=-1) + n)
           * float(np.linalg.norm(target)))
    return num / den


def _reference(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    df = common.read_csv("e13_space.csv")
    sub = df[(df["dataset"] == dataset)
             & np.isclose(df["scale_over_pi"], REFERENCE / np.pi)]
    kta = sub.pivot_table(index="seed", columns="circuit_id", values="kta")
    r2 = sub.pivot_table(index="seed", columns="circuit_id", values="r2")
    assert list(kta.index) == sorted(sub["seed"].unique())
    return kta.to_numpy(), r2.to_numpy()


def _check_against_e3(dataset: str, ktas: np.ndarray) -> None:
    df = common.read_csv("e3_expressivity.csv")
    ref = df[df["space"] == dataset].sort_values("circuit_id")["kta"].to_numpy()
    assert len(ref) == len(ktas), (dataset, len(ref), len(ktas))
    gap = float(np.abs(ref - ktas).max())
    assert gap < 1e-9, f"{dataset}: E14 diverges from E3 by {gap:.2e}"
    print(f"  recoupement E3 ({dataset}) : écart max {gap:.2e}")


def _check_kta_paths(heuristic, states, y, ktas: np.ndarray) -> None:
    from qksearch.metric import TargetAlignment

    kta = TargetAlignment()
    for state, value in zip(states, ktas):
        direct = float(kta(heuristic.evaluator.gram(state), y))
        assert abs(direct - value) < 1e-12, (direct, value)


def _check_sampler(upper: np.ndarray, shots: int, n: int) -> None:
    rng = np.random.default_rng(12345)
    K = _rebuild(_sample_upper(rng, upper, shots), n)
    assert np.array_equal(K, K.T)
    assert np.all(np.diag(K) == 1.0)
    assert np.all((0.0 <= K) & (K <= 1.0))
    counts = K[np.triu_indices(n, k=1)] * shots
    assert np.allclose(counts, np.round(counts))


def _run_rank() -> tuple[list[dict], list[dict]]:
    from qksearch.enumerate_circuits import strip_identity
    from qksearch.iqp import describe
    from scipy.stats import spearmanr

    rows: list[dict] = []
    noise: list[dict] = []
    t0 = time.time()
    for dataset, n in DATASETS.items():
        gs = _gate_set(n)
        states = _enumerate(gs, SPACE_DEPTH)
        assert len(states) == common.space_size(n, SPACE_DEPTH)
        ref_kta, ref_r2 = _reference(dataset)
        details = [describe(strip_identity(s)) or "identité" for s in states]

        for seed in SEEDS:
            train, _ = _split(dataset, seed)
            heuristic = _heuristic(train, gs, n, SPACE_DEPTH)
            nt = len(train.y)
            iu = np.triu_indices(nt, k=1)
            uppers = np.empty((len(states), len(iu[0])))
            for ci, state in enumerate(states):
                uppers[ci] = np.clip(heuristic.evaluator.gram(state)[iu],
                                     0.0, 1.0)
            kta_exact = _kta_batch(uppers, train.y)
            r2_exact = ref_r2[seed]

            if seed == SEEDS[0]:
                _check_kta_paths(heuristic, states[:20], train.y, kta_exact[:20])
                _check_sampler(uppers[0], 100, nt)
            if seed == 0:
                _check_against_e3(dataset, kta_exact)
            gap = float(np.abs(ref_kta[seed] - kta_exact).max())
            assert gap < 1e-9, f"{dataset} seed {seed}: E13 gap {gap:.2e}"

            i_best_kta = int(np.argmax(kta_exact))
            top_exact = set(np.argsort(kta_exact)[::-1][:TOP_K].tolist())
            rank_of = np.empty(len(states), dtype=int)
            rank_of[np.argsort(kta_exact)[::-1]] = np.arange(len(states))
            pairs = nt * (nt - 1) // 2

            for si, shots in enumerate((*SHOTS, None)):
                for rep in range(1 if shots is None else N_REPS):
                    rng = np.random.default_rng([seed, si, rep])
                    noisy = _sample_upper(rng, uppers, shots)
                    kn = _kta_batch(noisy, train.y)
                    i = int(np.argmax(kn))
                    order = np.argsort(kn)[::-1]
                    rank_noisy = np.empty(len(states), dtype=int)
                    rank_noisy[order] = np.arange(len(states))
                    rows.append({
                        "dataset": dataset, "seed": seed,
                        "shots": float(shots) if shots else float("inf"),
                        "rep": rep,
                        "shots_per_circuit": (float(shots) * pairs if shots
                                              else float("inf")),
                        "spearman_rho": float(spearmanr(kn, kta_exact).statistic),
                        "top10_overlap": len(top_exact
                                             & set(order[:TOP_K].tolist()))
                        / float(TOP_K),
                        "rank_of_exact_best": int(
                            np.flatnonzero(order == i_best_kta)[0]),
                        "elected_id": i,
                        "elected_rank_exact": int(rank_of[i]),
                        "kta_elected_noisy": float(kn[i]),
                        "kta_elected_exact": float(kta_exact[i]),
                        "kta_best_exact": float(kta_exact[i_best_kta]),
                        "regret_kta": float(kta_exact[i_best_kta]
                                            - kta_exact[i]),
                        "regret_proxy": float(r2_exact.max()
                                              - r2_exact[i_best_kta]),
                        "regret_select": float(r2_exact[i_best_kta]
                                               - r2_exact[i]),
                        "r2_elected_exact": float(r2_exact[i]),
                        "r2_best_exact": float(r2_exact.max()),
                    })
                    if seed == SEEDS[0]:
                        for ci in range(len(states)):
                            noise.append({
                                "dataset": dataset,
                                "shots": (float(shots) if shots
                                          else float("inf")),
                                "rep": rep, "circuit_id": ci,
                                "detail": details[ci],
                                "kta_exact": round(float(kta_exact[ci]), 9),
                                "kta_noisy": round(float(kn[ci]), 9),
                                "rank_exact": int(rank_of[ci]),
                                "rank_noisy": int(rank_noisy[ci]),
                            })
            print(f"  axe A : {dataset} seed {seed} ({time.time() - t0:.0f}s)")
    return rows, noise


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    keep = (x > 0) & (y > 0)
    if keep.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[keep]), np.log(y[keep]), 1)[0])


def _print_rankings(sub) -> dict:
    exact = (sub.drop_duplicates("circuit_id")
             .sort_values("rank_exact").head(TOP_K))
    labels = dict(zip(sub["circuit_id"], sub["detail"]))
    width = max(len(d) for d in exact["detail"])
    shot_levels = [s for s in sorted(sub["shots"].unique()) if np.isfinite(s)]
    out: dict = {"exact": [], "by_shots": {}}

    for _, row in exact.iterrows():
        out["exact"].append({"circuit_id": int(row["circuit_id"]),
                             "detail": row["detail"],
                             "kta": float(row["kta_exact"])})
    for shots in shot_levels:
        g = sub[(sub["shots"] == shots) & (sub["rep"] == 0)]
        top = g.sort_values("rank_noisy").head(TOP_K)
        out["by_shots"][str(int(shots))] = [
            {"circuit_id": int(r["circuit_id"]), "detail": r["detail"],
             "kta": float(r["kta_noisy"]), "rank_exact": int(r["rank_exact"])}
            for _, r in top.iterrows()]

    for shots in shot_levels:
        measured = out["by_shots"][str(int(shots))]
        hits = sum(1 for m in measured
                   if m["rank_exact"] < TOP_K)
        print(f"\n  M = {int(shots)} — {hits}/{TOP_K} du top-{TOP_K} exact "
              f"retrouvés (tirage rep = 0)")
        print(f"    {'rang':>4s}  {'classement exact':<{width}s}  KTA     "
              f"| {'classement mesuré':<{width}s}  KTA     rang exact")
        for k, (e, m) in enumerate(zip(out["exact"], measured)):
            flag = " " if m["rank_exact"] < TOP_K else "*"
            print(f"    {k:>4d}  {labels[e['circuit_id']]:<{width}s}  "
                  f"{e['kta']:+.4f} | {m['detail']:<{width}s}  "
                  f"{m['kta']:+.4f}  {m['rank_exact']:>6d}{flag}")
    return out


def _summarize(df_rank, df_noise) -> dict:
    out: dict = {"rank": {}, "noise": {}, "ranking": {}}

    print("\n— axe A : le classement du KTA sous bruit de mesure —")
    for dataset in DATASETS:
        sub = df_rank[df_rank["dataset"] == dataset]
        exact_pick = (sub[~np.isfinite(sub["shots"])]
                      .set_index("seed")["elected_id"])
        stats: dict[str, dict] = {}
        for shots in sorted(sub["shots"].unique()):
            g = sub[sub["shots"] == shots]
            key = "inf" if not np.isfinite(shots) else str(int(shots))
            stats[key] = {
                "rho": float(g["spearman_rho"].median()),
                "agreement": float((g["elected_id"]
                                    == g["seed"].map(exact_pick)).mean()),
                "top10": float(g["top10_overlap"].median()),
                "rank_of_best": float(g["rank_of_exact_best"].median()),
                "regret_select": float(g["regret_select"].median()),
            }
        finite = [s for s in sorted(sub["shots"].unique()) if np.isfinite(s)]
        star = next((int(s) for s in finite
                     if stats[str(int(s))]["rho"] >= RHO_FLOOR
                     and stats[str(int(s))]["agreement"] >= AGREEMENT_FLOOR),
                    None)
        out["rank"][dataset] = {
            "by_shots": stats, "shots_star": star,
            "regret_proxy": float(sub["regret_proxy"].median()),
        }
        print(f"\n{dataset} : M* = {star}, "
              f"regret de proxy (constante, E3/E13) "
              f"{sub['regret_proxy'].median():+.4f}")
        print("  M       ρ      top-10  rang du vrai   accord   regret sél.")
        for shots in [*finite, float("inf")]:
            key = "inf" if not np.isfinite(shots) else str(int(shots))
            s = stats[key]
            print(f"  {key:>6s}  {s['rho']:+.4f}  {s['top10']:5.0%}  "
                  f"{s['rank_of_best']:11.1f}  {s['agreement']:6.0%}  "
                  f"{s['regret_select']:+.4f}")

    print("\n— le bruit du KTA : un biais en 1/M, une dispersion en 1/√M —")
    for dataset in DATASETS:
        sub = df_noise[df_noise["dataset"] == dataset]
        finite = sorted(s for s in sub["shots"].unique() if np.isfinite(s))
        n_circuits = sub["circuit_id"].nunique()
        bias, sd = [], []
        for shots in finite:
            g = sub[sub["shots"] == shots].groupby("circuit_id")
            bias.append(float((g["kta_noisy"].mean()
                               - g["kta_exact"].first()).median()))
            sd.append(float(g["kta_noisy"].std().median()))
        x = np.array(finite, dtype=float)
        bias_a, sd_a = np.abs(np.array(bias)), np.array(sd)
        floor = 3.0 * sd_a / np.sqrt(N_REPS * n_circuits)
        resolved = bias_a > floor
        b_slope = (_slope(x[resolved], bias_a[resolved])
                   if resolved.sum() >= 3 else float("nan"))
        s_slope = _slope(x, sd_a)
        keys = [str(int(v)) for v in finite]
        out["noise"][dataset] = {
            "bias_by_shots": dict(zip(keys, [round(b, 9) for b in bias])),
            "sd_by_shots": dict(zip(keys, [round(v, 9) for v in sd])),
            "bias_resolved_shots": [int(v) for v in x[resolved]],
            "bias_slope": b_slope, "sd_slope": s_slope,
        }
        shown = (f"M^{b_slope:+.2f}" if np.isfinite(b_slope)
                 else f"non résolu ({int(resolved.sum())} points sur {len(x)})")
        print(f"  {dataset:5s} biais ∝ {shown} "
              f"(attendu −1), dispersion ∝ M^{s_slope:+.2f} (attendu −0,5)")
        if not SMOKE:
            assert np.isnan(b_slope) or -1.3 < b_slope < -0.7, b_slope
            assert -0.65 < s_slope < -0.35, s_slope

    print("\n— les deux classements, côte à côte —")
    for dataset in DATASETS:
        sub = df_noise[df_noise["dataset"] == dataset]
        print(f"\n{dataset} :")
        out["ranking"][dataset] = _print_rankings(sub)
    return out


def run() -> None:
    import pandas as pd

    t0 = time.time()
    print("— axe A : classement —")
    rows_rank, rows_noise = _run_rank()
    common.write_csv(CSV_RANK, rows_rank)
    common.write_csv(CSV_NOISE, rows_noise)

    summary = _summarize(pd.DataFrame(rows_rank), pd.DataFrame(rows_noise))
    common.write_meta("e14_shots", time.time() - t0, seeds=SEEDS,
                      shots=list(SHOTS),
                      n_reps=N_REPS, top_k=TOP_K,
                      scale_over_pi=REFERENCE / np.pi,
                      space_depth=SPACE_DEPTH, datasets=DATASETS,
                      noise_model="binomial", rho_floor=RHO_FLOOR,
                      agreement_floor=AGREEMENT_FLOOR, smoke=SMOKE,
                      summary=summary)


TITLES = {"sine": "sine synthétique (2 qubits)",
          "mof": "MOF, capacité H$_2$ (4 qubits)"}


def _short(dataset: str) -> str:
    return TITLES[dataset].split(",")[0].split(" (")[0]


def plot() -> None:
    import matplotlib.pyplot as plt

    common.report_style()
    df_rank = common.read_csv(CSV_RANK)
    df_noise = common.read_csv(CSV_NOISE)
    datasets = [d for d in ("mof", "sine") if d in set(df_rank["dataset"])]
    main = datasets[0]

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.2))

    ax = axes[0][0]
    for dataset, color, marker in zip(datasets, (common.BLUE, common.MUTED),
                                      ("o", "s")):
        sub = df_rank[(df_rank["dataset"] == dataset)
                      & np.isfinite(df_rank["shots"])]
        x = np.array(sorted(sub["shots"].unique()))
        g = sub.groupby("shots")
        rho = g["spearman_rho"]
        ax.fill_between(x, rho.quantile(0.25).to_numpy(),
                        rho.quantile(0.75).to_numpy(), color=color, alpha=0.18,
                        lw=0)
        ax.plot(x, rho.median().to_numpy(), color=color, lw=1.6, marker=marker,
                ms=3.5, label=f"{_short(dataset)}, $\\rho$")
        ax.plot(x, g["top10_overlap"].median().to_numpy(), color=color, lw=1.2,
                ls="--", marker=marker, ms=3, mfc="white",
                label=f"{_short(dataset)}, top-10")
    ax.axhline(1.0, color=common.BASELINE, lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("shots par entrée du Gram $M$")
    ax.set_ylabel("accord avec le classement exact")
    ax.set_title("Le classement survit au bruit", fontsize=8.5)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=6, loc="lower right")

    ax = axes[0][1]
    sub = df_noise[(df_noise["dataset"] == main) & (df_noise["rep"] == 0)]
    finite = sorted(s for s in sub["shots"].unique() if np.isfinite(s))
    picks = [finite[0], finite[len(finite) // 2], finite[-1]]
    for shots, color in zip(picks, (common.RED, common.MUTED, common.BLUE)):
        g = sub[sub["shots"] == shots]
        ax.scatter(g["kta_exact"], g["kta_noisy"], s=5, alpha=0.35, lw=0,
                   color=color, label=f"$M={int(shots)}$")
    lim = [0.0, float(sub["kta_exact"].max()) * 1.15]
    ax.plot(lim, lim, color=common.BASELINE, lw=0.8, ls="--")
    ax.set_xlim(*lim)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("KTA exact")
    ax.set_ylabel("KTA mesuré")
    ax.set_title(f"{_short(main)} : le bruit se moyenne", fontsize=8.5)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=6.5, loc="upper left", markerscale=2)

    ax = axes[1][0]
    n_circuits = int(sub["circuit_id"].nunique())
    for shots, color in zip(picks, (common.RED, common.MUTED, common.BLUE)):
        g = sub[sub["shots"] == shots]
        ax.scatter(g["rank_exact"], g["rank_noisy"], s=5, alpha=0.35, lw=0,
                   color=color, label=f"$M={int(shots)}$")
    ax.plot([0, n_circuits], [0, n_circuits], color=common.BASELINE, lw=0.8,
            ls="--")
    ax.set_xlim(0, n_circuits)
    ax.set_ylim(0, n_circuits)
    ax.set_xlabel("rang exact")
    ax.set_ylabel("rang mesuré")
    ax.set_title(f"{_short(main)} : le classement, rang à rang", fontsize=8.5)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=6.5, loc="lower right", markerscale=2)

    ax = axes[1][1]
    exact_top = (sub.drop_duplicates("circuit_id")
                 .sort_values("rank_exact").head(TOP_K))
    ids = exact_top["circuit_id"].tolist()
    ypos = np.arange(len(ids))
    for shots, color, marker in zip(picks, (common.RED, common.MUTED,
                                            common.BLUE), ("^", "s", "o")):
        g = sub[sub["shots"] == shots].set_index("circuit_id")
        ax.plot([g.loc[ci, "rank_noisy"] for ci in ids], ypos, ls="none",
                marker=marker, ms=4, mfc="none", color=color,
                label=f"$M={int(shots)}$")
    ax.plot(ypos, ypos, color=common.BASELINE, lw=0.8, ls="--")
    ax.axvline(TOP_K - 0.5, color=common.INK2, lw=0.7, ls=":")
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"#{k}" for k in ypos], fontsize=6)
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=TOP_K)
    ax.set_xlim(0, n_circuits)
    ax.set_xlabel("rang mesuré (symlog)")
    ax.set_ylabel("rang exact")
    ax.set_title(f"Le top-{TOP_K} exact, vu par la mesure", fontsize=8.5)
    ax.grid(True, alpha=0.4, axis="x")
    ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    common.save_fig(fig, "fig_e14_shots")
    plt.close(fig)


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot()
