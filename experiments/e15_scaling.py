from __future__ import annotations

import os
import sys
import time

import numpy as np

from experiments import common

SMOKE = bool(os.environ.get("E15_SMOKE"))
_SUFFIX = "_smoke" if SMOKE else ""

CSV_CONTRAST = f"e15_contrast{_SUFFIX}.csv"
CSV_SHOTS = f"e15_shots{_SUFFIX}.csv"
CSV_SEARCH = f"e15_search{_SUFFIX}.csv"

QUBITS = [2, 4] if SMOKE else [2, 3, 4, 5, 6, 7, 8]
SEARCH_QUBITS = [2, 4] if SMOKE else [2, 4, 6, 8]
SEEDS = [0] if SMOKE else list(range(5))
N_SAMPLE = 60 if SMOKE else 800
SCALES = ((common.REFERENCE_SCALE,) if SMOKE
          else tuple(float(np.pi * 2.0 ** k) for k in (-3, -2, -1)))
SHOTS = (10, 1000) if SMOKE else (1, 3, 10, 30, 100, 1000, 10000)
N_REPS = 2 if SMOKE else 5
N_SHOT_CIRCUITS = 40 if SMOKE else 200
EVAL_BUDGET = 200 if SMOKE else 2000
DATASETS = ("sine-full", "sine-fixed")
REFERENCE = common.REFERENCE_SCALE
EXHAUSTIVE_MAX_N = 4
TOP_K = 10
PSI_CACHE = 256
RHO_TARGET = 0.9


def _sine_fixed_split(seed: int, n_features: int, scale: float,
                      n_train: int = 40, n_test: int = 80,
                      noise: float = 0.05):
    from qksearch.dataset import AngleScaler, RegressionDataset

    if n_features < 2:
        raise ValueError("sine-fixed needs at least the two informative features")
    rng = np.random.default_rng(seed)
    n_samples = n_train + n_test
    X = rng.uniform(-np.pi / 2, np.pi / 2, size=(n_samples, n_features))
    y = np.sin(X[:, :2]).prod(axis=1) + rng.normal(scale=noise, size=n_samples)
    raw = RegressionDataset(X, y).standardized()
    data = AngleScaler.fit(raw.X[:n_train], scale=scale,
                           bound=common.BOUND_SIGMAS * scale).apply(raw)
    return (RegressionDataset(data.X[:n_train], data.y[:n_train]),
            RegressionDataset(data.X[n_train:], data.y[n_train:]))


def _split(dataset: str, n: int, seed: int, scale: float):
    if dataset == "sine-full":
        return common.sine_split(seed, n_features=n, scale=scale)
    return _sine_fixed_split(seed, n, scale)


def _gate_set(n: int, n_features: int = 0):
    from qksearch.gates import GateSpec, ProductGateSet

    return ProductGateSet(
        [GateSpec("rz", 1, uses_feature=True),
         GateSpec("rzz", 2, uses_feature=True, symmetric=True)],
        n_qubits=n, n_features=n_features or n, with_identity=True)


def _enumerate(gs, depth: int) -> list:
    states = [()]
    for _ in range(depth):
        states = [s + (op,) for s in states for op in gs.placements(s)]
    return states


def _heuristic(train, gs, n: int, depth: int):
    from qksearch.backend import NumpyBackend
    from qksearch.iqp import IqpKernelHeuristic
    from qksearch.metric import TargetAlignment

    h = IqpKernelHeuristic(
        gate_set=gs, backend=NumpyBackend(n), data=train,
        metric=TargetAlignment(), max_depth=depth, n_qubits=n,
        regularizer=None)
    h.evaluator.psi_cache_size = PSI_CACHE
    return h


def _n_entanglers(state) -> int:
    from qksearch.enumerate_circuits import strip_identity

    return sum(1 for op in strip_identity(state) if len(op[1]) == 2)


def _n_gates(state) -> int:
    from qksearch.enumerate_circuits import strip_identity

    return len(strip_identity(state))


def _sample_states(gs, depth: int, n_draw: int, rng) -> list:
    catalog = gs.catalog
    m = len(catalog)
    offset = np.arange(depth)
    out = []
    for _ in range(n_draw):
        idx = np.sort(rng.choice(m + depth - 1, size=depth, replace=False)) - offset
        out.append(tuple(int(i) for i in idx))
    out.sort()
    return [tuple(catalog[i] for i in idx) for idx in out]


def _uppers(evaluator, states) -> np.ndarray:
    n = evaluator.data.n_samples
    iu = np.triu_indices(n, k=1)
    out = np.empty((len(states), len(iu[0])))
    for i, state in enumerate(states):
        out[i] = evaluator.gram(state)[iu]
    return out


def _kta_batch(uppers: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(y)
    iu = np.triu_indices(n, k=1)
    target = np.outer(y, y)
    num = 2.0 * (uppers @ target[iu]) + float((y ** 2).sum())
    den = (np.sqrt(2.0 * (uppers ** 2).sum(axis=-1) + n)
           * float(np.linalg.norm(target)))
    return num / den


def _sample_upper(rng, upper: np.ndarray, shots) -> np.ndarray:
    if shots is None:
        return upper
    return rng.binomial(shots, np.clip(upper, 0.0, 1.0)) / float(shots)


def _r2(K_train, K_test, train, test) -> float:
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.metrics import r2_score

    krr = KernelRidge(alpha=common.RIDGE_ALPHA,
                      kernel="precomputed").fit(K_train, train.y)
    return float(r2_score(test.y, krr.predict(K_test)))


def _quantum_r2(heuristic, state, train, test) -> float:
    ev = heuristic.evaluator
    K_train = ev.gram(state)
    K_test = ev.backend.cross_gram(ev.circuit(state), test.X, train.X)
    return _r2(K_train, K_test, train, test)


def _gamma_by_cv(train, folds) -> float:
    from sklearn.metrics.pairwise import rbf_kernel

    gammas = np.logspace(-2, 2, 41)
    scores = [common.cv_r2(rbf_kernel(train.X, gamma=g), train.y, folds)
              for g in gammas]
    return float(gammas[int(np.argmax(scores))])


def _check_idle_qubits() -> None:
    train, _ = common.mof_split(0)
    gs4 = _gate_set(4, n_features=4)
    state = tuple(gs4.catalog[:1]) * 2
    K = [_heuristic(train, _gate_set(n, n_features=4), n, 4).evaluator.gram(state)
         for n in (4, 6)]
    gap = float(np.abs(K[0] - K[1]).max())
    assert gap < 1e-12, f"les qubits inactifs ne sont pas neutres : {gap:.2e}"
    print(f"  qubits inactifs (4 features sur 4 vs 6 qubits) : "
          f"écart max {gap:.2e}")


def _check_arms_agree() -> None:
    for seed in (0, 1):
        a = _split("sine-full", 2, seed, REFERENCE)
        b = _split("sine-fixed", 2, seed, REFERENCE)
        for da, db in zip(a, b):
            assert np.array_equal(da.X, db.X) and np.array_equal(da.y, db.y), \
                f"les deux bras divergent à n=2, seed {seed}"
    print("  bras sine-full / sine-fixed identiques à n=2 : ok")


def _check_sampler_uniform() -> None:
    from scipy.stats import chisquare

    gs = _gate_set(2)
    m, depth = len(gs.catalog), 2
    space = _enumerate(gs, depth)
    draws = _sample_states(gs, depth, 400 * len(space), np.random.default_rng(0))
    index = {s: i for i, s in enumerate(space)}
    counts = np.zeros(len(space))
    for s in draws:
        counts[index[s]] += 1
    repeats = [i for i, st in enumerate(space) if len(set(st)) == 1]
    share = float(counts[repeats].sum() / counts.sum())
    uniform, iid = len(repeats) / len(space), (1.0 / m) ** (depth - 1)
    assert abs(share - uniform) < abs(share - iid) / 2.0, (
        f"tirage non uniforme : part des répétitions {share:.4f} "
        f"(uniforme {uniform:.4f}, i.i.d. {iid:.4f})")
    p = float(chisquare(counts).pvalue)
    assert p > 1e-4, f"tirage non uniforme sur les multiensembles (p = {p:.2e})"
    print(f"  tirage uniforme sur {len(space)} multiensembles (m={m}, d={depth}) : "
          f"répétitions {share:.4f} (uniforme {uniform:.4f}, i.i.d. {iid:.4f}), "
          f"chi2 p = {p:.3f}")


def _check_binomial(upper: np.ndarray, shots: int) -> None:
    rng = np.random.default_rng(4242)
    draws = np.stack([_sample_upper(rng, upper, shots) for _ in range(4000)])
    bias = float(np.abs(draws.mean(axis=0) - upper).max())
    expected = upper * (1.0 - upper) / shots
    spread = float(np.abs(draws.var(axis=0) - expected).max())
    counts = draws * shots
    assert np.allclose(counts, np.round(counts))
    assert bias < 1e-2, f"estimateur biaisé : {bias:.2e}"
    assert spread < 5e-3, f"variance hors de K(1-K)/M : {spread:.2e}"
    print(f"  estimateur binomial à M={shots} : biais max {bias:.2e}, "
          f"écart de variance {spread:.2e}")


def _check_kta_paths(evaluator, states, y, ktas: np.ndarray) -> None:
    from qksearch.metric import TargetAlignment

    kta = TargetAlignment()
    for state, value in zip(states, ktas):
        direct = float(kta(evaluator.gram(state), y))
        assert abs(direct - value) < 1e-12, (direct, value)


def _check_against_e13() -> None:
    df = common.read_csv("e13_space.csv")
    ref = df[(df["dataset"] == "sine")
             & np.isclose(df["scale_over_pi"], REFERENCE / np.pi)
             & (df["seed"] == 0)].sort_values("circuit_id")["kta"].to_numpy()
    train, _ = common.sine_split(0, n_features=2, scale=REFERENCE)
    gs = _gate_set(2)
    states = _enumerate(gs, 4)
    ktas = _heuristic(train, gs, 2, 4).evaluator.kta_many(states)
    assert len(ref) == len(ktas), (len(ref), len(ktas))
    gap = float(np.abs(ref - ktas).max())
    assert gap < 1e-9, f"E15 diverge de E13 de {gap:.2e}"
    print(f"  recoupement E13 (sine 2q, profondeur 4) : écart max {gap:.2e}")


def _distinct(states) -> np.ndarray:
    seen: dict = {}
    for i, state in enumerate(states):
        seen.setdefault(state, i)
    return np.array(sorted(seen.values()))


def _shot_subset(kta: np.ndarray, n_take: int, rng) -> np.ndarray:
    if n_take >= len(kta):
        return np.arange(len(kta))
    order = np.argsort(kta)[::-1]
    head = order[:n_take // 2]
    rest = rng.choice(order[n_take // 2:], size=n_take - len(head), replace=False)
    return np.sort(np.concatenate([head, rest]))


def _run_contrast() -> tuple[list[dict], list[dict]]:
    contrast_rows: list[dict] = []
    shot_rows: list[dict] = []
    checked = False
    for n in QUBITS:
        depth = n
        for dataset in DATASETS:
            for seed in SEEDS:
                for scale in SCALES:
                    train, _ = _split(dataset, n, seed, scale)
                    gs = _gate_set(n)
                    ev = _heuristic(train, gs, n, depth).evaluator
                    rng = np.random.default_rng((n, seed, DATASETS.index(dataset)))

                    groups = [("sample", _sample_states(gs, depth, N_SAMPLE, rng))]
                    if n <= EXHAUSTIVE_MAX_N:
                        groups.append(("exhaustive", _enumerate(gs, depth)))

                    kta_by_source = {}
                    for source, states in groups:
                        uppers = _uppers(ev, states)
                        ktas = _kta_batch(uppers, train.y)
                        kta_by_source[source] = ktas
                        if not checked:
                            _check_kta_paths(ev, states[:5], train.y, ktas[:5])
                            _check_binomial(uppers[0], 100)
                            checked = True
                        best = states[int(np.argmax(ktas))]
                        contrast_rows.append({
                            "n_qubits": n, "dataset": dataset, "seed": seed,
                            "scale": scale, "scale_over_pi": scale / np.pi,
                            "depth": depth, "source": source,
                            "n_circuits": len(states),
                            "catalog_size": len(gs.catalog),
                            "space_size": common.space_size(n, depth),
                            "kta_max": float(ktas.max()),
                            "kta_median": float(np.median(ktas)),
                            "kta_sd": float(ktas.std()),
                            "kta_iqr": float(np.subtract(*np.percentile(ktas, [75, 25]))),
                            "kta_gap": float(ktas.max() - np.median(ktas)),
                            "contrast": float((ktas.max() - np.median(ktas))
                                              / ktas.std()) if ktas.std() > 0 else 0.0,
                            "var_offdiag_median": float(np.median(uppers.var(axis=1))),
                            "n_gates_best": _n_gates(best),
                            "n_entanglers_best": _n_entanglers(best),
                        })
                        if source == "sample" and np.isclose(scale, REFERENCE):
                            keep = _distinct(states)
                            shot_rows += _shots_for(
                                n, dataset, seed, uppers[keep], ktas[keep],
                                train.y, rng)

                    if "exhaustive" in kta_by_source:
                        gap = (kta_by_source["sample"].max()
                               - kta_by_source["exhaustive"].max())
                        assert gap < 1e-12, (
                            f"n={n} {dataset}: le max échantillonné dépasse "
                            f"l'exhaustif de {gap:.2e}")
                print(f"  n={n} {dataset:10s} seed {seed} fait")
    return contrast_rows, shot_rows


def _shots_for(n, dataset, seed, uppers, ktas, y, rng) -> list[dict]:
    from scipy.stats import spearmanr

    take = _shot_subset(ktas, N_SHOT_CIRCUITS, rng)
    sub_upper, sub_kta = uppers[take], ktas[take]
    exact_top = set(np.argsort(sub_kta)[::-1][:TOP_K].tolist())
    i_best = int(np.argmax(sub_kta))
    rows = []
    for shots in SHOTS:
        for rep in range(N_REPS):
            hat = _kta_batch(_sample_upper(rng, sub_upper, shots), y)
            order = np.argsort(hat)[::-1]
            rows.append({
                "n_qubits": n, "dataset": dataset, "seed": seed,
                "shots": shots, "rep": rep, "n_circuits": len(take),
                "spearman_rho": float(spearmanr(sub_kta, hat).statistic),
                "top10_overlap": len(exact_top & set(order[:TOP_K].tolist())) / TOP_K,
                "rank_of_true_best": int(np.flatnonzero(order == i_best)[0]) + 1,
                "kta_bias_median": float(np.median(hat - sub_kta)),
            })
    return rows


def _run_search() -> list[dict]:
    from sklearn.metrics.pairwise import linear_kernel, rbf_kernel

    from qksearch.search import MctsSearch, search as run_search

    rows: list[dict] = []
    for n in SEARCH_QUBITS:
        depth = n
        for dataset in DATASETS:
            for seed in SEEDS:
                train, test = _split(dataset, n, seed, REFERENCE)
                gs = _gate_set(n)
                heuristic = _heuristic(train, gs, n, depth)
                m = len(gs.catalog)
                folds = common.cv_folds(len(train.y), seed=seed)

                t0 = time.time()
                res = run_search((), heuristic,
                                 MctsSearch(n_iterations=max(1, EVAL_BUDGET // m),
                                            seed=seed))
                wall = time.time() - t0
                rows.append(_search_row(
                    n, dataset, seed, "mcts", res.n_evaluated, wall,
                    heuristic, res.best_state, train, test))

                t0 = time.time()
                rng = np.random.default_rng((n, seed, 999))
                states = _sample_states(gs, depth, res.n_evaluated, rng)
                ktas = _kta_batch(_uppers(heuristic.evaluator, states), train.y)
                best = states[int(np.argmax(ktas))]
                rows.append(_search_row(
                    n, dataset, seed, "random-best", len(states),
                    time.time() - t0, heuristic, best, train, test))

                gamma = _gamma_by_cv(train, folds)
                for arm, (K_tr, K_te) in {
                    "rbf-cv": (rbf_kernel(train.X, gamma=gamma),
                               rbf_kernel(test.X, train.X, gamma=gamma)),
                    "linear": (linear_kernel(train.X),
                               linear_kernel(test.X, train.X)),
                }.items():
                    rows.append({
                        "n_qubits": n, "dataset": dataset, "seed": seed,
                        "arm": arm, "n_evaluated": 0, "wall_s": 0.0,
                        "kta": float("nan"), "n_gates": 0, "n_entanglers": 0,
                        "r2_test": _r2(K_tr, K_te, train, test),
                        "detail": f"gamma={gamma:g}" if arm == "rbf-cv" else "",
                    })
                mcts_r2 = rows[-4]["r2_test"]
                print(f"  n={n} {dataset:10s} seed {seed} : "
                      f"R2 mcts {mcts_r2:+.3f}, "
                      f"aléatoire {rows[-3]['r2_test']:+.3f}, "
                      f"RBF {rows[-2]['r2_test']:+.3f}")
    return rows


def _search_row(n, dataset, seed, arm, n_evaluated, wall, heuristic, state,
                train, test) -> dict:
    return {
        "n_qubits": n, "dataset": dataset, "seed": seed, "arm": arm,
        "n_evaluated": int(n_evaluated), "wall_s": round(wall, 2),
        "kta": float(heuristic.kta(state)),
        "n_gates": _n_gates(state), "n_entanglers": _n_entanglers(state),
        "r2_test": _quantum_r2(heuristic, state, train, test),
        "detail": "",
    }


def run() -> None:
    t0 = time.time()
    print("garde-fous :")
    _check_idle_qubits()
    _check_arms_agree()
    _check_sampler_uniform()
    if not SMOKE:
        _check_against_e13()

    print("A+B — contraste entre circuits et coût en shots :")
    contrast_rows, shot_rows = _run_contrast()
    common.write_csv(CSV_CONTRAST, contrast_rows)
    common.write_csv(CSV_SHOTS, shot_rows)

    print("C — la recherche paie-t-elle encore :")
    search_rows = _run_search()
    common.write_csv(CSV_SEARCH, search_rows)

    common.write_meta("e15_scaling", time.time() - t0, qubits=QUBITS,
                      search_qubits=SEARCH_QUBITS, seeds=SEEDS,
                      n_sample=N_SAMPLE, scales=list(SCALES),
                      shots=list(SHOTS), n_reps=N_REPS,
                      n_shot_circuits=N_SHOT_CIRCUITS,
                      eval_budget=EVAL_BUDGET, datasets=list(DATASETS),
                      depth_rule="d = n", exhaustive_max_n=EXHAUSTIVE_MAX_N,
                      rho_target=RHO_TARGET, smoke=SMOKE)
    print(f"E15 terminée en {time.time() - t0:.1f}s")


def _ramp(k: int, total: int) -> str:
    base = np.array([0x2a, 0x78, 0xd6]) / 255.0
    t = 0.42 + 0.58 * (k / max(total - 1, 1))
    rgb = base * t + (1.0 - t)
    return "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in rgb)


ARMS = {"mcts": ("MCTS", BLUE_ := common.BLUE, "-", "o"),
        "random-best": ("tirage uniforme, même coût", common.RED, "--", "s"),
        "rbf-cv": ("RBF, $\\gamma$ par CV", common.MUTED, ":", "^"),
        "linear": ("noyau linéaire", common.BASELINE, "-.", "")}
DATASET_LABEL = {"sine-fixed": "cible figée ($n-2$ features de nuisance)",
                 "sine-full": "cible sur les $n$ features"}


def _median_band(ax, x, groups, color, ls, marker, label, alpha=0.15):
    med = groups.median().to_numpy()
    if alpha:
        ax.fill_between(x, groups.quantile(0.25).to_numpy(),
                        groups.quantile(0.75).to_numpy(),
                        color=color, alpha=alpha, lw=0)
    ax.plot(x, med, color=color, lw=1.6, ls=ls, marker=marker, ms=3.5,
            label=label)
    return med


def plot() -> None:
    import matplotlib.pyplot as plt

    common.report_style()
    contrast = common.read_csv(CSV_CONTRAST)
    shots = common.read_csv(CSV_SHOTS)
    search = common.read_csv(CSV_SEARCH)
    sample = contrast[contrast["source"] == "sample"]
    ref = sample[np.isclose(sample["scale_over_pi"], REFERENCE / np.pi)]
    ns = np.array(sorted(sample["n_qubits"].unique()))
    hard = "sine-full"

    fig, axes = plt.subplots(2, 3, figsize=(9.9, 5.4))

    ax = axes[0][0]
    scales = sorted(sample["scale_over_pi"].unique())
    for k, sc in enumerate(scales):
        g = sample[(sample["dataset"] == hard)
                   & np.isclose(sample["scale_over_pi"], sc)]
        y = g.groupby("n_qubits")["contrast"].mean().reindex(ns).to_numpy()
        color = _ramp(k, len(scales))
        ax.plot(ns, y, color=color, lw=1.8, marker="o", ms=3.5)
        ax.annotate(f"$\\pi/{int(round(1 / sc))}$" if sc < 1 else "$\\pi$",
                    (ns[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=7, color=common.INK2, va="center")
    ax.set_xlabel("nombre de qubits $n$ (profondeur $d = n$)")
    ax.set_ylabel(r"contraste $(\mathrm{KTA}_{\max}-\mathrm{med})/\mathrm{sd}$")
    ax.set_title("Le contraste tient — la bande passante\nen fixe le niveau",
                 fontsize=8.5)
    ax.set_xticks(ns)
    ax.set_xlim(ns[0] - 0.3, ns[-1] + 0.9)
    ax.grid(True, alpha=0.4)

    ax = axes[0][1]
    series = [("var_offdiag_median", r"$\mathrm{Var}[K_{ij}]$ (la mesure d'E1)",
               common.MUTED, ":", "s"),
              ("kta_gap", r"écart absolu $\mathrm{KTA}_{\max}-\mathrm{med}$",
               common.RED, "--", "^"),
              ("contrast", "contraste normalisé", common.BLUE, "-", "o")]
    for column, label, color, ls, marker in series:
        y = ref[ref["dataset"] == hard].groupby("n_qubits")[column].mean()
        y = (y.reindex(ns) / y.loc[ns[0]]).to_numpy()
        ax.plot(ns, y, color=color, lw=1.6, ls=ls, marker=marker, ms=3.5)
        ax.annotate(label, (ns[-1], y[-1]), xytext=(5, 0),
                    textcoords="offset points", fontsize=6.5, color=color,
                    va="center", ha="left")
    ax.axhline(1.0, color=common.BASELINE, lw=0.7, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("nombre de qubits $n$")
    ax.set_ylabel("valeur rapportée à $n = 2$")
    ax.set_title("Ce qu'E1 mesure baisse, ce que la\nrecherche consomme non",
                 fontsize=8.5)
    ax.set_xticks(ns)
    ax.set_xlim(ns[0] - 0.3, ns[-1] + 4.6)
    ax.grid(True, alpha=0.4)

    ax = axes[0][2]
    sub = shots[shots["dataset"] == hard]
    ms_grid = np.array(sorted(sub["shots"].unique()))
    for k, n in enumerate(ns):
        y = (sub[sub["n_qubits"] == n].groupby("shots")["spearman_rho"]
             .mean().reindex(ms_grid).to_numpy())
        ax.plot(ms_grid, y, color=_ramp(k, len(ns)), lw=1.5,
                marker="o", ms=2.8)
        if n in (ns[0], ns[-1]):
            dy = 9 if n == ns[0] else -3
            ax.annotate(f"$n={n}$", (ms_grid[0], y[0]), xytext=(6, dy),
                        textcoords="offset points", fontsize=7,
                        color=common.INK2, va="center", ha="left")
    ax.axhline(0.9, color=common.RED, lw=1.0, ls="--")
    ax.text(0.98, 0.885, r"$\rho = 0{,}9$", transform=ax.transAxes,
            fontsize=6.5, color=common.RED, va="top", ha="right")
    ax.set_xscale("log")
    ax.set_xlim(0.55, 2.2e4)
    ax.set_xlabel("shots par entrée du Gram $M$")
    ax.set_ylabel(r"$\rho$ de Spearman avec le classement exact")
    ax.set_title("Les shots ne cassent pas :\n$M^\\star \\leq 30$ jusqu'à $n = 8$",
                 fontsize=8.5)
    ax.grid(True, alpha=0.4)

    ax = axes[1][0]
    for dataset, dash in (("sine-fixed", "-"), ("sine-full", "--")):
        for arm in ("mcts", "random-best"):
            g = search[(search["dataset"] == dataset) & (search["arm"] == arm)]
            label, color, _, marker = ARMS[arm]
            xs = np.array(sorted(g["n_qubits"].unique()))
            ax.plot(xs, g.groupby("n_qubits")["kta"].median().reindex(xs).to_numpy(),
                    color=color, lw=1.6, ls=dash, marker=marker, ms=3.5,
                    mfc="white" if dash == "--" else color,
                    label=f"{label} — {'cible figée' if dataset == 'sine-fixed' else 'cible sur $n$'}")
    ax.set_xlabel("nombre de qubits $n$")
    ax.set_ylabel("KTA du circuit élu")
    ax.set_title("La recherche décroche : MCTS passe\nsous le tirage uniforme",
                 fontsize=8.5)
    ax.set_xticks(sorted(search["n_qubits"].unique()))
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=5.6, loc="center right")

    lo = float(search["r2_test"].groupby(
        [search["dataset"], search["n_qubits"], search["arm"]]).median().min())
    for ax, dataset in ((axes[1][1], "sine-fixed"), (axes[1][2], "sine-full")):
        for arm, (label, color, ls, marker) in ARMS.items():
            g = search[(search["dataset"] == dataset) & (search["arm"] == arm)]
            xs = np.array(sorted(g["n_qubits"].unique()))
            _median_band(ax, xs, g.groupby("n_qubits")["r2_test"], color, ls,
                         marker, label, alpha=0.12 if marker else 0.0)
        ax.axhline(0.0, color=common.BASELINE, lw=0.8, ls=":")
        ax.set_ylim(min(lo * 1.1, -0.6), 1.7)
        ax.set_xlabel("nombre de qubits $n$")
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.4)
        ax.set_title(f"$R^2$ test — {DATASET_LABEL[dataset]}", fontsize=8.5)
    axes[1][1].set_ylabel("$R^2$ test (médiane sur 5 graines)")
    axes[1][2].tick_params(labelleft=False)
    axes[1][2].legend(fontsize=6, loc="upper right")

    fig.suptitle("E15 — passage à l'échelle en qubits : ce qui casse en premier "
                 "n'est ni le contraste, ni les shots, mais la recherche",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    common.save_fig(fig, "fig_e15_scaling")
    plt.close(fig)


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot()
