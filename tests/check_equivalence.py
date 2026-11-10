from __future__ import annotations

import random
from itertools import permutations

import numpy as np


from qksearch.backend import NumpyBackend, QiskitBackend
from qksearch.backend.base import _gram_from_states
from qksearch.dataset import (
    AngleScaler,
    Dataset,
    RegressionDataset,
    moons_dataset,
    sine_dataset,
    toy_dataset,
)
from qksearch.dla import DlaCalculator, is_abelian, state_generators
from qksearch.expressibility import (
    expressibility,
    haar_bin_probabilities,
    kl_divergence,
    n_features_of,
)
from qksearch.gates import (
    IDENTITY_OP,
    RANDOM_ORDER,
    GateSpec,
    ProductGateSet,
    iqp_gate_set,
    pauli_gate_set,
    phase_power_gate_set,
)
from qksearch.iqp import (
    DEFAULT_WEIGHTS,
    IqpKernelHeuristic,
    build_heuristic,
    default_metric,
    hadamard_layer,
    iqp_circuit,
)
from qksearch.kernel import CircuitEvaluator
from qksearch.metric import CenteredAlignment, TargetAlignment
from qksearch.search import (
    BacktrackingSearch,
    BestFirstSearch,
    GreedySearch,
    MctsSearch,
    ProbabilisticSearch,
    search,
)

TOL = 1e-12


def _catalog(n: int) -> list:
    ops = list(iqp_gate_set(n, n).placements(()))
    ops += list(pauli_gate_set(n, n).placements(()))
    ops += [("h", (q,), None) for q in range(n)] + [IDENTITY_OP]
    if n >= 2:
        ops += list(phase_power_gate_set(n).placements(()))
        ops += [("cx", (0, 1), None), ("cx", (1, 0), None), ("cz", (0, 1), None)]
    return ops


def _random_states(cat, count, max_len=9, seed=0):
    rng = random.Random(seed)
    return [tuple(rng.choice(cat) for _ in range(rng.randint(1, max_len)))
            for _ in range(count)]


def _train_data(n_samples=40, n_features=2):
    data = (moons_dataset(n_samples=n_samples + 20, noise=0.15, seed=0)
            if n_features == 2
            else toy_dataset(n_samples=n_samples + 20, n_features=n_features, seed=0))
    train = Dataset(data.X[:n_samples], data.y[:n_samples])
    return AngleScaler.fit(train.X).apply(train)


def _regression_data(n_samples=40, n_features=2):
    data = sine_dataset(n_samples=n_samples + 20, n_features=n_features,
                        noise=0.05, seed=0)
    train = RegressionDataset(data.X[:n_samples], data.y[:n_samples])
    return AngleScaler.fit(train.X).apply(train)


def _random_grams(m=40, count=6, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        A = rng.normal(size=(m, m))
        K = A @ A.T
        d = np.sqrt(np.diag(K))
        out.append((K / np.outer(d, d)) ** 2)
    return np.stack(out)


def test_numpy_backend_matches_qiskit() -> None:
    rng = np.random.default_rng(0)
    worst = 0.0
    for n in (1, 2, 3, 4):
        X = rng.normal(size=(7, n))
        nb, qb = NumpyBackend(n), QiskitBackend(n)
        for state in _random_states(_catalog(n), 40, seed=n):
            worst = max(
                worst,
                np.abs(nb.gram(state, X) - qb.gram(state, X)).max(),
                np.abs(nb.cross_gram(state, X[:4], X[4:])
                       - qb.cross_gram(state, X[:4], X[4:])).max(),
                np.abs(nb.states(state, X) - np.asarray(qb._states(state, X))).max(),
            )
    assert worst < TOL, f"NumPy/Qiskit discrepancy = {worst}"


def test_vectorized_gram() -> None:
    rng = np.random.default_rng(1)
    vecs = list(rng.normal(size=(9, 8)) + 1j * rng.normal(size=(9, 8)))

    def loop(vs):
        K = np.eye(len(vs))
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                K[i, j] = K[j, i] = abs(np.vdot(vs[i], vs[j])) ** 2
        return K

    K_ref, K_new = loop(vecs), _gram_from_states(vecs)
    off = ~np.eye(len(vecs), dtype=bool)
    assert np.abs(K_ref[off] - K_new[off]).max() < TOL
    unit = [v / np.linalg.norm(v) for v in vecs]
    assert np.abs(loop(unit) - _gram_from_states(unit)).max() < TOL


def test_diagonal_gates_commute() -> None:
    n, be = 3, NumpyBackend(3)
    X = np.random.default_rng(2).normal(size=(6, n))
    cat = list(iqp_gate_set(n, n).placements(()))
    base = [cat[0], cat[2], cat[3], cat[5]]
    ref = be.gram(iqp_circuit(tuple(base), n), X)
    for order in permutations(base):
        assert np.abs(be.gram(iqp_circuit(order, n), X) - ref).max() < TOL
    a = cat[0]
    assert np.abs(be.gram(iqp_circuit((a, a), n), X)
                  - be.gram(iqp_circuit((a,), n), X)).max() > 1e-3


def test_canonicalization_loses_no_circuit() -> None:
    n, depth = 2, 5
    be = NumpyBackend(n)
    X = np.random.default_rng(3).normal(size=(6, n))

    def walk(gs):
        states = [()]
        for _ in range(depth):
            states = [s + (op,) for s in states for op in gs.placements(s)]
        return states

    def gram_keys(states):
        return {np.round(be.gram(iqp_circuit(s, n), X), 10).tobytes() for s in states}

    canon = walk(iqp_gate_set(n, n, canonical=True))
    plain = walk(iqp_gate_set(n, n, canonical=False))
    assert not gram_keys(plain) - gram_keys(canon), "canonicalization loses circuits"
    assert len(canon) == len(set(canon)) and len(canon) < len(plain) / 2
    from math import comb
    assert len(canon) == comb(3 + depth - 1, depth)


def _canonical_walk(gs, depth: int):
    states = [()]
    for _ in range(depth):
        states = [s + (op,) for s in states for op in gs.placements(s)]
    return states


def test_order_seed_permutes_without_touching_the_space() -> None:
    n, depth = 2, 4
    be = NumpyBackend(n)
    X = np.random.default_rng(6).normal(size=(6, n))

    def grams(gs):
        return {np.round(be.gram(iqp_circuit(s, n), X), 10).tobytes()
                for s in _canonical_walk(gs, depth)}

    reference = grams(iqp_gate_set(n, n))
    seen_reorder = False
    for seed in range(20):
        gs = iqp_gate_set(n, n, order_seed=seed)
        assert gs.order_seed == seed
        assert sorted(gs.catalog) == sorted(iqp_gate_set(n, n).catalog)
        assert len(_canonical_walk(gs, depth)) == len(reference)
        assert grams(gs) == reference, seed
        seen_reorder |= gs.catalog != iqp_gate_set(n, n).catalog
    assert seen_reorder, "no seed reordered the catalog: the test proves nothing"


def test_random_order_is_redrawn_at_each_launch() -> None:
    n = 4
    draws = [iqp_gate_set(n, n, order_seed=RANDOM_ORDER) for _ in range(8)]
    assert all(isinstance(gs.order_seed, int) for gs in draws)
    assert len({gs.order_seed for gs in draws}) > 1, "the draw is not random"
    for gs in draws:
        assert iqp_gate_set(n, n, order_seed=gs.order_seed).catalog == gs.catalog
    assert iqp_gate_set(n, n).order_seed is None, "the default must stay pinned"


def test_canonical_insert_is_order_independent() -> None:
    n = 3
    gs = iqp_gate_set(n, n, order_seed=1)
    cat = list(gs.catalog)
    picks = [cat[0], cat[2], cat[2], cat[-1]]
    states = set()
    for order in permutations(picks):
        state = ()
        for op in order:
            state = gs.insert(state, op)
        states.add(state)
    assert len(states) == 1, f"{len(states)} representatives for one multiset"
    (canonical,) = states
    assert sorted(canonical, key=gs._rank.__getitem__) == list(canonical)
    assert sorted(canonical) == sorted(picks), "insert lost or invented a gate"

    plain = iqp_gate_set(n, n, canonical=False)
    assert plain.insert((cat[0],), cat[1]) == (cat[0], cat[1]), "no reordering here"


def test_free_branching_unblocks_the_greedy_descent() -> None:
    n, depth = 3, 5
    train = _train_data(n_features=n)

    def heuristic(order_seed):
        gs = ProductGateSet(
            [GateSpec("rz", 1, uses_feature=True),
             GateSpec("rzz", 2, uses_feature=True, symmetric=True)],
            n_qubits=n, n_features=n, with_identity=True, order_seed=order_seed)
        return IqpKernelHeuristic(
            gate_set=gs, backend=NumpyBackend(n), data=train,
            metric=TargetAlignment(), max_depth=depth, n_qubits=n), gs

    h, gs = heuristic(None)
    opt = float(h.evaluator.kta_many(_canonical_walk(gs, depth)).max())

    class Restricted(GreedySearch):
        path_following = False

    seeds = [None, 0, 1, 2, 3, 4, 5]
    free = [search((), heuristic(s)[0], GreedySearch()).best_score for s in seeds]
    restricted = [search((), heuristic(s)[0], Restricted()).best_score for s in seeds]

    assert all(abs(v - opt) < 1e-9 for v in free), (min(free), opt)
    assert max(free) - min(free) < 1e-9, (
        "free branching still depends on the catalog order")
    assert min(restricted) < opt - 1e-3, "restricted branching lost nothing here"
    assert len(set(restricted)) > 1, "the order bias did not show: test is vacuous"


def test_mapping_feature_qubit() -> None:
    from math import comb

    for n in (2, 3, 4):
        cat = list(iqp_gate_set(n, n).placements(()))
        assert len(cat) == n + comb(n, 2), (n, len(cat))
        assert len(set(cat)) == len(cat)
        for name, qubits, f in cat:
            assert f == qubits, (name, qubits, f)

    for n_qubits, n_features in ((2, 3), (3, 0)):
        try:
            iqp_gate_set(n_qubits, n_features)
        except ValueError as exc:
            assert "n_features" in str(exc), exc
        else:
            raise AssertionError(f"({n_qubits}, {n_features}) accepted")
    phase_power_gate_set(3)

    X = np.random.default_rng(5).normal(size=(6, 2))
    nb, qb = NumpyBackend(2), QiskitBackend(2)
    rzz = ("rzz", (0, 1), (0, 1))
    product = np.column_stack([X[:, 0] * X[:, 1], np.zeros(len(X))])
    assert np.abs(nb.gram(iqp_circuit((rzz,), 2), X)
                  - qb.gram(iqp_circuit((("rzz", (0, 1), (0,)),), 2), product)).max() < TOL


def test_extra_qubits_ignored() -> None:
    n_features = 2
    narrow = iqp_gate_set(n_features, n_features)
    base = list(narrow.placements(()))
    for n_qubits in (3, 5):
        large = iqp_gate_set(n_qubits, n_features)
        assert large.n_active == n_features and large.n_qubits == n_qubits
        assert list(large.placements(())) == base, n_qubits
        assert (list(large.placements((base[-1],)))
                == list(narrow.placements((base[-1],))))

    train = _train_data(n_features=n_features)
    ref = build_heuristic(train, n_qubits=n_features, n_data_gates=6,
                          backend=NumpyBackend(n_features))
    states_ref = _random_walk(ref, count=12)
    for n_qubits in (3, 5):
        h = build_heuristic(train, n_qubits=n_qubits, n_data_gates=6,
                            backend=NumpyBackend(n_qubits))
        for s in states_ref:
            assert np.abs(h.evaluator.gram(s) - ref.evaluator.gram(s)).max() < TOL
            assert abs(h.score(s) - ref.score(s)) < TOL, (n_qubits, s)


def _pennylane_operator(gen):
    import pennylane as qml

    paulis = {"X": qml.X, "Y": qml.Y, "Z": qml.Z}
    op = paulis[gen[0][0]](gen[0][1])
    for axis, qubit in gen[1:]:
        op = op @ paulis[axis](qubit)
    return op


def test_abelian_dla_matches_lie_closure() -> None:
    import pennylane as qml

    dla = DlaCalculator()
    seen = {True: 0, False: 0}
    for n in (2, 3, 4):
        for gs in (iqp_gate_set(n, n), pauli_gate_set(n, n)):
            for state in _random_states(list(gs.placements(())), 12, 7, seed=n):
                full = hadamard_layer(n) + state
                gens = state_generators(full)
                seen[is_abelian(gens)] += 1
                ref = len(qml.lie_closure([_pennylane_operator(g) for g in gens]))
                assert dla.dimension(full) == ref, full
    assert seen[True] and seen[False], "one of the two paths was never exercised"


def _random_walk(heuristic, count=40, max_len=6, seed=4):
    rng = random.Random(seed)
    states = []
    for _ in range(count):
        s = ()
        for _ in range(rng.randint(1, max_len)):
            cands = heuristic.candidates(s)
            if not cands:
                break
            s += (rng.choice(list(cands)),)
        states.append(s)
    rng.shuffle(states)
    return states


def test_incremental_evaluator() -> None:
    n = 3
    train = _train_data(n_features=n)
    for regularization in ("dla", "variance"):
        ref = build_heuristic(train, n_qubits=n, n_data_gates=6,
                              regularization=regularization)
        fast = build_heuristic(train, n_qubits=n, n_data_gates=6,
                               regularization=regularization, backend=NumpyBackend(n))
        states = _random_walk(fast)
        for s in states:
            assert abs(ref.kta(s) - fast.kta(s)) < TOL
            assert abs(ref.score(s) - fast.score(s)) < TOL

    small = CircuitEvaluator(NumpyBackend(n), train, TargetAlignment(),
                             prefix=fast.prefix, suffix=fast.suffix, psi_cache_size=3)
    for s in states:
        assert abs(small.kta(s) - ref.kta(s)) < TOL


def test_gram_variance_value() -> None:
    n = 3
    c = 0.35
    train = _train_data(n_features=n)
    h = build_heuristic(train, n_qubits=n, n_data_gates=6, regularization="variance",
                        weight=c, backend=NumpyBackend(n))
    iu = np.triu_indices(train.X.shape[0], k=1)
    seen_nonzero = False
    for s in _random_walk(h, count=12):
        expected = 4.0 * c * h.evaluator.gram(s)[iu].var()
        assert abs(h.score(s) - h.kta(s) - expected) < TOL
        seen_nonzero |= expected > TOL
    assert seen_nonzero, "the bonus stayed zero everywhere"


def test_gram_variance_batched() -> None:
    n = 3
    train = _train_data(n_features=n)
    mk = lambda: build_heuristic(train, n_qubits=n, n_data_gates=6,
                                 regularization="variance", backend=NumpyBackend(n))
    un, lot = mk(), mk()
    states = [(op,) for op in un.candidates(())]
    states += [states[0] + (op,) for op in un.candidates(states[0])]
    assert np.abs(lot.scores(states)
                  - np.array([un.score(s) for s in states])).max() < TOL


def test_regularization_modes() -> None:
    n = 3
    train = _train_data(n_features=n)
    hs = {mode: build_heuristic(train, n_qubits=n, n_data_gates=6,
                                regularization=mode, weight=DEFAULT_WEIGHTS[mode],
                                backend=NumpyBackend(n))
          for mode in ("dla", "variance", "none")}
    seen = {"dla": False, "variance": False}
    for s in _random_walk(hs["none"], count=12):
        kta = hs["none"].kta(s)
        assert hs["none"].score(s) == kta
        assert hs["dla"].score(s) <= kta
        assert hs["variance"].score(s) >= kta
        seen["dla"] |= hs["dla"].score(s) < kta
        seen["variance"] |= hs["variance"].score(s) > kta
    assert all(seen.values()), f"one of the two terms stayed zero everywhere: {seen}"


def test_batched_scores() -> None:
    n = 3
    train = _train_data(n_features=n)
    un = build_heuristic(train, n_qubits=n, n_data_gates=6, backend=NumpyBackend(n))
    lot = build_heuristic(train, n_qubits=n, n_data_gates=6, backend=NumpyBackend(n))
    states = [(op,) for op in un.candidates(())]
    states += [states[0] + (op,) for op in un.candidates(states[0])]
    assert np.abs(lot.scores(states)
                  - np.array([un.score(s) for s in states])).max() < TOL


def test_centered_alignment_offset_invariant() -> None:
    train = _regression_data()
    centered, kta = CenteredAlignment(), TargetAlignment()
    seen_move = False
    for K in _random_grams(m=train.n_samples):
        assert abs(centered(K, train.y) - centered(K + 3.0, train.y)) < TOL
        seen_move |= abs(kta(K, train.y) - kta(K + 3.0, train.y)) > 1e-3
    assert seen_move, "the uncentered KTA did not move: the test proves nothing"


def test_centered_alignment_bounds_and_self_alignment() -> None:
    train = _regression_data()
    centered = CenteredAlignment()
    for K in _random_grams(m=train.n_samples):
        value = centered(K, train.y)
        assert 0.0 <= value <= 1.0, value
    assert abs(centered(centered.target_kernel(train.y), train.y) - 1.0) < TOL


def test_centered_alignment_constant_kernel() -> None:
    train = _regression_data()
    centered = CenteredAlignment()
    flat = np.ones((train.n_samples, train.n_samples))
    assert centered(flat, train.y) == 0.0
    batched = centered.batch(np.stack([flat, 0.5 * flat]), train.y)
    assert not np.isnan(batched).any()
    assert (batched == 0.0).all(), batched


def test_centered_alignment_batched() -> None:
    train = _regression_data()
    grams = _random_grams(m=train.n_samples)
    centered = CenteredAlignment()
    one_by_one = np.array([centered(K, train.y) for K in grams])
    assert np.abs(centered.batch(grams, train.y) - one_by_one).max() < TOL


def test_explicit_metric_overrides_default() -> None:
    train = _regression_data()
    assert isinstance(default_metric(_train_data()), TargetAlignment)
    assert isinstance(default_metric(train), TargetAlignment)
    h = build_heuristic(train, n_qubits=2, n_data_gates=4, backend=NumpyBackend(2))
    assert isinstance(h.evaluator.metric, TargetAlignment)
    h = build_heuristic(train, n_qubits=2, n_data_gates=4, backend=NumpyBackend(2),
                        metric=CenteredAlignment())
    assert isinstance(h.evaluator.metric, CenteredAlignment)


def test_regression_search_bounds_its_scores() -> None:
    train = _regression_data()
    mk = lambda: build_heuristic(train, n_qubits=2, n_data_gates=4,
                                 regularization="none", backend=NumpyBackend(2),
                                 metric=CenteredAlignment())
    un, lot = mk(), mk()
    states = [(op,) for op in un.candidates(())]
    states += [states[0] + (op,) for op in un.candidates(states[0])]
    values = np.array([un.kta(s) for s in states])
    assert ((values >= 0.0) & (values <= 1.0)).all(), (values.min(), values.max())
    assert np.abs(lot.scores(states) - values).max() < TOL


class KnapsackHeuristic:
    def __init__(self, items, capacity):
        self.items = items
        self.capacity = capacity

    def _weight(self, state):
        return sum(self.items[i][2] for i in state)

    def candidates(self, state):
        remaining = self.capacity - self._weight(state)
        return [i for i in range(len(self.items))
                if i not in state and self.items[i][2] <= remaining]

    def extend(self, state, element):
        return state | {element}

    def score(self, state):
        return float(sum(self.items[i][1] for i in state))

    def is_complete(self, state):
        return not self.candidates(state)


def test_strategies_on_the_knapsack() -> None:
    items = [("ingot", 10.0, 10.0), ("watch", 7.0, 5.0),
             ("ring", 7.0, 5.0), ("coin", 3.0, 1.0)]
    h = KnapsackHeuristic(items, capacity=11.0)
    assert search(frozenset(), h, GreedySearch()).best_score == 13.0
    assert search(frozenset(), h, BacktrackingSearch()).best_score == 17.0
    assert search(frozenset(), h, BestFirstSearch()).best_score >= 13.0
    assert search(frozenset(), h, MctsSearch(n_iterations=400, seed=0)).best_score == 17.0


def test_mcts_does_not_degrade_the_search() -> None:
    n = 3
    train = _train_data(n_features=n)
    mk = lambda: build_heuristic(train, n_qubits=n, n_data_gates=6,
                                 backend=NumpyBackend(n))
    greedy = search((), mk(), GreedySearch())
    mcts = search((), mk(), MctsSearch(n_iterations=1500, seed=0))
    assert mcts.best_score >= greedy.best_score, (mcts.best_score, greedy.best_score)
    assert mcts.best_state is not None and len(mcts.best_state) <= 6


class _ToyTree:
    def __init__(self, table, branching, depth):
        self.table = table
        self.branching = branching
        self.depth = depth

    def candidates(self, state):
        return [] if len(state) >= self.depth else list(range(self.branching))

    def extend(self, state, element):
        return state + (element,)

    def score(self, state):
        return float(self.table.get(state, 0.0))

    def is_complete(self, state):
        return len(state) >= self.depth


def _draw_frequencies(strategy, heuristic, n_draws) -> list[float]:
    strategy._reset()
    counts = [0] * heuristic.branching
    children = [heuristic.extend((), e) for e in strategy._expand(heuristic, ())]
    for _ in range(n_draws):
        weights = strategy._weights(strategy._aggregate(heuristic, children))
        counts[strategy.rng.choices(range(len(children)), weights=weights)[0]] += 1
    return [c / n_draws for c in counts]


def test_probabilistic_matches_the_target_distribution() -> None:
    h = _ToyTree({(0,): 1.0, (1,): 2.0, (2,): 3.0, (3,): 4.0}, branching=4, depth=1)
    freq = _draw_frequencies(ProbabilisticSearch(seed=0), h, 20000)
    for got, want in zip(freq, [0.1, 0.2, 0.3, 0.4]):
        assert abs(got - want) < 0.02, f"P = {freq}, expected [.1,.2,.3,.4]"


def test_probabilistic_shifts_negative_heuristics() -> None:
    h = _ToyTree({(0,): -2.0, (1,): -1.0, (2,): 0.0, (3,): 1.0}, branching=4, depth=1)
    freq = _draw_frequencies(ProbabilisticSearch(seed=0), h, 20000)
    total = 0.15 + 1.15 + 2.15 + 3.15
    for got, want in zip(freq, [x / total for x in (0.15, 1.15, 2.15, 3.15)]):
        assert abs(got - want) < 0.02, f"P = {freq}"
    assert freq[0] > 0.0, "the worst child was never drawn: the shift is inoperative"


def test_probabilistic_uniform_on_ties() -> None:
    for value in (5.0, -3.0, 0.0):
        table = {(i,): value for i in range(3)}
        h = _ToyTree(table, branching=3, depth=1)
        freq = _draw_frequencies(ProbabilisticSearch(seed=0), h, 6000)
        for got in freq:
            assert abs(got - 1.0 / 3.0) < 0.03, f"ties at {value} -> P = {freq}"


def test_probabilistic_lookahead_changes_the_ranking() -> None:
    table = {(0,): 10.0, (1,): 6.0,
             (0, 0): 0.0, (0, 1): 0.0, (1, 0): 20.0, (1, 1): 20.0}
    h = _ToyTree(table, branching=2, depth=2)
    shallow = ProbabilisticSearch(lookahead=1, seed=0)
    deep = ProbabilisticSearch(lookahead=2, gamma=0.5, seed=0)
    shallow._reset()
    deep._reset()
    children = [h.extend((), e) for e in shallow._expand(h, ())]

    assert shallow._aggregate(h, children) == [10.0, 6.0]
    deep_values = deep._aggregate(h, children)
    assert abs(deep_values[0] - 20.0 / 3.0) < TOL, deep_values
    assert abs(deep_values[1] - 32.0 / 3.0) < TOL, deep_values
    assert deep_values[1] > deep_values[0], "lookahead did not reverse the ranking"


def test_probabilistic_is_deterministic_under_seed() -> None:
    n = 3
    train = _train_data(n_features=n)
    mk = lambda: build_heuristic(train, n_qubits=n, n_data_gates=5,
                                 backend=NumpyBackend(n))
    mk_run = lambda seed: search((), mk(), ProbabilisticSearch(n_descents=8, seed=seed))
    a, b, c = mk_run(0), mk_run(0), mk_run(1)
    assert a.best_state == b.best_state and a.best_score == b.best_score
    assert (a.best_state, a.best_score) != (c.best_state, c.best_score), (
        "two different seeds gave the same run: the draw is not random")


def test_probabilistic_on_the_knapsack() -> None:
    items = [("ingot", 10.0, 10.0), ("watch", 7.0, 5.0),
             ("ring", 7.0, 5.0), ("coin", 3.0, 1.0)]
    h = KnapsackHeuristic(items, capacity=11.0)
    result = search(frozenset(), h, ProbabilisticSearch(n_descents=200, seed=0))
    assert 13.0 <= result.best_score <= 17.0, result.best_score
    capped = search(frozenset(), h, ProbabilisticSearch(n_descents=200, seed=0),
                    budget=25)
    assert capped.n_expanded <= 25, capped.n_expanded


def test_angle_scaler_centers_and_bounds() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(loc=[3.0, -50.0], scale=[0.01, 200.0], size=(200, 2))
    scaler = AngleScaler.fit(X)
    A = scaler.transform(X)
    assert np.abs(A.mean(axis=0)).max() < 0.05, A.mean(axis=0)
    assert np.abs(A.std(axis=0) - np.pi / 4).max() < 0.05, A.std(axis=0)
    assert np.abs(A).max() <= np.pi / 2 + TOL, np.abs(A).max()


def test_angle_scaler_inverse() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(loc=[1.0, -2.0], scale=[3.0, 0.5], size=(60, 2))
    scaler = AngleScaler.fit(X)

    A = scaler.transform(X)
    inside = np.abs(A).max(axis=1) < np.pi / 2 - 1e-9
    assert inside.any(), "no unclipped point: the test checks nothing"
    assert np.abs(scaler.inverse_transform(A[inside]) - X[inside]).max() < 1e-9

    lin = np.linspace(-np.pi / 2, np.pi / 2, 7)
    gx, gy = np.meshgrid(lin, lin)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    assert np.abs(scaler.transform(scaler.inverse_transform(pts)) - pts).max() < 1e-9


def test_angle_scaler_fitted_on_train() -> None:
    rng = np.random.default_rng(2)
    train_X = rng.normal(size=(40, 2))
    scaler = AngleScaler.fit(train_X)
    reference = scaler.transform(train_X)

    for offset in (0.0, 100.0):
        mixed = np.vstack([train_X, rng.normal(size=(80, 2)) + offset])
        assert np.abs(scaler.transform(mixed)[:40] - reference).max() < TOL

    fitted_on_all = AngleScaler.fit(mixed).transform(mixed)[:40]
    assert np.abs(fitted_on_all - reference).max() > 0.1, "the leak would go unnoticed"


def test_split_default_scale_unchanged() -> None:
    from qksearch.dataset import mof_dataset
    from experiments import common

    raw = mof_dataset(seed=0)
    reference = AngleScaler.fit(raw.X[:100], scale=np.pi / 4,
                                bound=np.pi / 2).apply(raw)
    train, test = common.mof_split(0)
    assert np.array_equal(train.X, reference.X[:100])
    assert np.array_equal(test.X, reference.X[100:])

    raw = sine_dataset(n_samples=120, n_features=2, noise=0.05, seed=0)
    reference = AngleScaler.fit(raw.X[:40], scale=np.pi / 4,
                                bound=np.pi / 2).apply(raw)
    train, test = common.sine_split(0)
    assert np.array_equal(train.X, reference.X[:40])
    assert np.array_equal(test.X, reference.X[40:])


def test_split_bound_follows_scale() -> None:
    from experiments import common

    quantiles = None
    for scale in (np.pi / 32, np.pi / 4, np.pi):
        train, _ = common.mof_split(0, scale=scale)
        bound = common.BOUND_SIGMAS * scale
        assert np.abs(train.X).max() <= bound + TOL, (scale, np.abs(train.X).max())
        clipped = np.abs(np.abs(train.X) - bound) < TOL
        rescaled = train.X / scale
        if quantiles is None:
            quantiles, reference_clipped = rescaled, clipped
        else:
            assert np.array_equal(clipped, reference_clipped), scale
            assert np.abs(rescaled - quantiles).max() < TOL, scale


def test_cv_r2_matches_manual_folds() -> None:
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.metrics import r2_score
    from sklearn.metrics.pairwise import rbf_kernel

    from experiments import common

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = X[:, 0] + 0.1 * rng.normal(size=40)
    K = rbf_kernel(X, gamma=0.5)

    folds = common.cv_folds(len(y), seed=0)
    manual = []
    for tr, va in folds:
        krr = KernelRidge(alpha=common.RIDGE_ALPHA, kernel="precomputed")
        krr.fit(K[np.ix_(tr, tr)], y[tr])
        manual.append(r2_score(y[va], krr.predict(K[np.ix_(va, tr)])))
    assert abs(common.cv_r2(K, y, folds) - float(np.mean(manual))) < 1e-12
    assert common.cv_folds(len(y), seed=1)[0][1].tolist() != folds[0][1].tolist()


def test_angle_scaler_constant_column() -> None:
    X = np.column_stack([np.arange(10.0), np.full(10, 2.5)])
    try:
        AngleScaler.fit(X)
    except ValueError as exc:
        assert "1" in str(exc), exc
    else:
        raise AssertionError("constant column accepted")


def _haar_states(dim: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=(count, dim)) + 1j * rng.normal(size=(count, dim))
    return psi / np.linalg.norm(psi, axis=1, keepdims=True)


def test_haar_bins_sum_to_one() -> None:
    edges = np.linspace(0.0, 1.0, 76)
    for n_qubits in range(1, 5):
        q = haar_bin_probabilities(edges, 1 << n_qubits)
        assert (q >= 0.0).all()
        assert abs(float(q.sum()) - 1.0) < TOL, (n_qubits, q.sum())
    q = haar_bin_probabilities(edges, 8)
    assert (np.diff(q) < 0.0).all(), "Haar mass is not decreasing"


def test_kl_divergence_convention() -> None:
    p = np.array([0.5, 0.5, 0.0])
    assert abs(kl_divergence(p, p)) < TOL
    q = np.array([0.25, 0.25, 0.5])
    assert kl_divergence(p, q) > 0.0
    assert abs(kl_divergence(p, q) - np.log(2.0)) < 1e-12


def test_expressibility_haar_states() -> None:
    n_qubits, n_pairs, n_bins = 3, 20000, 75
    dim = 1 << n_qubits
    a = _haar_states(dim, n_pairs, seed=0)
    b = _haar_states(dim, n_pairs, seed=1)
    F = np.abs(np.einsum("ij,ij->i", a.conj(), b)) ** 2

    counts, edges = np.histogram(F, bins=n_bins, range=(0.0, 1.0))
    kl = kl_divergence(counts / n_pairs, haar_bin_probabilities(edges, dim))
    assert kl < 0.05, f"KL against the Haar law is too large: {kl}"


def test_n_features_of() -> None:
    assert n_features_of(()) == 0
    assert n_features_of(hadamard_layer(2)) == 0
    assert n_features_of((("rzz", (0, 1), (0, 2)),)) == 3


def test_expressibility_trivial_circuit() -> None:
    backend = NumpyBackend(2)
    diagonal = (("rz", (0,), (0,)), ("rzz", (0, 1), (0, 1)))

    bare = expressibility(backend, diagonal, n_features=2, n_pairs=2000, seed=0)
    full_circ = expressibility(backend, iqp_circuit(diagonal, 2),
                             n_features=2, n_pairs=2000, seed=0)
    assert np.isfinite(bare) and np.isfinite(full_circ)
    assert bare > full_circ, (bare, full_circ)
    assert expressibility(backend, (), n_features=2, n_pairs=2000) == bare


def test_expressibility_orders_circuits() -> None:
    backend = NumpyBackend(2)
    kwargs = dict(n_features=2, n_pairs=20000, seed=0)
    plain = expressibility(backend, iqp_circuit((), 2), **kwargs)
    entangling = expressibility(
        backend, iqp_circuit((("rz", (0,), (0,)), ("rz", (1,), (1,)),
                              ("rzz", (0, 1), (0, 1))), 2), **kwargs)
    assert entangling < plain, f"the entangling gate does not help: {entangling} vs {plain}"
    assert abs(entangling - plain) > 0.1, "the two circuits are not told apart"


def test_expressibility_deterministic() -> None:
    backend = NumpyBackend(2)
    circuit = iqp_circuit((("rz", (0,), (0,)), ("rzz", (0, 1), (0, 1))), 2)
    a = expressibility(backend, circuit, n_features=2, n_pairs=8000, seed=0)
    assert a == expressibility(backend, circuit, n_features=2, n_pairs=8000, seed=0)
    b = expressibility(backend, circuit, n_features=2, n_pairs=8000, seed=7)
    assert a != b, "two seeds give the same sample"
    assert abs(a - b) < 0.1 * max(a, b), f"unstable estimator: {a} vs {b}"


def test_havlicek_encoding() -> None:
    from math import comb

    from qksearch.backend.base import feature_angle, feature_angles
    from qksearch.gates import ANGLE_ENCODING

    rng = np.random.default_rng(3)
    X = rng.uniform(-np.pi / 2, np.pi / 2, size=(9, 3))
    expected = (np.pi - X[:, 0]) * (np.pi - X[:, 2])
    got = feature_angles(X, (0, 2), "rzzh")
    assert np.abs(got - expected).max() < TOL, np.abs(got - expected).max()
    for name in ("rzz", None):
        raw = feature_angles(X, (0, 2), name)
        assert np.abs(raw - X[:, 0] * X[:, 2]).max() < TOL, name
    assert abs(feature_angle(X[0], (0, 2), "rzzh") - expected[0]) < TOL

    scaled = feature_angles(X, (0, 2), "rzzhn")
    assert np.abs(scaled - expected / (2 * np.pi)).max() < TOL
    full = np.array([[-np.pi / 2, 0.0, -np.pi / 2], [np.pi / 2, 0.0, np.pi / 2]])
    sweep = np.ptp(feature_angles(full, (0, 2), "rzzhn"))
    assert abs(sweep - np.pi) < 1e-9, sweep

    dla = DlaCalculator()
    for pair in ((0, 1), (1, 2)):
        a = (("rz", (0,), (0,)), ("rzz", pair, pair))
        b = (("rz", (0,), (0,)), ("rzzh", pair, pair))
        assert state_generators(a) == state_generators(b), pair
        assert dla.dimension(a) == dla.dimension(b), pair

    for n in (2, 3):
        cat = list(iqp_gate_set(n, n, extra_entanglers=("rzzh", "rzzhn")).placements(()))
        assert len(cat) == n + 3 * comb(n, 2), (n, len(cat))
        Xn = rng.uniform(-np.pi / 2, np.pi / 2, size=(6, n))
        nb, qb = NumpyBackend(n), QiskitBackend(n)
        for name, scale in (("rzzh", 1.0), ("rzzhn", 1 / (2 * np.pi))):
            state = iqp_circuit((("rz", (0,), (0,)), (name, (0, 1), (0, 1))), n)
            a = (np.pi - Xn[:, 0]) * (np.pi - Xn[:, 1]) * scale
            closed = (np.cos((Xn[:, None, 0] - Xn[None, :, 0]) / 2) ** 2
                      * np.cos((a[:, None] - a[None, :]) / 2) ** 2)
            assert np.abs(nb.gram(state, Xn) - closed).max() < TOL, (n, name)
            assert np.abs(nb.gram(state, Xn) - qb.gram(state, Xn)).max() < TOL, (n, name)

    assert ANGLE_ENCODING.get("rzz") is None, "rzz must keep the raw product"
    try:
        iqp_gate_set(2, 2, extra_entanglers=("nope",))
    except ValueError as exc:
        assert "nope" in str(exc), exc
    else:
        raise AssertionError("unknown entangler accepted")


def _genome_specs():
    from qksearch.genome import iqp_genome_spec

    return (iqp_genome_spec(4, 4, n_layers=1),
            iqp_genome_spec(4, 4, n_layers=2, phases=("t", "cs")))


def test_genome_bijection() -> None:
    from qksearch.genome import from_int, random_genome, to_int

    mono, spec = _genome_specs()
    assert len(mono.genes) == 10 and mono.n_bits == 30, mono.describe()
    assert len(spec.genes) == 30 and spec.n_bits == 84, spec.describe()

    pyrng, rng = random.Random(0), np.random.default_rng(0)
    for _ in range(2000):
        v = pyrng.getrandbits(spec.n_bits)
        assert to_int(from_int(v, spec), spec) == v, v
    seen = set()
    for _ in range(2000):
        g = random_genome(spec, rng, max_support=6)
        v = to_int(g, spec)
        assert 0 <= v < (1 << spec.n_bits)
        assert from_int(v, spec) == g
        seen.add((v, g))
    assert len({v for v, _ in seen}) == len({g for _, g in seen})


def test_genome_decode_is_canonical() -> None:
    from qksearch.genome import decode, encode, random_genome
    from qksearch.iqp import hadamard_layer

    _mono, spec = _genome_specs()
    rng = np.random.default_rng(1)
    h = hadamard_layer(spec.n_qubits)
    for _ in range(500):
        g = random_genome(spec, rng, max_support=6)
        state = decode(g, spec)
        assert encode(state, spec) == g, g
        layers, current, i = [], [], 0
        while i < len(state):
            if tuple(state[i:i + len(h)]) == h:
                layers.append(current)
                current, i = [], i + len(h)
            else:
                current.append(state[i])
                i += 1
        layers.append(current)
        assert len(layers) == spec.n_layers, (len(layers), spec.n_layers)

        for layer, ops in enumerate(layers):
            genes = spec.genes[spec.layer_slice(layer)]
            by_op = {gene.op: j for j, gene in enumerate(genes)}
            by_qubits = {gene.op[1]: j for j, gene in enumerate(genes)
                         if gene.is_phase}
            seq = [by_op.get(op, by_qubits.get(op[1])) for op in ops]
            assert None not in seq, (layer, ops)
            assert seq == sorted(seq), (layer, seq)


def test_phase_folding_matches_named_gate() -> None:
    from qksearch.gates import PHASE_POWERS
    from qksearch.genome import PHASE_BY_POWER, phase_order

    assert phase_order("t") == 8 and phase_order("cs") == 4
    backend = NumpyBackend(3)
    rng = np.random.default_rng(2)
    X = rng.uniform(-np.pi / 2, np.pi / 2, size=(6, 3))
    for name, qubits in (("t", (0,)), ("cs", (0, 1))):
        _arity, base = PHASE_POWERS[name]
        for k in range(1, phase_order(name)):
            folded = PHASE_BY_POWER[len(qubits)][(base * k) % 8]
            a = backend.states(((name, qubits, None),) * k, X)
            b = backend.states(((folded, qubits, None),), X)
            assert np.abs(a - b).max() < TOL, (name, k, folded)


def test_phase_gates_are_inert_in_one_layer() -> None:
    from qksearch.iqp import hadamard_layer

    backend, h = NumpyBackend(4), hadamard_layer(4)
    rng = np.random.default_rng(3)
    X = rng.uniform(-np.pi / 2, np.pi / 2, size=(30, 4))
    base = (("rz", (0,), (0,)), ("rz", (2,), (2,)), ("rzz", (1, 2), (1, 2)))
    ref = backend.gram(h + base + h, X)
    for extra in ((("t", (0,), None),), (("cs", (1, 2), None),),
                  (("t", (0,), None), ("cs", (0, 3), None), ("tdg", (3,), None))):
        got = backend.gram(h + base + extra + h, X)
        assert np.abs(got - ref).max() < TOL, extra
        assert (state_generators(h + base + extra + h)
                == state_generators(h + base + h)), extra


def test_phase_gates_are_active_in_two_layers() -> None:
    from qksearch.iqp import hadamard_layer

    backend, h = NumpyBackend(4), hadamard_layer(4)
    rng = np.random.default_rng(4)
    X = rng.uniform(-np.pi / 2, np.pi / 2, size=(30, 4))
    d1 = (("rz", (0,), (0,)), ("rzz", (1, 2), (1, 2)))
    d2 = (("rz", (3,), (3,)), ("rzz", (0, 3), (0, 3)))
    ref = backend.gram(h + d1 + h + d2 + h, X)

    live = backend.gram(h + d1 + (("t", (0,), None),) + h + d2 + h, X)
    assert np.abs(live - ref).max() > 1e-2, "t(0) should reach the kernel"
    off = backend.gram(h + d1 + (("t", (1,), None),) + h + d2 + h, X)
    assert np.abs(off - ref).max() < TOL, "t(1) has no support in layer 2"
    last = backend.gram(h + d1 + h + d2 + (("t", (0,), None),) + h, X)
    assert np.abs(last - ref).max() < TOL, "last-layer phase must stay inert"


def test_one_layer_embeds_in_two() -> None:
    from qksearch.genome import decode, iqp_genome_spec, random_genome

    mono = iqp_genome_spec(4, 4, n_layers=1)
    duo = iqp_genome_spec(4, 4, n_layers=2, phases=())
    backend = NumpyBackend(4)
    rng = np.random.default_rng(5)
    X = rng.uniform(-np.pi / 2, np.pi / 2, size=(25, 4))
    from qksearch.iqp import hadamard_layer

    h = hadamard_layer(4)
    for _ in range(20):
        g = random_genome(mono, rng, max_support=6, max_weight=6)
        flat = g + (0,) * (len(duo.genes) - len(mono.genes))
        a = backend.gram(h + decode(g, mono) + h, X)
        b = backend.gram(h + decode(flat, duo) + h, X)
        assert np.abs(a - b).max() < TOL, g


def test_genome_covers_the_e4_space() -> None:
    from qksearch.genome import decode, enumerate_genomes, iqp_genome_spec

    spec = iqp_genome_spec(4, 4, n_layers=1)
    gs = ProductGateSet(
        [GateSpec("rz", 1, uses_feature=True),
         GateSpec("rzz", 2, uses_feature=True, symmetric=True)],
        n_qubits=4, n_features=4, with_identity=True)
    states = [()]
    for _ in range(6):
        states = [s + (op,) for s in states for op in gs.placements(s)]
    e4 = {tuple(op for op in s if op[0] != "id") for s in states}

    genomes = list(enumerate_genomes(spec, max_weight=6))
    assert len(genomes) == 8008, len(genomes)
    assert {decode(g, spec) for g in genomes} == e4, "not a bijection with E4"


def test_genome_repair_respects_bounds() -> None:
    from qksearch.genome import random_genome, repair, support, weight

    _mono, spec = _genome_specs()
    rng = np.random.default_rng(6)
    for _ in range(500):
        g = random_genome(spec, rng, max_support=5, max_weight=6)
        assert support(g) <= 5 and weight(g, spec) <= 6, g
        assert all(0 <= k <= gene.cap for gene, k in zip(spec.genes, g)), g
        full = repair(spec.caps, spec, rng, max_support=5, max_weight=6)
        assert support(full) <= 5 and weight(full, spec) <= 6, full


def test_genetic_recovers_the_exhaustive_optimum() -> None:
    from qksearch.genome import decode, enumerate_genomes, iqp_genome_spec
    from qksearch.metric import TargetAlignment
    from qksearch.search import search
    from qksearch.search.genetic import GeneticSearch, seed_population

    data = toy_dataset(n_samples=14, n_features=2, seed=0)
    spec = iqp_genome_spec(2, 2, n_layers=1)
    heuristic = IqpKernelHeuristic(
        gate_set=ProductGateSet(
            [GateSpec("rz", 1, uses_feature=True),
             GateSpec("rzz", 2, uses_feature=True, symmetric=True)],
            n_qubits=2, n_features=2, with_identity=True),
        backend=NumpyBackend(2), data=data, metric=TargetAlignment(),
        max_depth=4, n_qubits=2, regularizer=None)

    genomes = list(enumerate_genomes(spec, max_weight=4))
    opt = float(heuristic.scores([decode(g, spec) for g in genomes]).max())

    rng = np.random.default_rng(0)
    pop = seed_population(spec, 20, rng, max_support=3, max_weight=4)
    ga = GeneticSearch(spec, population=pop, n_generations=15, max_support=3,
                       max_weight=4, seed=0)
    result = search((), heuristic, ga)
    assert result.best_score >= opt - TOL, (result.best_score, opt)
    assert result.n_evaluated <= len(genomes), "the GA cost more than the sweep"


def test_genetic_is_deterministic_under_seed() -> None:
    from qksearch.genome import iqp_genome_spec
    from qksearch.metric import TargetAlignment
    from qksearch.search import search
    from qksearch.search.genetic import GeneticSearch, seed_population

    data = toy_dataset(n_samples=12, n_features=2, seed=0)
    spec = iqp_genome_spec(2, 2, n_layers=1)
    heuristic = IqpKernelHeuristic(
        gate_set=ProductGateSet(
            [GateSpec("rz", 1, uses_feature=True),
             GateSpec("rzz", 2, uses_feature=True, symmetric=True)],
            n_qubits=2, n_features=2, with_identity=True),
        backend=NumpyBackend(2), data=data, metric=TargetAlignment(),
        max_depth=4, n_qubits=2, regularizer=None)

    runs = []
    for _ in range(2):
        rng = np.random.default_rng(3)
        pop = seed_population(spec, 16, rng, max_support=3, max_weight=4)
        ga = GeneticSearch(spec, population=pop, n_generations=8,
                           max_support=3, max_weight=4, seed=3)
        result = search((), heuristic, ga)
        runs.append((result.best_score, ga.best_genome,
                     [s.best for s in ga.history], result.n_evaluated))
    assert runs[0] == runs[1], "same seed, different run"


def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:
            failures += 1
            print(f"  FAIL    {name}\n          {type(exc).__name__}: {exc}")
        else:
            print(f"  ok      {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
