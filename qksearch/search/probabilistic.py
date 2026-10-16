from __future__ import annotations

import random
from typing import Optional, Sequence

from .base import E, S, SearchStrategy


class ProbabilisticSearch(SearchStrategy[S, E]):
    path_following = True

    def __init__(self, lookahead: int = 1, gamma: float = 0.5,
                 shift_fraction: float = 0.05,
                 max_nodes_per_level: Optional[int] = None,
                 n_descents: int = 100, seed: Optional[int] = None,
                 record_complete_only: bool = False) -> None:
        super().__init__(record_complete_only)
        if lookahead < 1:
            raise ValueError(f"lookahead must be >= 1, got {lookahead}")
        if gamma < 0.0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if shift_fraction <= 0.0:
            raise ValueError(f"shift_fraction must be > 0, got {shift_fraction}")
        if n_descents < 1:
            raise ValueError(f"n_descents must be >= 1, got {n_descents}")
        if max_nodes_per_level is not None and max_nodes_per_level < 1:
            raise ValueError(
                f"max_nodes_per_level must be >= 1, got {max_nodes_per_level}")
        self.lookahead = lookahead
        self.gamma = gamma
        self.shift_fraction = shift_fraction
        self.max_nodes_per_level = max_nodes_per_level
        self.n_descents = n_descents
        self.rng = random.Random(seed)

    def _weights(self, values: Sequence[float]) -> list[float]:
        lo = min(values)
        if lo > 0.0:
            return list(values)
        span = max(values) - lo
        shift = self.shift_fraction * span if span > 0.0 else 1.0
        return [v - lo + shift for v in values]

    def _aggregate(self, heuristic, states: Sequence[S]) -> list[float]:
        scores = self._score_many(heuristic, states)
        for state, value in zip(states, scores):
            self._record(heuristic, state, value)
        if self.lookahead == 1:
            return list(scores)

        totals = list(scores)
        norms = [1.0] * len(states)
        frontiers = [[state] for state in states]
        weight = 1.0

        for _ in range(1, self.lookahead):
            weight *= self.gamma
            batch: list[S] = []
            spans: list[tuple[int, int]] = []
            next_frontiers: list[list[S]] = []
            for frontier in frontiers:
                start = len(batch)
                children: list[S] = []
                for state in frontier:
                    if heuristic.is_complete(state):
                        continue
                    for element in self._expand(heuristic, state):
                        children.append(heuristic.extend(state, element))
                cap = self.max_nodes_per_level
                if cap is not None and len(children) > cap:
                    children = self.rng.sample(children, cap)
                batch.extend(children)
                spans.append((start, len(batch)))
                next_frontiers.append(children)

            if not batch:
                break
            values = self._score_many(heuristic, batch)
            for state, value in zip(batch, values):
                self._record(heuristic, state, value)
            for i, (start, stop) in enumerate(spans):
                if stop > start:
                    totals[i] += weight * sum(values[start:stop]) / (stop - start)
                    norms[i] += weight
            frontiers = next_frontiers

        return [total / norm for total, norm in zip(totals, norms)]

    def _descend(self, heuristic, initial, budget) -> bool:
        state = initial
        moved = False
        while not heuristic.is_complete(state) and self._budget_left(budget):
            elements = self._expand(heuristic, state)
            if not elements:
                break
            children = [heuristic.extend(state, e) for e in elements]
            weights = self._weights(self._aggregate(heuristic, children))
            index = self.rng.choices(range(len(children)), weights=weights)[0]
            state = children[index]
            moved = True
        return moved

    def search(self, initial, heuristic, *, budget=None):
        self._reset()
        root_value = self._score(heuristic, initial)
        self._record(heuristic, initial, root_value)

        for _ in range(self.n_descents):
            if not self._budget_left(budget):
                break
            if not self._descend(heuristic, initial, budget):
                break
        return self._result()
