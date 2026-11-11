"""Depth-first search with optional pruning."""

from __future__ import annotations

from typing import Callable, Optional

from .base import E, S, SearchStrategy


class BacktrackingSearch(SearchStrategy[S, E]):
    """Depth-first search that visits the most promising child first.

    Parameters
    ----------
    prune
        ``prune(state, score) -> bool``; pruned states are not expanded.
    """

    def __init__(self, prune: Optional[Callable[[S, float], bool]] = None,
                 record_complete_only: bool = False) -> None:
        super().__init__(record_complete_only)
        self.prune = prune

    def search(self, initial, heuristic, *, budget=None):
        self._reset()
        score = self._score(heuristic, initial)
        self._record(heuristic, initial, score)
        stack = [(initial, score)]

        while stack and self._budget_left(budget):
            state, score = stack.pop()
            if heuristic.is_complete(state):
                continue
            if self.prune is not None and self.prune(state, score):
                continue
            states = [heuristic.extend(state, e)
                      for e in self._expand(heuristic, state)]
            children = list(zip(states, self._score_many(heuristic, states)))
            for child, s in children:
                self._record(heuristic, child, s)
            children.sort(key=lambda p: p[1])
            stack.extend(children)
        return self._result()
