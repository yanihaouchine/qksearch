from __future__ import annotations

from .base import E, S, SearchStrategy


class GreedySearch(SearchStrategy[S, E]):
    path_following = True

    def __init__(self, allow_worsening: bool = True,
                 record_complete_only: bool = False) -> None:
        super().__init__(record_complete_only)
        self.allow_worsening = allow_worsening

    def search(self, initial, heuristic, *, budget=None):
        self._reset()
        state = initial
        current = self._score(heuristic, state)
        self._record(heuristic, state, current)

        while not heuristic.is_complete(state) and self._budget_left(budget):
            elements = self._expand(heuristic, state)
            if not elements:
                break
            children = [heuristic.extend(state, e) for e in elements]
            scores = self._score_many(heuristic, children)
            i = max(range(len(children)), key=scores.__getitem__)
            best, best_state = scores[i], children[i]
            if not self.allow_worsening and best <= current:
                break
            state, current = best_state, best
            self._record(heuristic, state, current)
        return self._result()
