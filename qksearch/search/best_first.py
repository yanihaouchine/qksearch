from __future__ import annotations

import heapq
import itertools

from .base import E, S, SearchStrategy


class BestFirstSearch(SearchStrategy[S, E]):
    def search(self, initial, heuristic, *, budget=None):
        self._reset()
        counter = itertools.count()
        score = self._score(heuristic, initial)
        frontier = [(-score, next(counter), initial)]
        self._record(heuristic, initial, score)

        while frontier and self._budget_left(budget):
            neg, _, state = heapq.heappop(frontier)
            if heuristic.is_complete(state):
                continue
            children = [heuristic.extend(state, e)
                        for e in self._expand(heuristic, state)]
            for child, s in zip(children, self._score_many(heuristic, children)):
                self._record(heuristic, child, s)
                heapq.heappush(frontier, (-s, next(counter), child))
        return self._result()
