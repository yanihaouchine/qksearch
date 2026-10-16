from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, Protocol, Sequence, TypeVar, runtime_checkable

S = TypeVar("S")
E = TypeVar("E")

@runtime_checkable
class Heuristic(Protocol[S, E]):
    def candidates(self, state: S) -> Sequence[E]:
        ...

    def extend(self, state: S, element: E) -> S:
        ...

    def score(self, state: S) -> float:
        ...

    def is_complete(self, state: S) -> bool:
        ...


@dataclass(frozen=True)
class SearchResult(Generic[S]):
    best_state: Optional[S]
    best_score: float
    n_expanded: int
    n_evaluated: int


class SearchStrategy(ABC, Generic[S, E]):
    record_complete_only: bool = False

    path_following: bool = False

    def __init__(self, record_complete_only: bool = False) -> None:
        self.record_complete_only = record_complete_only

    @abstractmethod
    def search(self, initial: S, heuristic: Heuristic[S, E], *,budget: Optional[int] = None, ) -> SearchResult[S]:
        ...

    def _reset(self) -> None:
        self._n_expanded = 0
        self._n_evaluated = 0
        self._best_state: Optional[S] = None
        self._best_score = float("-inf")

    def _score(self, heuristic: Heuristic[S, E], state: S) -> float:
        self._n_evaluated += 1
        return heuristic.score(state)

    def _score_many(self, heuristic: Heuristic[S, E], states: Sequence[S]) -> list[float]:
        batch = getattr(heuristic, "scores", None)
        if batch is None:
            return [self._score(heuristic, s) for s in states]
        self._n_evaluated += len(states)
        return [float(v) for v in batch(states)]

    def _expand(self, heuristic: Heuristic[S, E], state: S) -> Sequence[E]:
        self._n_expanded += 1
        if self.path_following:
            free = getattr(heuristic, "free_candidates", None)
            if free is not None:
                return free(state)
        return heuristic.candidates(state)

    def _budget_left(self, budget: Optional[int]) -> bool:
        return budget is None or self._n_expanded < budget

    def _record(self, heuristic: Heuristic[S, E], state: S, score: float) -> None:
        if self.record_complete_only and not heuristic.is_complete(state):
            return
        if score > self._best_score:
            self._best_state = state
            self._best_score = score

    def _result(self) -> SearchResult[S]:
        return SearchResult(self._best_state, self._best_score,
                            self._n_expanded, self._n_evaluated)


def search(initial: S, heuristic: Heuristic[S, E], strategy: SearchStrategy[S, E], *, budget: Optional[int] = None, ) -> SearchResult[S]:
    return strategy.search(initial, heuristic, budget=budget)
