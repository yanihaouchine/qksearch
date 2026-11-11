"""Generic tree-search interface shared by every strategy in :mod:`qksearch.search`.

A search problem is described by a :class:`Heuristic` (how to grow a partial
state and how to score it); a :class:`SearchStrategy` decides which states to
grow. Strategies only count expansions and evaluations, so they are agnostic
to what a state is: here, a tuple of quantum gates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, Protocol, Sequence, TypeVar, runtime_checkable

S = TypeVar("S")
E = TypeVar("E")

@runtime_checkable
class Heuristic(Protocol[S, E]):
    """Search problem: legal moves, transition, score and stopping rule."""

    def candidates(self, state: S) -> Sequence[E]:
        """Elements that may be added to ``state``; empty when it is a leaf."""
        ...

    def extend(self, state: S, element: E) -> S:
        """Return the child of ``state`` obtained by adding ``element``."""
        ...

    def score(self, state: S) -> float:
        """Objective to maximize."""
        ...

    def is_complete(self, state: S) -> bool:
        """Whether ``state`` is a leaf of the search tree."""
        ...


@dataclass(frozen=True)
class SearchResult(Generic[S]):
    """Best state found, its score, and the cost of the search.

    ``n_expanded`` counts states whose children were generated (this is what
    ``budget`` limits); ``n_evaluated`` counts calls to the score.
    """

    best_state: Optional[S]
    best_score: float
    n_expanded: int
    n_evaluated: int


class SearchStrategy(ABC, Generic[S, E]):
    """Base class of the search algorithms.

    Subclasses implement :meth:`search` with the protected helpers below,
    which keep the expansion/evaluation counters and the incumbent up to date.

    Parameters
    ----------
    record_complete_only
        Only leaves can become the incumbent (default: any visited state).

    Notes
    -----
    ``path_following`` asks the heuristic for *all* placements instead of the
    canonical ones: a greedy descent then cannot be trapped by the canonical
    gate order (see :meth:`qksearch.gates.ProductGateSet.free_placements`).
    """

    record_complete_only: bool = False

    path_following: bool = False

    def __init__(self, record_complete_only: bool = False) -> None:
        self.record_complete_only = record_complete_only

    @abstractmethod
    def search(self, initial: S, heuristic: Heuristic[S, E], *,
               budget: Optional[int] = None) -> SearchResult[S]:
        """Explore from ``initial``, expanding at most ``budget`` states."""
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


def search(initial: S, heuristic: Heuristic[S, E], strategy: SearchStrategy[S, E], *,
           budget: Optional[int] = None) -> SearchResult[S]:
    """Run ``strategy`` on ``heuristic`` from ``initial``."""
    return strategy.search(initial, heuristic, budget=budget)
