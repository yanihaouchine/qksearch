from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Generic, Hashable, Optional, Sequence

from .base import E, S
from .probabilistic import ProbabilisticSearch


@dataclass(frozen=True)
class Individual(Generic[S]):
    state: S
    score: float


def multiset_distance(a: Sequence[Hashable], b: Sequence[Hashable]) -> int:
    ca, cb = Counter(a), Counter(b)
    return sum(((ca - cb) + (cb - ca)).values())


class PopulationSearch(ProbabilisticSearch[S, E]):
    def __init__(self, *, pad: Optional[Callable[[S], S]] = None,
                 complete_only: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pad = pad
        self.complete_only = complete_only
        self._pool: dict[S, float] = {}
        self.n_relaxed = 0

    @property
    def pool(self) -> dict[S, float]:
        return self._pool

    def _reset(self) -> None:
        super()._reset()
        self._pool = {}

    def _record(self, heuristic, state: S, score: float) -> None:
        super()._record(heuristic, state, score)
        if heuristic.is_complete(state):
            self._pool[state] = score
        elif self.pad is not None:
            self._pool[self.pad(state)] = score
        elif not self.complete_only:
            self._pool[state] = score

    def population(
        self,
        size: int,
        *,
        key: Optional[Callable[[S], Hashable]] = None,
        distance: Optional[Callable[[Hashable, Hashable], float]] = None,
        min_distance: float = 1.0,
    ) -> list[Individual[S]]:
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")
        self.n_relaxed = 0

        best: dict[Hashable, tuple[S, float]] = {}
        for state, score in self._pool.items():
            k = state if key is None else key(state)
            current = best.get(k)
            if current is None or score > current[1]:
                best[k] = (state, score)

        ranked = sorted(best.items(), key=lambda item: -item[1][1])

        selected: list[Individual[S]] = []
        chosen_keys: list[Hashable] = []
        deferred: list[tuple[S, float]] = []
        for k, (state, score) in ranked:
            if len(selected) >= size:
                break
            if (distance is not None and min_distance > 1.0
                    and any(distance(k, other) < min_distance
                            for other in chosen_keys)):
                deferred.append((state, score))
                continue
            selected.append(Individual(state, score))
            chosen_keys.append(k)

        for state, score in deferred:
            if len(selected) >= size:
                break
            selected.append(Individual(state, score))
            self.n_relaxed += 1

        selected.sort(key=lambda ind: -ind.score)
        return selected
