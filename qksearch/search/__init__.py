"""Search strategies over gate sequences, all driven by the same :class:`Heuristic` protocol."""

from __future__ import annotations

from .backtracking import BacktrackingSearch
from .base import E, Heuristic, S, SearchResult, SearchStrategy, search
from .best_first import BestFirstSearch
from .genetic import GenerationStats, GeneticSearch, seed_population
from .greedy import GreedySearch
from .mcts import MctsSearch
from .population import (
    Individual,
    PopulationSearch,
    multiset_distance,
)
from .probabilistic import ProbabilisticSearch

__all__ = [
    "E",
    "S",
    "Heuristic",
    "SearchResult",
    "SearchStrategy",
    "search",
    "GreedySearch",
    "GeneticSearch",
    "GenerationStats",
    "seed_population",
    "BestFirstSearch",
    "BacktrackingSearch",
    "MctsSearch",
    "ProbabilisticSearch",
    "PopulationSearch",
    "Individual",
    "multiset_distance",
]
