"""Combinatorial search for quantum-kernel feature maps.

Circuits are grown gate by gate by a search strategy (:mod:`qksearch.search`)
and scored, without any training, by a metric on their Gram matrix
(:mod:`qksearch.metric`). :func:`build_heuristic` sets up the IQP search used
throughout the experiments.
"""

from qksearch.backend import NumpyBackend
from qksearch.dataset import AngleScaler, RegressionDataset, mof_dataset, sine_dataset
from qksearch.iqp import build_heuristic, describe, iqp_circuit
from qksearch.metric import RidgeLooMse, TargetAlignment, WeightedSum
from qksearch.search import BestFirstSearch, GreedySearch, MctsSearch, search

__version__ = "1.0.0"

__all__ = [
    "AngleScaler",
    "BestFirstSearch",
    "GreedySearch",
    "MctsSearch",
    "NumpyBackend",
    "RegressionDataset",
    "RidgeLooMse",
    "TargetAlignment",
    "WeightedSum",
    "build_heuristic",
    "describe",
    "iqp_circuit",
    "mof_dataset",
    "search",
    "sine_dataset",
]
