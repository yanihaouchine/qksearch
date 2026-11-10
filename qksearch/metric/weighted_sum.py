from __future__ import annotations

from typing import Mapping

import numpy as np

from .base import Metric


class GramTerm:
    def __init__(self, feature) -> None:
        self.feature = feature

    def __call__(self, gram: np.ndarray, y: np.ndarray) -> float:
        return float(self.feature(gram))

    def batch(self, grams: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(self.feature.batch(grams), dtype=float)


class WeightedSum:
    def __init__(self, terms: Mapping[str, tuple[Metric, float]]) -> None:
        if not terms:
            raise ValueError("terms must not be empty")
        self.terms = dict(terms)

    def components(self, grams: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        return {name: weight * np.asarray(metric.batch(grams, y), dtype=float)
                for name, (metric, weight) in self.terms.items()}

    def __call__(self, gram: np.ndarray, y: np.ndarray) -> float:
        return float(sum(weight * metric(gram, y)
                         for metric, weight in self.terms.values()))

    def batch(self, grams: np.ndarray, y: np.ndarray) -> np.ndarray:
        return sum(self.components(grams, y).values())
