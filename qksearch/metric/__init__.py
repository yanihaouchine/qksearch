"""Scores of a Gram matrix against the training targets (higher is better).

Every metric is a callable ``metric(gram, y) -> float``; most also offer
``batch(grams, y)`` for a stack of Gram matrices of shape ``(b, n, n)``.
"""

from __future__ import annotations

from .base import Metric
from .centered_alignment import CenteredAlignment
from .ridge_loo import RidgeLooMse
from .target_alignment import TargetAlignment
from .weighted_sum import GramTerm, WeightedSum
from .zero import ZeroMetric

__all__ = [
    "CenteredAlignment",
    "GramTerm",
    "Metric",
    "RidgeLooMse",
    "TargetAlignment",
    "WeightedSum",
    "ZeroMetric",
]
