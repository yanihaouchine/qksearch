"""Constant metric, as a null baseline."""

from __future__ import annotations

import numpy as np


class ZeroMetric:
    """Always 0: the search is then driven by the regularizer alone."""

    def __call__(self, gram: np.ndarray, y: np.ndarray) -> float:
        return 0.0
