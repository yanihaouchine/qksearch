from __future__ import annotations

import numpy as np


class ZeroMetric:
    def __call__(self, gram: np.ndarray, y: np.ndarray) -> float:
        return 0.0
