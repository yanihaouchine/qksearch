from __future__ import annotations

from typing import Protocol

import numpy as np


class Metric(Protocol):
    def __call__(self, gram: np.ndarray, y: np.ndarray) -> float:
        ...
