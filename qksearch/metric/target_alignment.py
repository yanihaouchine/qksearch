from __future__ import annotations

import numpy as np


class TargetAlignment:
    def __call__(self, gram: np.ndarray, y: np.ndarray) -> float:
        target = np.outer(y, y)
        denom = np.linalg.norm(gram) * np.linalg.norm(target)
        if denom == 0.0:
            return 0.0
        return float((gram * target).sum() / denom)

    def batch(self, grams: np.ndarray, y: np.ndarray) -> np.ndarray:
        target = np.outer(y, y)
        num = np.einsum("bij,ij->b", grams, target)
        denom = np.sqrt(np.einsum("bij,bij->b", grams, grams)) * np.linalg.norm(target)
        return np.divide(num, denom, out=np.zeros_like(num), where=denom != 0.0)
