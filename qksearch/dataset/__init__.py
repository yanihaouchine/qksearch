"""Datasets: the MOF adsorption table, synthetic benchmarks and the feature-to-angle scaler."""

from __future__ import annotations

from .base import BaseDataset, Dataset, RegressionDataset
from .mof import mof_dataset
from .moons import moons_dataset
from .scaling import AngleScaler
from .sine import sine_dataset, sine_target
from .toy import toy_dataset

__all__ = [
    "AngleScaler",
    "BaseDataset",
    "Dataset",
    "RegressionDataset",
    "mof_dataset",
    "moons_dataset",
    "sine_dataset",
    "sine_target",
    "toy_dataset",
]
