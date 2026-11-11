from __future__ import annotations

import importlib
import sys

import pytest

from experiments import common


def test_producers_are_resolved():
    assert common._producer("e14_rank.csv") == "e14_shots"
    assert common._producer("e5_benchmark.csv") == "e5_benchmark"
    assert common._producer("e16_mse_space_d6.npz") == "e16_mse_heuristic"
    assert common._producer("population_mof.csv") == "population_circuits"
    assert common._producer("notes.csv") is None


def test_unknown_missing_file_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RESULTS_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="no experiment produces it"):
        common.result_path("notes.csv")


def test_e14_runs_from_empty_results(tmp_path, monkeypatch):
    monkeypatch.setenv("E14_SMOKE", "1")
    monkeypatch.setattr(common, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(common, "REPORT_FIG_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["e14", "--no-show"])
    sys.modules.pop("experiments.e14_shots", None)
    e14 = importlib.import_module("experiments.e14_shots")
    try:
        e14.plot()
    finally:
        sys.modules.pop("experiments.e14_shots", None)
    assert (tmp_path / "e14_rank_smoke.csv").exists()


@pytest.mark.parametrize("module", ["population_circuits", "population_metrics"])
def test_scripts_accept_no_show(module, monkeypatch):
    monkeypatch.setattr(sys, "argv", [module, "--no-show"])
    assert importlib.import_module(f"experiments.{module}").parse_args().no_show
