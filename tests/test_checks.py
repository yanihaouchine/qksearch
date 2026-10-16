from __future__ import annotations

import sys

import pytest

from tests import check_mof_havlicek, check_x0_vs_x1


def _run(main, argv: list[str], monkeypatch) -> int:
    monkeypatch.setattr(sys, "argv", ["check", *argv])
    return main()


def test_mof_havlicek_runs(monkeypatch):
    assert _run(check_mof_havlicek.main, ["--no-fig"], monkeypatch) == 0


@pytest.mark.slow
def test_x0_vs_x1_runs(monkeypatch):
    assert _run(check_x0_vs_x1.main, ["--no-fig"], monkeypatch) == 0
