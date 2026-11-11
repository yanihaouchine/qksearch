from __future__ import annotations

import runpy
from pathlib import Path

QUICKSTART = Path(__file__).resolve().parents[1] / "examples" / "quickstart.py"


def test_quickstart_runs(capsys):
    runpy.run_path(str(QUICKSTART), run_name="__main__")
    out = capsys.readouterr().out
    assert out.count("test R²") == 3
