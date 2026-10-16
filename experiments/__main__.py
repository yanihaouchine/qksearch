from __future__ import annotations

import os
import runpy
import sys

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))


def available() -> dict[str, str]:
    modules = {}
    for entry in os.listdir(EXPERIMENTS_DIR):
        name, ext = os.path.splitext(entry)
        if ext != ".py" or "_" not in name or not name.startswith("e"):
            continue
        number = name.split("_", 1)[0][1:]
        if number.isdigit():
            modules[f"e{int(number)}"] = name
    return dict(sorted(modules.items(), key=lambda kv: int(kv[0][1:])))


def resolve(args: list[str]) -> tuple[str, list[str]]:
    modules = available()
    if not args:
        raise SystemExit(
            "usage: python -m experiments N [--plot-only]\n"
            f"expériences : {', '.join(modules)}")
    key = args[0] if args[0].startswith("e") else f"e{args[0]}"
    if key not in modules:
        raise SystemExit(f"expérience inconnue : {key!r} "
                         f"(disponibles : {', '.join(modules)})")
    return modules[key], args[1:]


def main(args: list[str]) -> None:
    module, rest = resolve(args)
    sys.argv = [os.path.join(EXPERIMENTS_DIR, f"{module}.py")] + rest
    try:
        runpy.run_module(f"experiments.{module}", run_name="__main__")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"dépendance manquante : {exc.name}. Installer le projet avec\n"
            f'    pip install -e .') from exc


if __name__ == "__main__":
    main(sys.argv[1:])
