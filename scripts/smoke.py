#!/usr/bin/env python
"""Basic smoke test: ensure every first-party Python module imports without errors.

Run via `python scripts/smoke.py` or include in CI before deploy.
Stops with non-zero exit status if any module fails to import/compile so that
pipeline can block the release.
"""
from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path
import py_compile

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 1. Byte-compile every *.py file (fast syntax check) -------------------------
print("[smoke] byte-compiling source files …", flush=True)

fail_compile: list[Path] = []
for py_file in PROJECT_ROOT.rglob("*.py"):
    # Skip virtualenvs and git metadata if any accidentally inside project dir
    if any(part in ("venv", ".venv", ".git") for part in py_file.parts):
        continue
    try:
        py_compile.compile(py_file, doraise=True)
    except py_compile.PyCompileError as exc:
        fail_compile.append(py_file)
        print(f"[compile-fail] {py_file}: {exc}")

# 2. Import every package/module under first-party namespace ------------------
# Treat any directory with __init__.py as a package root
print("[smoke] importing packages …", flush=True)

fail_import: list[tuple[str, Exception]] = []

# Ensure project root is importable
sys.path.insert(0, str(PROJECT_ROOT))

for module_info in pkgutil.walk_packages(path=[str(PROJECT_ROOT)], prefix=""):
    name = module_info.name
    # Skip third-party vendored folders or tests if desired
    if name.startswith(("venv", "tests", "scripts")):
        continue
    try:
        importlib.import_module(name)
    except Exception as err:  # pylint: disable=broad-except
        fail_import.append((name, err))
        print(f"[import-fail] {name}: {err}")

if fail_compile or fail_import:
    print("\n[smoke] FAILURES:\n" + "\n".join(
        [str(f) for f in fail_compile] + [f"{n}: {e}" for n, e in fail_import]
    ))
    sys.exit(1)

print("[smoke] all modules compiled & imported successfully ✔")
