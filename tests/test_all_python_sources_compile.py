from __future__ import annotations

import py_compile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _iter_python_sources() -> list[Path]:
    roots = [
        REPO_ROOT / "scripts",
        REPO_ROOT / "send_data",
        REPO_ROOT / "exp_future",
        REPO_ROOT / "evaluation_accident",
    ]
    top_level = [
        REPO_ROOT / "analyze_sweep.py",
        REPO_ROOT / "check_runs.py",
    ]

    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    for path in top_level:
        if path.exists():
            files.append(path)

    # Unique + stable order for deterministic test output.
    return sorted(set(files), key=lambda p: p.as_posix())


def test_python_source_list_is_not_empty() -> None:
    sources = _iter_python_sources()
    assert sources, "No Python source files found for compile smoke test."


def test_all_python_sources_compile() -> None:
    failures: list[str] = []
    for source in _iter_python_sources():
        try:
            py_compile.compile(str(source), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(f"{source}: {exc.msg}")

    assert not failures, "Compile failures:\n" + "\n".join(failures)
