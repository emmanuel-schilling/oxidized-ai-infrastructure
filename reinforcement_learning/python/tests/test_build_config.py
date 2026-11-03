"""Checks on how the oxidized_rl extension is built."""

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_the_extension_is_built_with_a_profile_that_unwinds_panics() -> None:
    # With panic = "abort", a Rust panic would kill the Python process; with
    # "unwind", PyO3 raises it as a Python exception instead.
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    cargo = tomllib.loads((REPO / "Cargo.toml").read_text(encoding="utf-8"))
    profile = pyproject["tool"]["maturin"]["profile"]
    assert cargo["profile"][profile]["panic"] == "unwind"
    assert cargo["profile"][profile]["inherits"] == "release"
