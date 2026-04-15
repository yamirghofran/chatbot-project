
from __future__ import annotations

from pathlib import Path


def find_project_root() -> Path:
    """Find the project root by walking up until pyproject.toml is found.

    Returns:
        Absolute path to the directory containing pyproject.toml.

    Raises:
        FileNotFoundError: If pyproject.toml is not found in any parent directory.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("pyproject.toml not found in any parent directory")
