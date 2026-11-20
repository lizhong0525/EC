"""Content fingerprints for the actual implementation and resolved local runtime."""

import hashlib
import platform
from importlib.metadata import distributions, version
from pathlib import Path
from typing import Any

from ecocompanion.research.annotation import digest


def source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    return digest(
        {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix in {".py", ".html"}
        }
    )


def runtime_manifest() -> dict[str, Any]:
    packages = sorted(
        (dist.metadata["Name"].lower().replace("_", "-"), dist.version)
        for dist in distributions()
        if dist.metadata["Name"]
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "installed_distribution_inventory": digest(packages),
        "distributions": packages,
        **{
            name: version(name)
            for name in (
                "ecocompanion",
                "pydantic",
                "numpy",
                "scikit-learn",
                "fastapi",
                "typer",
                "uvicorn",
            )
        },
    }
