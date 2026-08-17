"""The Dockerfile must reference files that exist.

This is the one part of the system whose failures are invisible to every other
test: the container build runs on a remote builder at deploy time, so a
Dockerfile that cannot possibly succeed sits green in CI and fails in front of
whoever is watching the deploy.

Two defects, both found by a failed deploy rather than by anything here:

    COPY copilot/__init__.py copilot/__init__.py
    → failed to compute cache key: "/copilot/__init__.py": not found

`copilot` was an implicit namespace package. Python and setuptools resolve those
without complaint, so the package imported correctly in every test, in the dev
server and in the editor. Only the image build cared.

And there was no `.dockerignore` at all, so `COPY . .` sent the entire working
tree to the builder — `data/` at 2.7 GB and `results/` at 4.3 GB — for an
application whose source is under a megabyte.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"


def _copy_sources() -> list[str]:
    """Every local path a COPY instruction reads from.

    Skips `--from=` copies, whose sources live in an earlier build stage rather
    than on disk.
    """
    sources: list[str] = []
    for raw in DOCKERFILE.read_text().splitlines():
        line = raw.strip()
        if not line.upper().startswith("COPY "):
            continue
        if "--from=" in line:
            continue
        parts = [p for p in line.split()[1:] if not p.startswith("--")]
        sources.extend(parts[:-1])   # the last argument is the destination
    return sources


class TestEveryCopySourceExists:
    def test_the_dockerfile_is_present(self):
        assert DOCKERFILE.exists()

    def test_copy_instructions_were_parsed(self):
        assert _copy_sources(), "no COPY sources found; the parser is broken"

    @pytest.mark.parametrize("source", _copy_sources())
    def test_the_source_exists(self, source):
        if source == ".":
            return
        assert (ROOT / source).exists(), (
            f"Dockerfile copies {source!r}, which does not exist. The image "
            f"build fails at that layer; nothing else in the suite can see it."
        )


class TestThePackageIsExplicit:
    def test_copilot_has_an_init(self):
        """The dependency-caching layer copies it on its own, so it has to be a
        real file rather than an implicit namespace package."""
        assert (ROOT / "copilot" / "__init__.py").exists()

    def test_it_is_importable_and_versioned(self):
        import copilot

        assert copilot.__version__


class TestTheBuildContextIsBounded:
    """`COPY . .` ships whatever the context contains."""

    def test_a_dockerignore_exists(self):
        assert DOCKERIGNORE.exists(), (
            "without one, COPY . . sends every local artifact to the builder"
        )

    @pytest.mark.parametrize("heavy", ["results/", "data/*.duckdb", ".venv", ".git"])
    def test_heavy_paths_are_excluded(self, heavy):
        assert heavy in DOCKERIGNORE.read_text()

    def test_the_warehouse_is_not_shipped(self):
        """It is generated from the CSV by `make build`. Baking a 2.7 GB
        derived artifact into the image ships the output instead of the
        recipe, and makes every rebuild a re-upload."""
        assert "duckdb" in DOCKERIGNORE.read_text()

    def test_the_source_data_IS_shipped(self):
        """The CSV is the input the warehouse is built from, so it must survive
        the ignore rules — excluding all of `data/` would produce an image that
        starts and then cannot answer anything."""
        ignored = DOCKERIGNORE.read_text().splitlines()
        assert "data/" not in [line.strip() for line in ignored]
        assert (ROOT / "data" / "ai4i2020.csv").exists()
