"""Argus — margin-based industrial maintenance copilot.

An explicit package rather than an implicit namespace one. That is not a style
preference: the Dockerfile installs dependencies in their own layer by copying
`pyproject.toml` and this file alone, so source edits do not re-resolve the
wheel set. Without this file that COPY had no source, and the image build failed
at the layer with:

    failed to compute cache key: "/copilot/__init__.py": not found

The package imported fine locally the whole time, because setuptools resolves
namespace packages happily — so nothing in the test suite or the dev server
could see it. Only the container build could, and only at deploy time.
"""

__version__ = "0.1.0"
