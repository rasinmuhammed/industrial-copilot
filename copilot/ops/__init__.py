"""Operator registry.

Importing this package registers every implemented operator. The registry is
closed at import time — a planner cannot introduce an operation at runtime.
"""

from copilot.ops import compare as _compare  # noqa: F401
from copilot.ops import describe as _describe  # noqa: F401
from copilot.ops import rate as _rate  # noqa: F401
from copilot.ops import root_cause as _root_cause  # noqa: F401
from copilot.ops.registry import (
    ExecutionContext,
    data_fingerprint,
    execute,
    get_op,
    kb_version,
    registered,
)

__all__ = [
    "ExecutionContext",
    "execute",
    "get_op",
    "registered",
    "data_fingerprint",
    "kb_version",
]
