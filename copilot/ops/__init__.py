"""Operator registry.

Importing this package registers every implemented operator. The registry is
closed at import time — a planner cannot introduce an operation at runtime.
"""

from copilot.ops import compare as _compare  # noqa: F401
from copilot.ops import counterfactual as _counterfactual  # noqa: F401
from copilot.ops import data_quality as _data_quality  # noqa: F401
from copilot.ops import describe as _describe  # noqa: F401
from copilot.ops import drift as _drift  # noqa: F401
from copilot.ops import drivers as _drivers  # noqa: F401
from copilot.ops import envelope as _envelope  # noqa: F401
from copilot.ops import forecast as _forecast  # noqa: F401
from copilot.ops import rate as _rate  # noqa: F401
from copilot.ops import records as _records  # noqa: F401
from copilot.ops import root_cause as _root_cause  # noqa: F401
from copilot.ops import sql_explore as _sql_explore  # noqa: F401
from copilot.ops import trend as _trend  # noqa: F401
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
