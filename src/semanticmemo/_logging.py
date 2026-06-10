"""Internal logging setup.

SemanticMemo is a library, so it is silent by default: it attaches a single
``logging.NullHandler`` to the ``SemanticMemo`` parent logger and never configures
handlers or levels itself. An application that wants to see SemanticMemo logs opts
in explicitly, e.g.::

    import logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("semanticmemo").setLevel(logging.DEBUG)
"""

from __future__ import annotations

import logging

# Installed once, at import time, on the package's root logger. Child loggers
# (``SemanticMemo.orchestrator`` etc.) propagate to it, so this single handler
# keeps the whole library quiet unless the application configures logging.
logging.getLogger("semanticmemo").addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``SemanticMemo`` namespace.

    Call as ``get_logger(__name__)`` from within the package so the logger name
    is, for example, ``SemanticMemo.store.sqlite_store``.
    """
    return logging.getLogger(name)
