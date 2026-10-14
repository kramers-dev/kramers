"""Every entry's function under its own name, for direct use.

    >>> from kramers.formulas import passage_time_saddle_node
"""

from __future__ import annotations

from typing import Any

from .registry import entries


def __getattr__(name: str) -> Any:
    for e in entries():
        if e.function.__name__ == name:
            return e.function
    raise AttributeError(f"no formula named {name!r}; see kramers.search()")


def __dir__() -> list[str]:
    return sorted(e.function.__name__ for e in entries())
