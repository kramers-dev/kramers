"""kramers: an open handbook of first-passage and escape results in statistical physics.

Every result is a function, tied to the paper and equation that print it, and checked
against an independent solver. Start with::

    import kramers
    kramers.search("resetting")          # find entries
    e = kramers.get("K0014")             # one entry: e.function, e.sources, e.cite()
    e.verify()                           # run its checks
    from kramers.formulas import resetting_mean_time
"""

from ._version import __version__
from .registry import AREAS, KINDS, Entry, entries, get, search, to_json
from .result import OutsideDomain, Result, ValidityWarning
from .solvers import FirstPassage1D

__all__ = [
    "AREAS",
    "KINDS",
    "Entry",
    "FirstPassage1D",
    "OutsideDomain",
    "Result",
    "ValidityWarning",
    "__version__",
    "entries",
    "get",
    "search",
    "to_json",
]
