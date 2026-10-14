"""What every solver returns, and the warning a formula raises outside its regime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["Result", "ValidityWarning", "OutsideDomain"]


class ValidityWarning(UserWarning):
    """A formula was evaluated where the regime it was derived for does not hold.

    The value is still returned. The message says how large the error is known to be
    there, when a validity map gives it. Turn it into an error with
    ``warnings.simplefilter("error", kramers.ValidityWarning)``.
    """


class OutsideDomain(ValueError):
    """The request is outside the mathematical domain of a formula or solver."""


@dataclass(frozen=True)
class Result:
    """A computed number with the information needed to judge it.

    Attributes
    ----------
    value
        The number (or array of numbers).
    error
        Estimated absolute error. For deterministic solvers it is the difference
        between two resolutions; for simulation it is one standard error.
    method
        What was done, in words.
    details
        Anything else worth keeping: resolutions, truncation points, iterations.
    """

    value: float | NDArray[np.float64]
    error: float | NDArray[np.float64]
    method: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __float__(self) -> float:
        return float(np.asarray(self.value))

    def __array__(self, dtype: Any = None, copy: Any = None) -> NDArray[Any]:
        arr = np.asarray(self.value)
        return arr.astype(dtype) if dtype is not None else arr

    @property
    def relative_error(self) -> float | NDArray[np.float64]:
        value = np.asarray(self.value, dtype=float)
        error = np.asarray(self.error, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(error) / np.abs(value)
        return float(rel) if rel.ndim == 0 else rel

    def __repr__(self) -> str:
        value = np.asarray(self.value)
        if value.ndim == 0:
            text = f"{float(value):.12g} ± {float(np.asarray(self.error)):.2g}"
        else:
            text = f"array of shape {value.shape}"
        return f"Result({text}; {self.method})"
