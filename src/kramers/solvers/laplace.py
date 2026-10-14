"""Numerical inversion of Laplace transforms (the fixed Talbot method).

Many first-passage results are exact only in Laplace space: the transform of a
first-passage-time density, or a mean under resetting expressed through it. This
module turns them back into functions of time.

Method (published): J. Abate and P. P. Valko, "Multi-precision Laplace transform
inversion", Int. J. Numer. Methods Eng. 60, 979 (2004), the fixed Talbot contour.
In double precision about 0.6 M significant digits are reachable while M stays below
about 20; the default M = 18 gives about 1e-9 relative accuracy for smooth transforms
(computed check in ``tests/test_laplace.py``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..result import OutsideDomain

__all__ = ["invert"]


def invert(F: Callable[[NDArray[np.complex128]], NDArray[np.complex128]], t: ArrayLike, M: int = 18) -> NDArray[np.float64]:
    """f(t) from its Laplace transform F(s), for t > 0.

    ``F`` must accept complex arrays and be analytic to the right of its singularities,
    which must lie on or left of the imaginary axis' neighbourhood used by the contour.
    """
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    if np.any(t_arr <= 0):
        raise OutsideDomain("t must be positive")
    k = np.arange(1, M)
    theta = k * np.pi / M
    cot = 1.0 / np.tan(theta)
    sigma = theta + (theta * cot - 1.0) * cot
    out = np.empty_like(t_arr)
    for i, ti in enumerate(t_arr):
        r = 2.0 * M / (5.0 * ti)
        s = r * theta * (cot + 1j)
        first = 0.5 * np.exp(r * ti) * np.real(F(np.array([r + 0j]))[0])
        rest = np.real(np.exp(ti * s) * F(s) * (1.0 + 1j * sigma))
        out[i] = (r / M) * (first + np.sum(rest))
    return out
