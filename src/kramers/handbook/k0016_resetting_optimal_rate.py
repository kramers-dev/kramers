"""K0016. Optimal resetting rate for diffusion to a target (Evans and Majumdar)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .._checking import relative
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from .k0015_resetting_mean_time import resetting_mean_time


@lru_cache(maxsize=1)
def _z_star() -> float:
    """Positive root of z/2 = 1 - exp(-z)."""
    return float(brentq(lambda z: z / 2.0 - 1.0 + np.exp(-z), 0.5, 3.0, xtol=1e-16, rtol=1e-15))


def resetting_optimal_rate(x0: float, D: float) -> float:
    """Resetting rate that minimises the mean time to the target: r* = z*^2 D / x0^2, z* = 1.5936..."""
    if not (x0 > 0 and D > 0):
        raise OutsideDomain("need x0 > 0 and D > 0")
    return _z_star() ** 2 * D / x0**2


def _check_minimum():
    rows = []
    for x0, D in ((1.0, 1.0), (0.4, 2.5)):
        res = minimize_scalar(lambda lr, x0=x0, D=D: np.log(resetting_mean_time(x0, np.exp(lr), D)), bounds=(-10.0, 10.0),
                              method="bounded", options={"xatol": 1e-10})
        rows.append((f"x0={x0}, D={D}", resetting_optimal_rate(x0, D), float(np.exp(res.x))))
    return relative(rows, 1e-6, ref="numerical minimum of K0015")


def _check_printed():
    return relative([("z* (printed 1.59362...)", round(_z_star(), 5), 1.59362),
                     ("r* x0^2 / D (Pal, Kundu and Evans: 2.53964...)", round(resetting_optimal_rate(1.0, 1.0), 5), 2.53964)],
                    1e-12, ref="printed digits")


register(Entry(
    id="K0016",
    slug="resetting-optimal-rate",
    title="Optimal resetting rate for diffusion to a target",
    area="resetting",
    kind="exact",
    statement=(
        "The mean time for a resetting Brownian searcher to find a target at distance x0 is "
        "smallest at the rate r* = z*^2 D/x0^2, where z* = 1.59362... solves z/2 = 1 - exp(-z)."
    ),
    model=r"\text{as K0015}",
    formula=r"r^*=\frac{(z^*)^2D}{x_0^2},\quad \frac{z^*}{2}=1-e^{-z^*},\ z^*=1.59362\ldots",
    function=resetting_optimal_rate,
    parameters={"x0": "start and reset position", "D": "diffusion coefficient"},
    sources=(
        Source(
            citation="M. R. Evans and S. N. Majumdar, Phys. Rev. Lett. 106, 160601 (2011)",
            doi="10.1103/PhysRevLett.106.160601",
            locator="Eq. (8) and the text after it (arXiv:1102.2704v1 numbering)",
            version_read="arXiv:1102.2704v1 HTML, opened 14 Oct 2026",
            printed=r"\frac{z^{*}}{2}=1-{\rm e}^{-z^{*}},\ z^*=1.59362\ldots,\ r^*=(z^*)^2D/x_0^2",
            notation="same symbols",
        ),
        Source(
            citation="A. Pal, A. Kundu and M. R. Evans, J. Phys. A 49, 225001 (2016)",
            doi="10.1088/1751-8113/49/22/225001",
            locator="text next to Eq. (57) (arXiv:1512.08211v2): 2.53964...",
            version_read="arXiv:1512.08211v2 HTML, opened 14 Oct 2026",
            role="restatement",
        ),
    ),
    validity="As K0015. Exact.",
    checks=(
        Check("is the minimum of the mean time", "numerical minimisation of K0015", _check_minimum),
        Check("the printed digits", "Evans and Majumdar; Pal, Kundu and Evans", _check_printed),
    ),
    aliases=("optimal restart", "resetting transition"),
    related=("K0015", "K0018"),
    added="2026-10-14",
))
