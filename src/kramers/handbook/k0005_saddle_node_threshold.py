"""K0005. Passage through a noisy saddle-node exactly at the bifurcation: mean, variance, CV."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.special import gamma

from .._checking import relative
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from ..solvers.quadrature import FirstPassage1D

_G13 = gamma(1.0 / 3.0)


class PassageStatistics(NamedTuple):
    mean: float
    variance: float
    rate: float
    cv: float


def saddle_node_threshold_statistics(D: float) -> PassageStatistics:
    """Statistics of the passage from -inf to +inf of dx/dt = x^2 + sqrt(2D) xi(t).

    This is the type I (quadratic integrate-and-fire, theta) neuron biased exactly at
    rheobase: mean interspike interval, its variance, the firing rate and the
    coefficient of variation.
    """
    if not D > 0:
        raise OutsideDomain("D must be positive")
    mean = _G13**2 * (1.0 / (3.0 * D)) ** (1.0 / 3.0)
    variance = _G13**4 * (1.0 / (3.0 * D)) ** (2.0 / 3.0) / 3.0
    return PassageStatistics(mean=mean, variance=variance, rate=1.0 / mean, cv=1.0 / np.sqrt(3.0))


def _check_quadrature():
    rows = []
    for D in (0.05, 1.0, 7.0):
        fp = FirstPassage1D(lambda x, D=D: -(x**3) / (3.0 * D), D=D, left="natural", right="exit",
                            dlog_psi=lambda x, D=D: x**2 / D)
        m = fp.moments(-np.inf, n=2).value
        s = saddle_node_threshold_statistics(D)
        rows.append((f"mean, D={D}", s.mean, m[0]))
        rows.append((f"variance, D={D}", s.variance, m[1] - m[0] ** 2))
    return relative(rows, 1e-11, ref="quadrature solver (entrance at -inf, exit at +inf)")


def _check_printed_numbers():
    s = saddle_node_threshold_statistics(1.0)
    return relative([("rate coefficient, printed 0.201", round(s.rate, 3), 0.201),
                     ("CV, printed 1/sqrt(3) ~ 0.577", round(s.cv, 3), 0.577)], 1e-12, ref="Lindner et al., p. 1771")


register(Entry(
    id="K0005",
    slug="saddle-node-at-threshold",
    title="Noisy saddle-node exactly at the bifurcation: mean, variance and CV of the passage time",
    area="escape",
    kind="exact",
    statement=(
        "At the saddle-node bifurcation point, the time to pass from minus to plus infinity "
        "under dx/dt = x^2 + sqrt(2D) xi has mean Gamma(1/3)^2 (3D)^(-1/3) and variance one "
        "third of the mean squared, so its coefficient of variation is 1/sqrt(3) whatever "
        "the noise intensity."
    ),
    model=r"\dot x = x^2 + \sqrt{2D}\,\xi(t),\quad x(0)=-\infty,\ \text{passage to } +\infty",
    formula=(r"\langle T\rangle=\Gamma(\tfrac13)^2(3D)^{-1/3},\quad "
             r"\langle\Delta T^2\rangle=\tfrac13\langle T\rangle^2,\quad CV=1/\sqrt3"),
    function=saddle_node_threshold_statistics,
    parameters={"D": "noise intensity"},
    sources=(
        Source(
            citation="B. Lindner, A. Longtin and A. Bulsara, Neural Comput. 15, 1761 (2003)",
            doi="10.1162/08997660360675035",
            locator="Eqs. (3.4)-(3.6), p. 1771",
            version_read="author's PDF (amor.cms.hu-berlin.de/~lindnebe), page images read 12 Oct 2026",
            printed=(r"\langle T(\beta=0)\rangle=[\Gamma(1/3)]^2\left(\frac{1}{3D}\right)^{1/3};\ "
                     r"\langle\Delta T^2(\beta=0)\rangle=\frac13[\Gamma(1/3)]^4\left(\frac{1}{3D}\right)^{2/3}=\frac13\langle T\rangle^2;\ "
                     r"r(\beta=0,D)=\frac{1}{[\Gamma(1/3)]^2}(3D)^{1/3}\approx0.201D^{1/3},\ CV=1/\sqrt3"),
            notation="their Eq. (2.1) x' = beta + x^2 + sqrt(2D) xi(t) with beta = 0 and thresholds at -inf, +inf",
        ),
        Source(
            citation="D. Sigeti and W. Horsthemke, J. Stat. Phys. 54, 1217 (1989)",
            doi="10.1007/BF01044713",
            locator="credited by Lindner et al. for Eqs. (3.4)-(3.5)",
            version_read="not opened; citation only",
            role="original",
            evidence="citation-only",
        ),
    ),
    validity="White Gaussian noise, thresholds at plus and minus infinity, bias exactly zero. Exact.",
    checks=(
        Check("mean and variance against the quadrature solver", "quadrature with entrance and exit tails", _check_quadrature),
        Check("the numbers printed in the paper", "Lindner et al. (2003), p. 1771", _check_printed_numbers),
    ),
    aliases=("theta neuron at rheobase", "quadratic integrate-and-fire neuron at threshold",
             "Sigeti-Horsthemke", "coefficient of variation 1/sqrt(3)"),
    related=("K0004", "K0006", "K0007", "K0009"),
    added="2026-10-14",
))
