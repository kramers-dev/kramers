"""K0006. Escape rate near a saddle-node in the weak-noise (Kramers) limit."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import airy

from .._checking import combine, relative
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain, ValidityWarning


def _exact_rate(beta: np.ndarray, D: float) -> np.ndarray:
    """Reciprocal of the exact mean passage time (entry K0004), used for the warning."""
    arg = -beta / D ** (2.0 / 3.0)
    ai, _, bi, _ = airy(arg)
    return D ** (1.0 / 3.0) / (np.pi**2 * (ai**2 + bi**2))


def saddle_node_kramers_rate(beta: ArrayLike, D: float) -> np.ndarray | float:
    """Rate sqrt|beta|/pi * exp(-4|beta|^(3/2)/(3D)) for dx/dt = beta + x^2 + sqrt(2D) xi, beta < 0.

    Warns (``ValidityWarning``) when the barrier 4|beta|^(3/2)/(3D) is below 10, and
    says how far the formula is from the exact rate there.
    """
    b = np.asarray(beta, dtype=float)
    if np.any(b >= 0):
        raise OutsideDomain("the weak-noise rate needs beta < 0 (a barrier)")
    if not D > 0:
        raise OutsideDomain("D must be positive")
    barrier = 4.0 * np.abs(b) ** 1.5 / (3.0 * D)
    rate = np.sqrt(np.abs(b)) / np.pi * np.exp(-barrier)
    low = barrier < 10.0
    if np.any(low):
        err = np.max(np.abs(rate[low] / _exact_rate(b[low], D) - 1.0)) if b.ndim else abs(float(rate) / float(_exact_rate(b, D)) - 1.0)
        warnings.warn(f"barrier {float(np.min(barrier)):.2f} D is below 10: the weak-noise rate is off by up to "
                      f"{100 * err:.1f}% here (exact value: entry K0004)", ValidityWarning, stacklevel=2)
    return float(rate) if rate.ndim == 0 else rate


def _check_limit():
    rows = []
    for barrier in (25.0, 50.0, 100.0, 400.0):
        beta = -((0.75 * barrier) ** (2.0 / 3.0))  # D = 1
        rows.append((f"barrier {barrier:g}", saddle_node_kramers_rate(beta, 1.0), float(_exact_rate(np.array(beta), 1.0))))
    # the relative error must fall like 1/barrier: (Kramers / exact - 1) * B -> 5/18,
    # from the expansion Bi(z)^2 ~ exp(2 zeta) / (pi sqrt z) * (1 + 5 / (36 zeta)), zeta = B / 2
    first = relative(rows, 0.015, ref="exact rate, 1/T of K0004")
    scaled = [(c, abs(v / r - 1.0) * float(c.split()[1]), 5.0 / 18.0) for c, v, r in rows]
    second = relative(scaled, 0.06, what="error x barrier", ref="5/18")
    return combine(first, second)


def _check_printed_form():
    beta, D = -2.0, 0.3
    expected = np.sqrt(2.0) / np.pi * np.exp(-4.0 * 2.0**1.5 / (3.0 * 0.3))
    return relative([("beta=-2, D=0.3", saddle_node_kramers_rate(beta, D), expected)], 1e-15, ref="Eq. (3.10) evaluated by hand")


register(Entry(
    id="K0006",
    slug="saddle-node-weak-noise-rate",
    title="Escape rate near a saddle-node, weak noise (Kramers limit)",
    area="escape",
    kind="asymptotic",
    statement=(
        "Below the saddle-node bifurcation (beta < 0) and for weak noise, the rate of passage "
        "for dx/dt = beta + x^2 + sqrt(2D) xi is sqrt|beta|/pi times exp(-4|beta|^(3/2)/(3D)): "
        "Kramers' formula for the cubic potential. It is too high by 5/(18 B) to leading order "
        "with the barrier B = 4|beta|^(3/2)/(3D); the check maps it against the exact rate."
    ),
    model=r"\dot x = \beta + x^2 + \sqrt{2D}\,\xi(t),\ \beta<0,\ x_\pm=\pm\infty",
    formula=r"r \simeq \frac{\sqrt{|\beta|}}{\pi}\exp\!\left[-\frac{4|\beta|^{3/2}}{3D}\right],\quad D\ll|\beta|^{3/2}",
    function=saddle_node_kramers_rate,
    parameters={"beta": "bias (negative: below the bifurcation)", "D": "noise intensity"},
    sources=(
        Source(
            citation="B. Lindner, A. Longtin and A. Bulsara, Neural Comput. 15, 1761 (2003)",
            doi="10.1162/08997660360675035",
            locator="Eq. (3.10), p. 1772",
            version_read="author's PDF, page image read 12 Oct 2026",
            printed=r"r=\frac{\sqrt{|\beta|}}{\pi}\exp\left[-\frac{4|\beta|^{3/2}}{3D}\right]\quad\text{for }D\ll|\beta|^{3/2}",
            notation="same symbols",
        ),
        Source(
            citation="P. Colet, M. San Miguel, J. Casademunt and J. M. Sancho, Phys. Rev. A 39, 149 (1989)",
            doi="10.1103/PhysRevA.39.149",
            locator="credited by Lindner et al. for the saddle-point approximation",
            version_read="not opened; citation only",
            role="original",
            evidence="citation-only",
        ),
    ),
    validity=("Weak noise: barrier B = 4|beta|^(3/2)/(3D) large. The formula is too high by "
              "5/(18 B) + O(1/B^2) (computed check against K0004): 7.8% at B = 5, 3.2% at "
              "B = 10, 1.2% at B = 25, 0.57% at B = 50."),
    checks=(
        Check("approaches the exact rate as 1/barrier", "exact passage time in Airy functions (K0004)", _check_limit),
        Check("transcription", "Eq. (3.10) evaluated by hand", _check_printed_form),
    ),
    aliases=("Kramers rate of the cubic potential", "excitable regime firing rate of the theta neuron"),
    related=("K0004", "K0001"),
    added="2026-10-14",
))
