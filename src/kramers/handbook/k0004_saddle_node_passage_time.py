"""K0004. Mean passage time through a noisy saddle-node, in closed form (Airy functions)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import airy, airye, gamma

from .._checking import relative
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from ..solvers.quadrature import FirstPassage1D


def _scaled_T(alpha: np.ndarray) -> np.ndarray:
    """T(alpha) = 2^(1/3) pi^2 [Ai^2(z) + Bi^2(z)], z = -2^(2/3) alpha, without overflow."""
    z = -(2.0 ** (2.0 / 3.0)) * np.asarray(alpha, dtype=float)
    out = np.empty_like(z)
    pos = z > 0
    if np.any(pos):
        # for z > 0, Ai = eAi exp(-zeta) and Bi = eBi exp(+zeta): no overflow until the end
        eai, _, ebi, _ = airye(z[pos])
        zeta = (2.0 / 3.0) * z[pos] ** 1.5
        with np.errstate(over="ignore", under="ignore"):
            out[pos] = np.exp(2 * zeta) * (ebi**2 + eai**2 * np.exp(-4 * zeta))
    if np.any(~pos):
        ai, _, bi, _ = airy(z[~pos])  # bounded and oscillating for z <= 0
        out[~pos] = ai**2 + bi**2
    return 2.0 ** (1.0 / 3.0) * np.pi**2 * out


def passage_time_saddle_node(epsilon0: ArrayLike, g0: float) -> np.ndarray | float:
    """Mean time from -inf to +inf for dx/dt = x^2 + epsilon0 + g0 xi(t).

    tau = g0^(-2/3) T(alpha), alpha = epsilon0 / g0^(4/3), with Hathcock and Sethna's
    T(alpha) = 2^(1/3) pi^2 [Ai^2(-2^(2/3) alpha) + Bi^2(-2^(2/3) alpha)].
    epsilon0 < 0 is below the bifurcation (a well and a barrier of 4/3 |epsilon0|^(3/2)).
    With noise written as sqrt(2D) xi, g0 = sqrt(2D).
    """
    if not g0 > 0:
        raise OutsideDomain("g0 must be positive")
    eps = np.asarray(epsilon0, dtype=float)
    tau = g0 ** (-2.0 / 3.0) * _scaled_T(np.atleast_1d(eps / g0 ** (4.0 / 3.0)))
    if np.any(~np.isfinite(tau)):
        raise OutsideDomain("passage time beyond floating-point range (barrier above about 700 noise units)")
    return float(tau[0]) if eps.ndim == 0 else tau.reshape(eps.shape)


def _check_quadrature():
    rows = []
    for eps0, g0 in ((-3.0, 1.0), (-1.0, 0.5), (0.0, 1.3), (0.5, 1.0), (2.0, 0.7), (-20.0, 2.0)):
        D = g0 * g0 / 2.0
        fp = FirstPassage1D(lambda x, e=eps0, D=D: -(x**3 / 3.0 + e * x) / D, D=D, left="natural", right="exit",
                            dlog_psi=lambda x, e=eps0, D=D: (x * x + e) / D)
        rows.append((f"epsilon0={eps0}, g0={g0}", passage_time_saddle_node(eps0, g0), float(fp.mean_time(-np.inf).value)))
    return relative(rows, 1e-11, ref="quadrature solver (independent: Green's function, entrance and exit tails)")


def _check_lindner():
    rows = []
    for D in (0.05, 0.8, 3.0):
        lindner = gamma(1.0 / 3.0) ** 2 * (1.0 / (3.0 * D)) ** (1.0 / 3.0)  # Lindner et al. Eq. (3.4)
        rows.append((f"epsilon0=0, D={D}", passage_time_saddle_node(0.0, np.sqrt(2 * D)), lindner))
    return relative(rows, 1e-13, ref="Lindner et al. (2003), Eq. (3.4)")


def _check_limits():
    # Hathcock and Sethna Eqs. (8)-(9): T ~ pi |alpha|^(-1/2) e^(8/3 |alpha|^(3/2)) as alpha -> -inf,
    # T ~ pi alpha^(-1/2) as alpha -> +inf. Asymptotic, so the ratio must approach 1.
    rows = []
    for a in (-6.0, -12.0, -24.0):
        rows.append((f"alpha={a:g} against Eq. (8)", float(_scaled_T(np.array([a]))[0]),
                     np.pi / np.sqrt(abs(a)) * np.exp(8.0 / 3.0 * abs(a) ** 1.5)))
    for a in (100.0, 400.0):
        rows.append((f"alpha={a:g} against Eq. (9)", float(_scaled_T(np.array([a]))[0]), np.pi / np.sqrt(a)))
    return relative(rows, 0.01, ref="asymptotic limits, Eqs. (8)-(9)")


register(Entry(
    id="K0004",
    slug="saddle-node-passage-time",
    title="Mean passage time through a noisy saddle-node, closed form in Airy functions",
    area="escape",
    kind="exact",
    statement=(
        "The mean time for dx/dt = x^2 + epsilon0 + g0 xi to go from minus to plus infinity "
        "is g0^(-2/3) 2^(1/3) pi^2 [Ai^2 + Bi^2] of -2^(2/3) epsilon0 / g0^(4/3), at any "
        "distance from the bifurcation and any barrier height, on either side of the fold."
    ),
    model=r"\dot x = x^2 + \epsilon_0 + g_0\,\xi(t),\quad x: -\infty \to +\infty",
    formula=(r"\tau = g_0^{-2/3}\,\mathcal{T}(\alpha),\ \alpha=\epsilon_0/g_0^{4/3},\ "
             r"\mathcal{T}(\alpha)=2^{1/3}\pi^2\left[\mathrm{Ai}^2(-2^{2/3}\alpha)+\mathrm{Bi}^2(-2^{2/3}\alpha)\right]"),
    function=passage_time_saddle_node,
    parameters={"epsilon0": "distance from the bifurcation (negative: a barrier)", "g0": "noise amplitude (g0 = sqrt(2D))"},
    sources=(
        Source(
            citation="D. Hathcock and J. P. Sethna, Phys. Rev. Research 3, 013156 (2021)",
            doi="10.1103/PhysRevResearch.3.013156",
            locator="Eq. (15), p. 013156-4; scaling Eq. (7) and limits Eqs. (8)-(9), p. 013156-3",
            version_read="journal PDF (CC BY 4.0) and arXiv:1902.07382v3, page images read 14 Oct 2026",
            printed=r"\mathcal{T}(\alpha)=2^{1/3}\pi^2\left[\mathrm{Ai}^2(-2^{2/3}\alpha)+\mathrm{Bi}^2(-2^{2/3}\alpha)\right]",
            notation="their V(x) = -x^3/3 - epsilon0 x with constant noise g0; D = g0^2/2",
        ),
        Source(
            citation="B. Lindner, A. Longtin and A. Bulsara, Neural Comput. 15, 1761 (2003)",
            doi="10.1162/08997660360675035",
            locator="Eq. (3.1), p. 1770 (double integral); Eq. (3.3) (series, after Colet et al.)",
            version_read="author's PDF, page image read 12 Oct 2026",
            role="restatement",
        ),
        Source(
            citation="B. I. Halperin, Phys. Rev. 139, A104 (1965)",
            doi="10.1103/PhysRev.139.A104",
            locator="the same function of Ai and Bi, as a density of states in a white-noise potential",
            version_read="not opened; attribution from a search summary and from GPT-6 Astra",
            role="original",
            evidence="citation-only",
        ),
    ),
    validity="Constant (additive) white noise; start at -infinity, absorption at +infinity. Exact.",
    checks=(
        Check("against the quadrature solver", "Green's-function quadrature with entrance and exit tails", _check_quadrature),
        Check("at the bifurcation, against another paper", "Lindner et al. (2003), Eq. (3.4)", _check_lindner),
        Check("the printed asymptotic limits", "Hathcock and Sethna, Eqs. (8)-(9)", _check_limits),
    ),
    aliases=("noisy saddle-node bifurcation", "mean interspike interval of the theta neuron",
             "quadratic integrate-and-fire neuron", "barrier crossing at any barrier height",
             "Airy function passage time"),
    related=("K0005", "K0006", "K0008", "K0009", "K0011"),
    notes=("This is the mean passage time. Its reciprocal is the firing rate of the quadratic "
           "integrate-and-fire neuron (K0009) but not the quasi-stationary decay rate (K0008); "
           "the two differ by 43% at the bifurcation point."),
    added="2026-10-14",
))
