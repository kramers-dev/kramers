"""K0029. Most probable rupture force of a Bell bond under constant loading (Bell-Evans)."""

from __future__ import annotations

import numpy as np

from .._checking import relative
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from .k0024_switching_distribution import switching_distribution


def bell_evans_force(loading_rate: float, k0: float, x: float, kT: float = 1.0) -> float:
    """F* = (kT/x) ln(loading_rate x / (k0 kT)), the mode of the rupture-force density."""
    if not (loading_rate > 0 and k0 > 0 and x > 0 and kT > 0):
        raise OutsideDomain("all parameters must be positive")
    arg = loading_rate * x / (k0 * kT)
    if arg <= 1.0:
        raise OutsideDomain("loading too slow: below r x = k0 kT the most probable force is zero")
    return float(kT / x * np.log(arg))


def _check_mode():
    rows = []
    for k0, x, r in ((1e-3, 0.5, 20.0), (1e-6, 0.2, 1e4), (0.1, 1.0, 1.0)):
        F_star = bell_evans_force(r, k0, x)
        dist = switching_distribution(lambda f, k0=k0, x=x: k0 * np.exp(f * x), r, F_star + 40.0 / x, n=33)
        fine = np.linspace(F_star - 2.0 / x, F_star + 2.0 / x, 40001)
        rows.append((f"k0={k0}, x={x}, rate={r}", F_star, float(fine[np.argmax(dist.pdf(fine))])))
    return relative(rows, 2e-4, ref="mode of the exact density (K0024)")


def _check_dhs_limit():
    # Dudko, Hummer and Szabo Eq. (5) with nu = 1 and gamma -> 0 gives the mode
    k0, x, r, dG = 1e-3, 0.5, 20.0, 50.0
    nu = 1.0
    dhs = dG / (nu * x) * (1.0 - (np.log(k0 * np.exp(dG) / (x * r)) / dG) ** nu)
    return relative([("nu = 1, gamma -> 0", bell_evans_force(r, k0, x), float(dhs))], 1e-12,
                    ref="Dudko, Hummer and Szabo (2006) Eq. (5)")


register(Entry(
    id="K0029",
    slug="bell-evans-force",
    title="Most probable rupture force under constant loading (Bell-Evans)",
    area="driven",
    kind="exact",
    statement=(
        "A bond whose lifetime falls exponentially with force (Bell's model) breaks most often "
        "at the force (kT/x) ln(r x/(k0 kT)) when the force rises at rate r: the logarithmic "
        "loading-rate law of dynamic force spectroscopy."
    ),
    model=r"k(F)=k_0\,e^{Fx/k_BT},\quad F=\dot F\,t",
    formula=r"F^*=\frac{k_BT}{x}\ln\!\left[\frac{\dot F\,x}{k_0k_BT}\right]",
    function=bell_evans_force,
    parameters={"loading_rate": "dF/dt", "k0": "zero-force rate", "x": "distance to the transition state", "kT": "thermal energy"},
    sources=(
        Source(
            citation="Z. Liu, H. Liu, A. M. Vera, R. C. Bernardi, P. Tinnefeld and M. A. Nash, Nat. Commun. 11, 4321 (2020)",
            doi="10.1038/s41467-020-18063-x",
            locator="Eq. (2), Methods",
            version_read="Europe PMC full-text XML PMC7456326 (CC BY), opened 14 Oct 2026",
            printed=r"F=\frac{k_{\mathrm B}T}{\Delta x^\ddagger}\ln\left[\frac{r_{\mathrm f}\,\Delta x^\ddagger}{k_0k_{\mathrm B}T}\right]",
            notation="r_f = loading rate, Delta x^ddagger = x",
        ),
        Source(
            citation="E. Evans and K. Ritchie, Biophys. J. 72, 1541 (1997)",
            doi="10.1016/S0006-3495(97)78802-7",
            locator="Eq. (13), p. 1547: the condition for the most probable force (the closed form is not printed there)",
            version_read="scanned pages in PubMed Central PMC1184350, opened 14 Oct 2026",
            role="original",
        ),
    ),
    validity="Bell's rate law (force-independent x), single barrier, slow loading compared with molecular relaxation, r x > k0 kT. Exact for that model.",
    checks=(
        Check("is the mode of the exact rupture-force density", "K0024 with the Bell rate", _check_mode),
        Check("the nu = 1 limit of the mean-force formula", "Dudko, Hummer and Szabo (2006) Eq. (5)", _check_dhs_limit),
    ),
    aliases=("Bell model", "dynamic strength of bonds", "most probable unbinding force", "Evans-Ritchie"),
    related=("K0024", "K0026", "K0027"),
    notes=("Often cited to Evans and Ritchie (1997), who derive it (their Eq. 13 with the Bell "
           "rate) but print only that condition and the scaling; the closed form is printed in "
           "Liu et al. (2020), among others."),
    added="2026-10-14",
))
