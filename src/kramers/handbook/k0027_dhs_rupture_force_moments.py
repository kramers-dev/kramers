"""K0027. Mean and variance of the rupture force under constant loading (Dudko, Hummer and Szabo)."""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
from scipy.special import polygamma

from .._checking import relative
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain, ValidityWarning
from .k0024_switching_distribution import switching_distribution
from .k0026_dhs_rate import dhs_rate

_EULER = 0.5772156649015329
_GAMMA_TILDE = _EULER**2 - 3.0 / np.pi**2 * float(polygamma(2, 1))


class ForceMoments(NamedTuple):
    mean: float
    variance: float


def dhs_rupture_force_moments(loading_rate: float, k0: float, x: float, dG: float, nu: float, kT: float = 1.0) -> ForceMoments:
    """<F> and var F for constant loading rate (Dudko, Hummer and Szabo Eqs. 5-6, kT restored).

    <F> = dG/(nu x) {1 - [kT/dG ln(k0 kT e^(dG/kT + gamma) / (x Fdot))]^nu}
    var F = pi^2 kT^2 / (6 x^2) [kT/dG ln(k0 kT e^(dG/kT + gamma~) / (x Fdot))]^(2 nu - 2)
    with gamma = 0.5772 and gamma~ = gamma^2 - (3/pi^2) psi''(1) = 1.0638.
    """
    if not (loading_rate > 0 and k0 > 0 and x > 0 and dG > 0 and kT > 0 and 0 < nu <= 1):
        raise OutsideDomain("all parameters must be positive and 0 < nu <= 1")
    b = dG / kT
    arg_m = np.log(k0 * kT / (x * loading_rate)) + b + _EULER
    arg_v = np.log(k0 * kT / (x * loading_rate)) + b + _GAMMA_TILDE
    if arg_m <= 0 or arg_v <= 0:
        raise OutsideDomain("loading too fast: the logarithm is negative (the bond breaks before the barrier matters)")
    if arg_m >= b:
        raise OutsideDomain("loading too slow: the formula gives a mean force <= 0, because the bond breaks "
                            "spontaneously before the force matters (loading rate below k0 kT e^gamma / x); "
                            "use the full distribution, K0024 with the rate of K0026")
    mean = dG / (nu * x) * (1.0 - (arg_m / b) ** nu)
    var = np.pi**2 * kT**2 / (6.0 * x**2) * (arg_v / b) ** (2 * nu - 2)
    return ForceMoments(mean=float(mean), variance=float(var))


def _check_against_exact_moments():
    rows = []
    for nu in (2.0 / 3.0, 0.5):
        for dG in (20.0, 60.0):
            k0, x, rate = 1e-4, 0.4, 1e3
            Fc = dG / (nu * x)
            with warnings.catch_warnings():  # the far tail of the grid crosses the low-barrier region
                warnings.simplefilter("ignore", ValidityWarning)
                dist = switching_distribution(lambda f, nu=nu, dG=dG, k0=k0, x=x: dhs_rate(f, k0, x, dG, nu), rate, 0.9999 * Fc)
            m = dhs_rupture_force_moments(rate, k0, x, dG, nu)
            rows.append((f"mean, nu={nu:.3f}, dG={dG:g}", m.mean, dist.mean))
            rows.append((f"variance, nu={nu:.3f}, dG={dG:g}", m.variance, dist.variance))
    return relative(rows, 0.02, ref="moments of the exact quasi-static distribution (K0024 with K0026)")


register(Entry(
    id="K0027",
    slug="dhs-rupture-force-moments",
    title="Mean and variance of the rupture force under constant loading (Dudko, Hummer and Szabo)",
    area="driven",
    kind="asymptotic",
    statement=(
        "For a bond with the rate of K0026 pulled at constant loading rate, the mean rupture "
        "force and its variance have closed forms in the logarithm of the loading rate, "
        "which reduce to the Bell-Evans logarithmic law for nu = 1."
    ),
    model=r"\text{K0026 rate},\ F(t)=\dot F\,t",
    formula=(r"\langle F\rangle\cong\frac{\Delta G^\ddagger}{\nu x^\ddagger}\left\{1-\left[\frac{1}{\beta\Delta G^\ddagger}"
             r"\ln\frac{k_0k_BTe^{\beta\Delta G^\ddagger+\gamma}}{x^\ddagger\dot F}\right]^{\nu}\right\},\ "
             r"\sigma_F^2\cong\frac{\pi^2(k_BT)^2}{6x^{\ddagger2}}\left[\frac{1}{\beta\Delta G^\ddagger}"
             r"\ln\frac{k_0k_BTe^{\beta\Delta G^\ddagger+\tilde\gamma}}{x^\ddagger\dot F}\right]^{2\nu-2}"),
    function=dhs_rupture_force_moments,
    parameters={"loading_rate": "dF/dt", "k0": "zero-force rate", "x": "x^ddagger", "dG": "barrier",
                "nu": "2/3, 1/2 or 1", "kT": "thermal energy"},
    sources=(
        Source(
            citation="O. K. Dudko, G. Hummer and A. Szabo, Phys. Rev. Lett. 96, 108101 (2006)",
            doi="10.1103/PhysRevLett.96.108101",
            locator="Eqs. (5)-(6), p. 108101-2 (energies in kT; kT restored here)",
            version_read="published PDF in Zenodo record 1233953 (CC0), opened 14 Oct 2026",
            printed=(r"\langle F\rangle\cong\frac{\Delta G^\ddagger}{\nu x^\ddagger}\left\{1-\left[\frac{1}{\Delta G^\ddagger}"
                     r"\ln\frac{k_0e^{\Delta G^\ddagger+\gamma}}{x^\ddagger KV}\right]^{\nu}\right\}"),
            notation="KV = loading rate; with kT restored, k0/(x KV) becomes k0 kT/(x KV)",
        ),
        Source(
            citation="A. Garg, Phys. Rev. B 51, 15592 (1995)",
            doi="10.1103/PhysRevB.51.15592",
            locator="credited by Dudko, Hummer and Szabo for Eqs. (5)-(6)",
            version_read="not opened (closed access)",
            role="original",
            evidence="citation-only",
        ),
    ),
    validity=("Several kT of barrier at the typical rupture force; quasi-static (adiabatic) "
              "loading. Against the exact moments of the same model (computed check): within 2% "
              "at dG = 20 kT, better at higher barriers."),
    checks=(
        Check("against the exact moments of the switching distribution", "K0024 with K0026", _check_against_exact_moments),
    ),
    aliases=("dynamic force spectroscopy", "loading-rate dependence of rupture force", "mean rupture force"),
    related=("K0024", "K0026", "K0029"),
    added="2026-10-14",
))
