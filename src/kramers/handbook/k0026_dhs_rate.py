"""K0026. Force-dependent escape rate of Dudko, Hummer and Szabo (cusp, linear-cubic and Bell limits)."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike

from .._checking import combine, relative
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain, ValidityWarning
from ..solvers.quadrature import FirstPassage1D
from .k0001_kramers_overdamped import kramers_rate_overdamped


def dhs_rate(F: ArrayLike, k0: float, x: float, dG: float, nu: float, kT: float = 1.0) -> np.ndarray | float:
    """k(F) = k0 (1 - nu F x/dG)^(1/nu - 1) exp{dG/kT [1 - (1 - nu F x/dG)^(1/nu)]}.

    nu = 2/3: linear-cubic free-energy surface; nu = 1/2: harmonic well with a cusp;
    nu = 1: Bell's model (then x is force-independent). F < dG/(nu x) is required;
    near it the barrier vanishes and the rate picture fails.
    """
    if not (k0 > 0 and x > 0 and dG > 0 and kT > 0 and 0 < nu <= 1):
        raise OutsideDomain("need k0, x, dG, kT > 0 and 0 < nu <= 1")
    f = np.asarray(F, dtype=float)
    s = 1.0 - nu * f * x / dG
    if np.any(s <= 0):
        raise OutsideDomain("F must stay below the critical force dG / (nu x)")
    barrier = dG * s ** (1.0 / nu) / kT
    if nu < 1 and np.any(barrier < 3.0):
        warnings.warn("barrier below 3 kT: the rate formula (a Kramers-type limit) is unreliable there",
                      ValidityWarning, stacklevel=2)
    k = k0 * s ** (1.0 / nu - 1.0) * np.exp(dG / kT * (1.0 - s ** (1.0 / nu)))
    return float(k) if k.ndim == 0 else k


def _cubic_landscape(F: float, x: float, dG: float):
    """U0 = (3/2) dG (y/x) - 2 dG (y/x)^3 tilted by -F y (kT = 1), with its well, barrier and curvatures."""
    # stationary points of U = 1.5 dG t - 2 dG t^3 - F x t, t = y/x: 1.5 dG - 6 dG t^2 - F x = 0
    t0 = np.sqrt((1.5 * dG - F * x) / (6.0 * dG))
    U = lambda y: 1.5 * dG * (y / x) - 2.0 * dG * (y / x) ** 3 - F * y  # noqa: E731
    curv = 12.0 * dG * t0 / x**2
    well, top = -t0 * x, t0 * x
    return U, well, top, curv, U(top) - U(well)


def _check_kramers_identity():
    # nu = 2/3 is Kramers' formula for the tilted linear-cubic surface, at every force
    x, dG, zeta = 0.4, 20.0, 1.0
    _, _, _, c0, b0 = _cubic_landscape(0.0, x, dG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValidityWarning)
        k0 = kramers_rate_overdamped(c0, c0, b0, zeta)
        rows = []
        for frac in (0.1, 0.4, 0.7):
            F = frac * dG / ((2.0 / 3.0) * x)
            _, _, _, c, b = _cubic_landscape(F, x, dG)
            rows.append((f"F = {frac} Fc", dhs_rate(F, k0, x, dG, 2.0 / 3.0), kramers_rate_overdamped(c, c, b, zeta)))
    return relative(rows, 1e-12, ref="Kramers' formula (K0001) for the tilted surface")


def _check_exact_ratio():
    # the force dependence k(F)/k(0) approaches the exact decay rate ratio as the barrier grows
    parts = []
    for nu, label in ((2.0 / 3.0, "linear-cubic"), (0.5, "cusp")):
        rows = []
        for dG in (40.0, 120.0):
            x = 1.0
            Fc = dG / (nu * x)

            def exact(F, nu=nu, dG=dG, x=x):
                if nu == 0.5:
                    fp = FirstPassage1D(lambda y, dG=dG, F=F: dG * y * y - F * y, D=1.0, domain=(-np.inf, x), left="natural", right="absorbing")
                else:
                    U, _, _, _, _ = _cubic_landscape(F, x, dG)
                    fp = FirstPassage1D(U, D=1.0, left="natural", right="exit")
                return float(fp.decay_rate().value)

            F = 0.3 * Fc
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ValidityWarning)
                ratio = dhs_rate(F, 1.0, x, dG, nu)
            rows.append((f"{label}, dG={dG:g}", ratio, exact(F) / exact(0.0)))
        parts.append(relative(rows, 0.03, what="k(F)/k(0)", ref="exact decay-rate ratio"))
    return combine(*parts)


register(Entry(
    id="K0026",
    slug="dhs-force-dependent-rate",
    title="Force-dependent escape rate of Dudko, Hummer and Szabo",
    area="driven",
    kind="asymptotic",
    statement=(
        "Under a constant force F, the escape rate over a barrier of height dG at distance x "
        "is k0 (1 - nu F x/dG)^(1/nu - 1) exp{dG [1 - (1 - nu F x/dG)^(1/nu)]/kT}: nu = 2/3 "
        "for a linear-cubic surface, 1/2 for a harmonic well with a cusp, and nu = 1 gives "
        "Bell's exponential law. For nu = 2/3 it is exactly Kramers' formula for the tilted "
        "surface, the large-barrier limit of the noisy saddle-node."
    ),
    model=r"U(y)=U_0(y)-Fy,\ \text{overdamped},\ \nu=2/3:\ U_0=\tfrac32\Delta G^\ddagger y/x^\ddagger-2\Delta G^\ddagger(y/x^\ddagger)^3",
    formula=(r"k(F)=k_0\left(1-\frac{\nu Fx^\ddagger}{\Delta G^\ddagger}\right)^{1/\nu-1}"
             r"\exp\left\{\beta\Delta G^\ddagger\left[1-\left(1-\frac{\nu Fx^\ddagger}{\Delta G^\ddagger}\right)^{1/\nu}\right]\right\}"),
    function=dhs_rate,
    parameters={"F": "force", "k0": "rate at zero force", "x": "distance to the barrier x^ddagger",
                "dG": "barrier at zero force", "nu": "2/3, 1/2 or 1", "kT": "thermal energy"},
    sources=(
        Source(
            citation="O. K. Dudko, G. Hummer and A. Szabo, Phys. Rev. Lett. 96, 108101 (2006)",
            doi="10.1103/PhysRevLett.96.108101",
            locator="Eq. (3), p. 108101-2 (energies in kT)",
            version_read="published PDF in Zenodo record 1233953 (CC0), opened 14 Oct 2026",
            printed=(r"k(F)=k_0\left(1-\frac{\nu Fx^\ddagger}{\Delta G^\ddagger}\right)^{1/\nu-1}"
                     r"e^{\Delta G^\ddagger[1-(1-\nu Fx^\ddagger/\Delta G^\ddagger)^{1/\nu}]}"),
            notation="their energies are in units of kT; here beta = 1/kT is written out",
        ),
        Source(
            citation="O. K. Dudko, G. Hummer and A. Szabo, Proc. Natl. Acad. Sci. USA 105, 15755 (2008)",
            doi="10.1073/pnas.0806085105",
            locator="Eq. [5], p. 15757 (lifetime tau(F) = 1/k(F), beta explicit)",
            version_read="PubMed Central PMC2572921 equation images, opened 14 Oct 2026",
            role="restatement",
        ),
    ),
    validity=("Force below the critical force dG/(nu x) with a barrier of several kT left; "
              "overdamped, single barrier, soft pulling device. For nu = 2/3 the error against the "
              "exact rate is that of Kramers' formula (K0001: 3% at a 10 kT barrier)."),
    checks=(
        Check("nu = 2/3 equals Kramers' formula for the tilted surface", "K0001", _check_kramers_identity),
        Check("force dependence against exact decay rates", "quadrature solver, cubic and cusp landscapes", _check_exact_ratio, slow=True),
    ),
    aliases=("Dudko-Hummer-Szabo", "unified rate for force spectroscopy", "linear-cubic potential", "cusp potential"),
    related=("K0001", "K0024", "K0027", "K0029"),
    added="2026-10-14",
))
