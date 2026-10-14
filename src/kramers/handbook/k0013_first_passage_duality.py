"""K0013. First-passage duality: conditional exit times do not depend on the sign of the drift."""

from __future__ import annotations

import numpy as np

from .._checking import absolute, combine, relative
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from ..solvers import fokker_planck
from .k0012_drift_diffusion_interval import drift_diffusion_interval


def conditional_time_to_origin(ell: float, L: float, v: float, D: float) -> float:
    """Mean time to hit 0 before L from ell, given that 0 is hit first (Krapivsky and Redner, Eq. 28b).

    v > 0 is drift away from the origin. The value is the same for v and -v.
    """
    if not (0 < ell < L and D > 0 and v != 0):
        raise OutsideDomain("need 0 < ell < L, D > 0 and v != 0 (use K0012 for v = 0)")
    ea, el = np.exp(-v * L / D), np.exp(-v * ell / D)
    return float(ell / v * (ea + el) / (el - ea) - 2 * L / v * (1 - el) / (1 - ea) * ea / (el - ea))


def _conditional_densities(x0: float, L: float, v: float, D: float):
    sol = fokker_planck.solve(lambda x: v + 0 * x, D, (0.0, L), left="absorbing", right="absorbing", x0=x0,
                              t_end=6.0 * L * L / D, n_cells=800, dt=L * L / D / 500)
    pl, pr = sol.absorbed_left[-1], sol.absorbed_right[-1]
    return sol.t, sol.flux_left / pl, sol.flux_right / pr


def _check_against_k0012():
    rows = []
    for ell, L, v, D in ((0.3, 1.0, 2.0, 0.5), (1.5, 2.0, -1.0, 1.0), (0.9, 1.0, 7.0, 1.0)):
        rows.append((f"ell={ell}, v={v}", conditional_time_to_origin(ell, L, v, D), drift_diffusion_interval(ell, L, v, D).mean_time_left))
        rows.append((f"ell={ell}, -v", conditional_time_to_origin(ell, L, -v, D), drift_diffusion_interval(ell, L, v, D).mean_time_left))
    return relative(rows, 1e-11, ref="K0012 (Srivastava et al.), same and opposite drift")


def _check_distributions():
    # the whole conditional distribution at each end is unchanged by v -> -v (any start)
    x0, L, v, D = 0.3, 1.0, 3.0, 1.0
    t, fl_p, fr_p = _conditional_densities(x0, L, v, D)
    _, fl_m, fr_m = _conditional_densities(x0, L, -v, D)
    late = t > 0.02
    scale = max(np.max(fl_p), np.max(fr_p))
    rows = [("left end, max |f(v) - f(-v)| / max f", float(np.max(np.abs(fl_p - fl_m)[late]) / scale), 0.0),
            ("right end, max |f(v) - f(-v)| / max f", float(np.max(np.abs(fr_p - fr_m)[late]) / scale), 0.0)]
    return absolute(rows, 2e-3, ref="0 (time-dependent Fokker-Planck at +v and -v)")


def _check_midpoint():
    # from the midpoint the two ends have identical conditional distributions; off-centre they do not
    L, v, D = 1.0, 3.0, 1.0
    t, fl, fr = _conditional_densities(0.5, L, v, D)
    late = t > 0.02
    mid = absolute([("midpoint: max |f_left - f_right| / max f", float(np.max(np.abs(fl - fr)[late]) / np.max(fl)), 0.0)],
                   2e-3, ref="0 (Dagdug and Berezhkovskii, Eq. 1)")
    t, fl, fr = _conditional_densities(0.3, L, v, D)
    off = float(np.max(np.abs(fl - fr)[t > 0.02]) / np.max(fl))
    off_centre = absolute([("start at 0.3: the two ends must differ (difference > 0.1)", float(off > 0.1), 1.0)], 0.0,
                          ref="1 = yes")
    return combine(mid, off_centre)


register(Entry(
    id="K0013",
    slug="first-passage-duality",
    title="First-passage duality: conditional exit times do not depend on the sign of the drift",
    area="targets",
    kind="exact",
    statement=(
        "For drift-diffusion between two absorbing points, the distribution of the exit time "
        "conditioned on leaving through a given end is the same for drift v and -v, from any "
        "starting point. From the midpoint, the conditional distributions at the two ends "
        "are therefore identical, although the exit probabilities differ. The function gives "
        "the conditional mean at the origin as printed by Krapivsky and Redner."
    ),
    model=r"dX = v\,dt + \sqrt{2D}\,dW,\quad 0<X<L,\ \text{both ends absorbing}",
    formula=(r"T(\ell)=\frac{\ell}{v}\frac{e^{-vL/D}+e^{-v\ell/D}}{e^{-v\ell/D}-e^{-vL/D}}-\frac{2L}{v}\frac{1-e^{-v\ell/D}}"
             r"{1-e^{-vL/D}}\frac{e^{-vL/D}}{e^{-v\ell/D}-e^{-vL/D}}"),
    function=conditional_time_to_origin,
    parameters={"ell": "start", "L": "length", "v": "drift away from the origin", "D": "diffusion coefficient"},
    sources=(
        Source(
            citation="P. L. Krapivsky and S. Redner, J. Stat. Mech. (2018) 093208",
            doi="10.1088/1742-5468/aaddb3",
            locator="Eq. (28b), p. 11; half-line form Eq. (4)",
            version_read="NSF public-access copy of the journal PDF, page image read 14 Oct 2026; arXiv:1807.07651v4",
            printed=(r"T(\ell)=\frac{\ell}{v}\,\frac{e^{-vL/D}+e^{-v\ell/D}}{e^{-v\ell/D}-e^{-vL/D}}-\frac{2L}{v}\,"
                     r"\frac{1-e^{-v\ell/D}}{1-e^{-vL/D}}\,\frac{e^{-vL/D}}{e^{-v\ell/D}-e^{-vL/D}}"),
            notation="same symbols",
        ),
        Source(
            citation="L. Dagdug and A. M. Berezhkovskii, J. Chem. Phys. 131, 056101 (2009)",
            doi="10.1063/1.3179679",
            locator="Eq. (1) (identity of the two conditional distributions, start midway between the ends)",
            version_read="PubMed Central PMC2736570 (MathML), opened 14 Oct 2026",
            printed=r"\varphi_{+}(\tau)=\varphi_{-}(\tau)=\varphi(\tau)",
            notation="absorbing points at x0 +- L (their L is our L/2); periodic potentials allowed",
            role="restatement",
        ),
        Source(
            citation="A. M. Berezhkovskii, G. Hummer and S. M. Bezrukov, Phys. Rev. Lett. 97, 020601 (2006)",
            doi="10.1103/PhysRevLett.97.020601",
            locator="identity of direct uphill and downhill translocation-time distributions",
            version_read="abstract only (full text paywalled)",
            role="original",
            evidence="citation-only",
        ),
    ),
    validity=("Constant drift and diffusion. The drift-sign invariance holds for any start; the "
              "equality of the two ends holds only for a start at the midpoint."),
    checks=(
        Check("conditional mean against K0012 for v and -v", "Srivastava et al. formulas (K0012)", _check_against_k0012),
        Check("whole distributions at each end unchanged by v -> -v", "time-dependent Fokker-Planck solver", _check_distributions),
        Check("midpoint: the two ends coincide; off-centre they differ", "time-dependent Fokker-Planck solver", _check_midpoint),
    ),
    aliases=("uphill and downhill first passage", "translocation time symmetry", "conditional first-passage time"),
    related=("K0012",),
    added="2026-10-14",
))
