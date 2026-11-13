"""K0031. Mean time to reach an absorbing sphere for diffusion with resetting in d dimensions (Evans and Majumdar)."""

from __future__ import annotations

import numpy as np
from scipy import special

from .._checking import relative, statistical
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from ..solvers import simulate
from .k0015_resetting_mean_time import resetting_mean_time


def resetting_sphere_mean_time(R: float, a: float, r: float, D: float, d: int) -> float:
    """Mean time to reach the sphere |x| = a from distance R > a, resetting to the start at rate r, in d dimensions.

    T = [ (a/R)^nu K_|nu|(alpha0 a) / K_|nu|(alpha0 R) - 1 ] / r, with nu = 1 - d/2 and alpha0 = sqrt(r/D).
    Evaluated in logarithms with exponentially scaled Bessel functions, so it holds for large alpha0 R.
    """
    if not (int(d) == d and d >= 1):
        raise OutsideDomain("d must be a positive integer")
    if not (r > 0 and D > 0):
        raise OutsideDomain("need r > 0 and D > 0 (without resetting the mean is infinite)")
    if d >= 2 and not a > 0:
        raise OutsideDomain("in d >= 2 a point target is never reached: need a > 0")
    if not (R > a >= 0):
        raise OutsideDomain("need R > a >= 0 (start outside the sphere)")
    alpha0 = np.sqrt(r / D)
    if d == 1:                                      # K_{1/2} is elementary: the one-dimensional result at distance R - a
        return float(np.expm1(alpha0 * (R - a)) / r)
    nu = 1.0 - d / 2.0
    order = abs(nu)
    log_ratio = (nu * np.log(a / R) + np.log(special.kve(order, alpha0 * a)) - np.log(special.kve(order, alpha0 * R))
                 + alpha0 * (R - a))
    return float(np.expm1(log_ratio) / r)


def _check_one_dimension():
    # d = 1 is the one-dimensional result (K0015) at distance R - a
    rows = []
    for R, a, r, D in ((1.0, 0.5, 2.0, 1.0), (3.0, 0.2, 0.05, 0.7), (1.5, 0.0, 9.0, 0.3)):
        rows.append((f"R={R}, a={a}, r={r}, D={D}", resetting_sphere_mean_time(R, a, r, D, 1), resetting_mean_time(R - a, r, D)))
    return relative(rows, 1e-12, ref="K0015 at distance R - a")


def _check_three_dimensions():
    # In d = 3 the radial equation reduces to one dimension (u = rho q), so the transform without resetting is
    # (a/R) exp(-(R - a) sqrt(s/D)); the renewal formula (K0017) then gives (R/a exp(alpha0 (R - a)) - 1)/r.
    rows = []
    for R, a, r, D in ((1.0, 0.5, 2.0, 1.0), (4.0, 1.0, 0.3, 2.0), (1.2, 1.0, 50.0, 0.1), (30.0, 0.1, 1.0, 1.0)):
        Tt = (a / R) * np.exp(-(R - a) * np.sqrt(r / D))
        rows.append((f"R={R}, a={a}, r={r}, D={D}", resetting_sphere_mean_time(R, a, r, D, 3), (1 - Tt) / (r * Tt)))
    return relative(rows, 1e-12, ref="renewal formula with the elementary d = 3 transform")


def _check_simulation():
    rows = []
    for d, r, seed in ((2, 2.0, 311), (3, 2.0, 312), (3, 8.0, 313)):
        sim = simulate.first_passage_times(lambda x, d=d: (d - 1) / x, 1.0, 1.0, n_paths=40_000, dt=2e-4, t_max=400.0,
                                           absorbing=(0.5, None), reset_rate=r, seed=seed)
        m = sim.mean()
        rows.append((f"d={d}, R=1, a=0.5, r={r}, D=1", resetting_sphere_mean_time(1.0, 0.5, r, 1.0, d), m.value, m.error))
    return statistical(rows)


register(Entry(
    id="K0031",
    slug="resetting-sphere",
    title="Diffusion with resetting in d dimensions: mean time to reach an absorbing sphere",
    area="resetting",
    kind="exact",
    statement=(
        "A particle diffusing in d dimensions and reset to its start at rate r reaches an absorbing sphere of "
        "radius a, from distance R, after a finite mean time given by a ratio of modified Bessel functions of "
        "order |1 - d/2|. In d = 1 it is the one-dimensional resetting result; in d >= 2 it diverges as a -> 0."
    ),
    model=(r"\partial_t q = D\nabla^2 q - r\,q + r\,\delta(\mathbf{x}-\mathbf{X}_r),\ "
           r"\text{absorbing on } |\mathbf{x}|=a,\ |\mathbf{X}_r|=R"),
    formula=(r"T=\frac{1}{r}\left[\left(\frac{a}{R}\right)^{\nu}\frac{K_{|\nu|}(\alpha_0a)}{K_{|\nu|}(\alpha_0R)}-1\right],"
             r"\quad \nu=1-\frac{d}{2},\ \alpha_0=\sqrt{r/D}"),
    function=resetting_sphere_mean_time,
    parameters={"R": "distance of the start and reset point from the centre", "a": "radius of the absorbing sphere",
                "r": "resetting rate", "D": "diffusion coefficient (<x_i^2> = 2Dt)", "d": "dimension"},
    sources=(
        Source(
            citation="M. R. Evans and S. N. Majumdar, J. Phys. A: Math. Theor. 47, 285001 (2014)",
            doi="10.1088/1751-8113/47/28/285001",
            locator="Eq. (65), scaled form Eq. (67), limits Eqs. (70) and (75); symbols Eqs. (9), (11), (15), (16) (arXiv:1404.4574v1 numbering)",
            version_read="arXiv:1404.4574v1 HTML (ar5iv), opened 14 Oct and 13 Nov 2026; journal numbering not checked",
            printed=r"T(R_r)=\frac{1}{r}\left[\left(\frac{a}{R_r}\right)^{\nu}\frac{K_\nu(\alpha_0 a)}{K_\nu(\alpha_0 R_r)}-1\right]",
            notation="R_r -> R; the paper writes K_nu in Eq. (65) and K_|nu| in Eq. (67): K is even in its order (their Eq. 66)",
        ),
    ),
    validity=("Free diffusion in d dimensions, Poissonian instantaneous resetting to the starting point, perfectly "
              "absorbing sphere. Exact. In d >= 2 the target must have a > 0."),
    checks=(
        Check("d = 1 against the one-dimensional result", "entry K0015 at distance R - a", _check_one_dimension),
        Check("d = 3 against the renewal formula", "Reuveni (2016) Eq. (7) with the elementary d = 3 transform (entry K0017)",
              _check_three_dimensions),
        Check("against simulation", "Euler-Maruyama for the radial (Bessel) process with Poissonian resetting, 40,000 paths",
              _check_simulation, slow=True),
    ),
    aliases=("resetting in higher dimensions", "search with restart in 3D", "Bessel process with resetting"),
    related=("K0014", "K0015", "K0016", "K0017"),
    added="2026-11-13",
))
