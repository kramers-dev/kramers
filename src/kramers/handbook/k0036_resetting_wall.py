"""K0036. Diffusion with resetting between an absorbing end and a reflecting wall: mean time and the critical start point (Ahmad et al.)."""

from __future__ import annotations

import numpy as np
from scipy import optimize

from .._checking import combine, relative, statistical
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from ..solvers import simulate
from ..solvers.quadrature import FirstPassage1D
from .k0017_restart_renewal import restart_moments

CRITICAL_START = 1.0 - 1.0 / np.sqrt(5.0)  # Eq. (25): resetting helps only for x0/L below this


def _log_cosh(z: float) -> float:
    return float(abs(z) + np.log1p(np.exp(-2.0 * abs(z))) - np.log(2.0))


def _log_sinh(z: float) -> float:
    return float(z + np.log(-np.expm1(-2.0 * z)) - np.log(2.0))


def resetting_wall_mean_time(x0: float, L: float, D: float, r: float) -> float:
    """Mean time to reach an absorbing end at 0 from x0, with a reflecting wall at L and resetting to x0 at rate r.

    <T_r> = [cosh(a L) / cosh(a (L - x0)) - 1] / r with a = sqrt(r/D) (Eq. 26). Evaluated through
    cosh u - cosh v = 2 sinh((u+v)/2) sinh((u-v)/2), in logarithms, so there is no cancellation as
    r -> 0 and no overflow for large r. <x^2> = 2 D t.
    """
    if not (0 < x0 < L):
        raise OutsideDomain("need 0 < x0 < L")
    if not (D > 0 and r > 0):
        raise OutsideDomain("need D > 0 and r > 0 (for r = 0 the mean is x0 (2L - x0) / (2D))")
    a = np.sqrt(r / D)
    u, v = a * L, a * (L - x0)
    log_num = np.log(2.0) + _log_sinh(0.5 * (u + v)) + _log_sinh(0.5 * (u - v))
    return float(np.exp(log_num - _log_cosh(v)) / r)


def optimal_reset_rate_wall(x0: float, L: float, D: float) -> float:
    """The resetting rate that minimises the mean, as the root of Eq. (27); 0 when x0/L >= 1 - 1/sqrt(5).

    Eq. (27): L sinh(aL) - (L - x0) tanh(a(L - x0)) cosh(aL) = (2/a) [cosh(aL) - cosh(a(L - x0))], a = sqrt(r*/D).
    Both sides are divided by cosh(aL) before solving, so large a does not overflow.
    """
    if not (0 < x0 < L and D > 0):
        raise OutsideDomain("need 0 < x0 < L and D > 0")
    if x0 / L >= CRITICAL_START:
        return 0.0

    def f(a: float) -> float:
        ratio = np.exp(_log_cosh(a * (L - x0)) - _log_cosh(a * L))
        return L * np.tanh(a * L) - (L - x0) * np.tanh(a * (L - x0)) - (2.0 / a) * (1.0 - ratio)

    hi = 1.0 / L
    while f(hi) <= 0:
        hi *= 2.0
    lo = hi / 2.0
    while f(lo) > 0 and lo > 1e-12:
        lo /= 2.0
    a = optimize.brentq(f, lo, hi, xtol=1e-15, rtol=1e-15)
    return float(D * a * a)


def _transform(x0: float, L: float, D: float):
    # Laplace transform of the passage time without resetting (text above Eq. 25)
    return lambda s: float(np.cosh(np.sqrt(s / D) * (L - x0)) / np.cosh(np.sqrt(s / D) * L))


def _check_renewal():
    rows = []
    for x0, L, D, r in ((0.5, 1.0, 0.5, 0.5), (0.5, 1.0, 0.5, 3.0), (0.2, 1.5, 1.0, 0.05), (0.9, 1.0, 2.0, 7.0)):
        rows.append((f"x0={x0}, L={L}, D={D}, r={r}", resetting_wall_mean_time(x0, L, D, r),
                     restart_moments(_transform(x0, L, D), r).mean))
    return relative(rows, 1e-12, ref="Reuveni (2016) Eq. (7) with the wall transform (entry K0017)")


def _check_no_reset_and_transition():
    # without resetting: mean and variance from the quadrature solver; then the start point where the
    # standard deviation equals the mean (the transition of Eq. 25), found with the solver alone
    rows = []
    for x0, L, D in ((0.5, 1.0, 0.5), (0.3, 2.0, 1.3)):
        fp = FirstPassage1D(lambda x: 0 * x, D=D, domain=(0.0, L), left="absorbing", right="reflecting")
        rows.append((f"r -> 0, x0={x0}, L={L}, D={D}", resetting_wall_mean_time(x0, L, D, 1e-10), float(fp.mean_time(x0).value)))
    lim = relative(rows, 1e-8, ref="quadrature solver without resetting")
    fp = FirstPassage1D(lambda x: 0 * x, D=1.0, domain=(0.0, 1.0), left="absorbing", right="reflecting")

    def excess(k: float) -> float:
        return float(np.sqrt(fp.variance(k).value) / fp.mean_time(k).value - 1.0)

    kc = optimize.brentq(excess, 0.3, 0.8, xtol=1e-13)
    return combine(lim, relative([("critical start x0/L", CRITICAL_START, kc)], 1e-8, ref="variance = mean^2, quadrature"))


def _check_optimum():
    rows = []
    for x0, L, D in ((0.5, 1.0, 1.0), (0.2, 1.0, 0.5), (1.0, 3.0, 2.0)):
        rstar = optimal_reset_rate_wall(x0, L, D)
        res = optimize.minimize_scalar(lambda lr, x0=x0, L=L, D=D: resetting_wall_mean_time(x0, L, D, float(np.exp(lr))),
                                       bracket=(np.log(rstar) - 1, np.log(rstar), np.log(rstar) + 1), tol=1e-12)
        rows.append((f"x0={x0}, L={L}, D={D}", rstar, float(np.exp(res.x))))
    part = relative(rows, 1e-6, ref="numerical minimum of the mean")
    # above the critical start point the mean rises with r from the start
    m = [resetting_wall_mean_time(0.6, 1.0, 1.0, r) for r in (1e-6, 1e-3, 0.1, 1.0)]
    above = relative([("x0/L = 0.6 > 1 - 1/sqrt 5: mean increases with r", float(np.all(np.diff(m) > 0)), 1.0)], 1e-15,
                     ref="expected")
    return combine(part, above)


def _check_simulation():
    rows = []
    for r, seed in ((0.5, 361), (3.0, 362)):
        x0, L, D = 0.5, 1.0, 0.5
        sim = simulate.first_passage_times(lambda x: 0 * x, D, x0, n_paths=100_000, dt=1e-4, t_max=60.0,
                                           absorbing=(0.0, None), reflecting=(None, L), reset_rate=r, seed=seed)
        m = sim.mean()
        rows.append((f"x0={x0}, L={L}, D={D}, r={r}", resetting_wall_mean_time(x0, L, D, r), m.value, m.error))
    return statistical(rows)


register(Entry(
    id="K0036",
    slug="resetting-wall",
    title="Resetting between an absorbing end and a reflecting wall: mean time and the critical start point",
    area="resetting",
    kind="exact",
    statement=(
        "A Brownian particle between an absorbing end at 0 and a reflecting wall at L, reset to its start x0 at rate r, "
        "is absorbed after a mean time [cosh(aL)/cosh(a(L - x0)) - 1]/r with a = sqrt(r/D). Resetting shortens the "
        "mean only when the particle starts closer to the target than x0/L = 1 - 1/sqrt(5) = 0.553."
    ),
    model=r"\partial_t p=D\,\partial_x^2p-r\,p+r\,\delta(x-x_0)\,Q(t),\ x\in(0,L),\ p(0,t)=0,\ \partial_xp(L,t)=0",
    formula=(r"\langle T_r\rangle=\frac{1}{r}\left[\frac{\cosh(\sqrt{r/D}\,L)}{\cosh(\sqrt{r/D}\,(L-x_0))}-1\right],\quad "
             r"\frac{x_0}{L}<1-\frac{1}{\sqrt5}\ \text{for resetting to help}"),
    function=resetting_wall_mean_time,
    parameters={"x0": "start and reset position, 0 < x0 < L", "L": "position of the reflecting wall (target at 0)",
                "D": "diffusion coefficient (<x^2> = 2Dt)", "r": "resetting rate"},
    sources=(
        Source(
            citation="S. Ahmad, I. Nayak, A. Bansal, A. Nandi and D. Das, Phys. Rev. E 99, 022130 (2019)",
            doi="10.1103/PhysRevE.99.022130",
            locator="Sec. III 3 (box potential, n = infinity): Eq. (26) p. 5 (mean), Eq. (25) (critical point), "
                    "Eq. (27) (optimal rate), the transform and moments in the text above Eq. (25) (arXiv:1902.02955v1 numbering)",
            version_read="arXiv:1902.02955v1 TeX source and PDF p. 5, opened 22 Nov 2026 (two readers); journal numbering not checked",
            printed=(r"\langle T_r \rangle = \frac{1}{r}\bigg[\frac{\cosh(\sqrt{\frac{r}{D}} L)}{\cosh(\sqrt{\frac{r}{D}} "
                     r"(L - x_{0}))}-1\bigg];\ 5 K_c^2 - 10 K_c + 4 = 0 \Rightarrow K_c = 1 - \frac{1}{\sqrt{5}}"),
            notation="same symbols; K = x0/L; the wall is the n -> infinity limit of the paper's potential k0 (x/L)^n",
        ),
    ),
    validity=("One dimension, free diffusion between an absorbing end and a reflecting wall, Poissonian instantaneous "
              "resetting to the start. Exact. Not the two-absorbing-end interval of K0032."),
    checks=(
        Check("against the renewal formula", "Reuveni (2016) Eq. (7) with the wall transform (entry K0017)", _check_renewal),
        Check("no resetting, and the critical start point", "quadrature solver: mean, variance, and variance = mean^2",
              _check_no_reset_and_transition),
        Check("the optimal rate", "numerical minimisation of the mean", _check_optimum),
        Check("against simulation", "Euler-Maruyama with a reflecting wall and resetting, 100,000 paths", _check_simulation,
              slow=True),
    ),
    aliases=("resetting with a reflecting wall", "resetting in a box", "critical start point 1 - 1/sqrt 5",
             "vanishing of the optimal resetting rate"),
    related=("K0015", "K0016", "K0017", "K0032"),
    added="2026-11-22",
))
