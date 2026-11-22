"""K0035. Diffusion with resetting in a logarithmic potential: mean and spread of the time to reach the origin (Ray and Reuveni)."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import integrate, optimize, special

from .._checking import combine, relative, statistical
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from ..solvers import simulate
from ..solvers.quadrature import FirstPassage1D
from .k0015_resetting_mean_time import resetting_mean_time
from .k0017_restart_renewal import restart_moments


class RestartTime(NamedTuple):
    mean: float
    std: float


def _log_ratio(nu: float, y: float) -> float:
    """log[2^(nu-1) Gamma(nu) / (y^nu K_nu(y))], with K evaluated exponentially scaled."""
    return float((nu - 1.0) * np.log(2.0) + special.gammaln(nu) - nu * np.log(y) - np.log(special.kve(nu, y)) + y)


def _gamma_mean(fun, nu: float, y: float) -> float:
    """E[fun(y^2 / (4 G))] for G ~ Gamma(nu, 1): the passage time without resetting is x0^2/(4 D G) (Eq. 5).

    Integrated over v = ln G, where the integrand e^(nu v - e^v) fun(c e^-v) / Gamma(nu), c = y^2/4, is smooth
    and decays exponentially at both ends; the interval is split at the two scales v = ln c and v = ln nu.
    """
    c = y * y / 4.0
    lg = special.gammaln(nu)

    def integrand(v: float) -> float:
        return float(np.exp(nu * v - np.exp(v) - lg) * fun(c * np.exp(-v)))

    v_lo = np.log(c) - 60.0 / nu
    v_hi = np.log(max(800.0, 40.0 * nu))
    knots = sorted({v_lo, np.log(c), np.log(nu), v_hi})
    knots = [k for k in knots if v_lo <= k <= v_hi]
    return float(sum(integrate.quad(integrand, a, b, epsabs=0.0, epsrel=1e-13, limit=400)[0]
                     for a, b in zip(knots[:-1], knots[1:], strict=True)))


def _one_minus_exp(w: float) -> float:
    return float(-np.expm1(-w))


def _g2(w: float) -> float:
    # 1 - e^-w (1 + w), without cancellation at small w
    if w < 1e-3:
        return w * w / 2 - w**3 / 3 + w**4 / 8 - w**5 / 30
    return float(-np.expm1(-w) - w * np.exp(-w))


def log_potential_restart(x0: float, beta_U0: float, D: float, r: float) -> RestartTime:
    """Mean and standard deviation of the time for a particle in U(x) = U0 ln|x| to reach the origin from x0 > 0,
    resetting to x0 at rate r.

    With nu = (1 + beta U0)/2 and y = sqrt(r/D) x0 (Eqs. 29 and 31):
    mean = [2^(nu-1) Gamma(nu) / (y^nu K_nu(y)) - 1]/r,
    std = sqrt(2^(nu-1) Gamma(nu) [2^(nu-1) Gamma(nu) - y^(1+nu) K_(nu-1)(y)] / (y^(2 nu) K_nu(y)^2) - 1)/r.
    beta U0 > -1 is required (otherwise the particle is pushed away and need not reach the origin);
    beta U0 = 0 is free diffusion (entry K0015). <x^2> = 2 D t for U0 = 0.

    For y >= 0.5 the printed Bessel forms are evaluated in logarithms. For y < 0.5, where they lose
    digits (the bracket tends to 1), the same moments are computed from the renewal formula with the
    transform written as an expectation over the Gamma-distributed variable of Eq. (5): 1 - T~(r) =
    E[1 - e^-w] and r T~'(r) - T~(r) + 1 = E[1 - e^-w (1 + w)], w = y^2/(4G), which has no cancellation.
    """
    if not (x0 > 0 and D > 0 and r > 0):
        raise OutsideDomain("need x0 > 0, D > 0 and r > 0")
    if not beta_U0 > -1.0:
        raise OutsideDomain("need beta U0 > -1: a strongly repulsive log potential keeps the particle from the origin")
    nu = 0.5 * (1.0 + beta_U0)
    y = float(np.sqrt(r / D) * x0)
    if y >= 0.5:
        lr = _log_ratio(nu, y)
        mean = float(np.expm1(lr) / r)
        # second bracket: A y^(1-nu) K_(nu-1)(y) / K_nu(y)^2 with A = 2^(nu-1) Gamma(nu), in logarithms
        lt = ((nu - 1.0) * np.log(2.0) + special.gammaln(nu) + (1.0 - nu) * np.log(y)
              + np.log(special.kve(nu - 1.0, y)) - 2.0 * np.log(special.kve(nu, y)) + y)
        q = np.exp(2.0 * lr) - np.exp(lt) - 1.0
        return RestartTime(mean=mean, std=float(np.sqrt(max(q, 0.0)) / r))
    one_minus_T = _gamma_mean(_one_minus_exp, nu, y)
    Tt = 1.0 - one_minus_T
    mean = one_minus_T / (r * Tt)
    second = 2.0 * _gamma_mean(_g2, nu, y) / (r * r * Tt * Tt)
    return RestartTime(mean=float(mean), std=float(np.sqrt(max(second - mean * mean, 0.0))))


def optimal_reset_rate(x0: float, beta_U0: float, D: float) -> float:
    """The resetting rate that minimises the mean (found numerically; 0 when beta U0 >= 5, where resetting cannot help)."""
    if not beta_U0 < 5.0:
        return 0.0

    def f(lz: float) -> float:
        return log_potential_restart(x0, beta_U0, D, D * np.exp(2 * lz) / x0**2).mean

    res = optimize.minimize_scalar(f, bounds=(np.log(1e-3), np.log(30.0)), method="bounded",
                                   options={"xatol": 1e-12})
    return float(D * np.exp(2 * res.x) / x0**2)


def _transform(x0: float, nu: float, D: float):
    # Laplace transform of the passage time without resetting, Eq. (25)
    return lambda s: float(np.exp(-_log_ratio(nu, np.sqrt(s * x0 * x0 / D))))


def _check_renewal():
    rows = []
    for nu, r in ((0.25, 0.5), (1.5, 0.3), (2.5, 1.0), (4.0, 0.2)):
        x0, D = 2.0, 1.0
        res = log_potential_restart(x0, 2 * nu - 1, D, r)
        mom = restart_moments(_transform(x0, nu, D), r)
        rows.append((f"mean, nu={nu}, r={r}", res.mean, mom.mean))
        rows.append((f"second moment, nu={nu}, r={r}", res.std**2 + res.mean**2, mom.second_moment))
    return relative(rows, 1e-8, ref="Reuveni (2016) Eq. (7) with the transform of Eq. (25) (entry K0017)")


def _check_free():
    rows = [(f"beta U0 = 0, x0={x0}, D={D}, r={r}", log_potential_restart(x0, 0.0, D, r).mean, resetting_mean_time(x0, r, D))
            for x0, D, r in ((2.0, 1.0, 0.8), (0.7, 0.3, 3.0))]
    return relative(rows, 1e-12, ref="entry K0015")


def _check_no_reset_limit():
    # r -> 0 for beta U0 > 1: the mean time to reach the origin without resetting, from the quadrature solver
    # with the potential beta U0 ln x on (eps, infinity), absorbing at eps (the origin is a singular point)
    rows = []
    # beta U0 >= 4 (nu >= 5/2), where the second moment without resetting is finite and the mean approaches its
    # r = 0 value linearly in r; for 1 < beta U0 < 3 the approach is like r^(nu - 1), too slow for this test
    for bU0 in (4.0, 6.0, 9.0):
        x0, D = 2.0, 1.0
        fp = FirstPassage1D(lambda x, b=bU0: b * np.log(x), D=D, domain=(1e-9, np.inf), left="absorbing", right="natural")
        rows.append((f"beta U0 = {bU0:g}, r = 1e-10", log_potential_restart(x0, bU0, D, 1e-10).mean, float(fp.mean_time(x0).value)))
    lim = relative(rows, 1e-8, ref="quadrature solver without resetting")
    # the two evaluations meet at y = 0.5: the Bessel form just above against the expectation form just below
    cont = []
    for bU0 in (-0.5, 0.0, 2.0, 4.0, 7.0):
        x0, D = 2.0, 1.0
        r_hi, r_lo = (0.5 * (1 + 1e-9) / x0) ** 2, (0.5 * (1 - 1e-9) / x0) ** 2
        a, b = log_potential_restart(x0, bU0, D, r_hi), log_potential_restart(x0, bU0, D, r_lo)
        cont.append((f"mean at y = 0.5, beta U0 = {bU0}", a.mean, b.mean))
        cont.append((f"std at y = 0.5, beta U0 = {bU0}", a.std, b.std))
    return combine(lim, relative(cont, 1e-8, ref="the other evaluation"))


def _check_optimum():
    # at beta U0 = 0 the optimum is Evans and Majumdar's z/2 = 1 - exp(-z) (entry K0016's condition)
    zs = optimize.brentq(lambda z: z / 2 - 1 + np.exp(-z), 0.5, 3.0, xtol=1e-15)
    x0, D = 2.0, 1.0
    part = relative([("beta U0 = 0", optimal_reset_rate(x0, 0.0, D), D * zs**2 / x0**2)], 1e-6, ref="z/2 = 1 - e^-z")
    # resetting helps exactly when the coefficient of variation without it exceeds 1, i.e. beta U0 < 5 (Eq. 34)
    signs = []
    for bU0 in (4.5, 5.5):
        d = log_potential_restart(x0, bU0, D, 1e-4).mean - log_potential_restart(x0, bU0, D, 1e-6).mean
        signs.append((f"beta U0 = {bU0}: mean {'falls' if bU0 < 5 else 'rises'} with small r", float(np.sign(d)),
                      -1.0 if bU0 < 5 else 1.0))
    return combine(part, relative(signs, 1e-15, ref="the CV = 1 criterion"))


def _event_driven(x0: float, beta_U0: float, D: float, r: float, n: int, seed: int) -> np.ndarray:
    # exact sampling, no time step: between resets the particle is a Bessel process, whose time to reach the
    # origin from x0 is x0^2 / (4 D G) with G ~ Gamma(nu, 1) (the classical hitting law, Eq. 5 of the paper);
    # resets come at exponential times, and each one starts a fresh attempt from x0
    rng = np.random.default_rng(seed)
    nu = 0.5 * (1.0 + beta_U0)
    total = np.zeros(n)
    alive = np.arange(n)
    while alive.size:
        hit = x0 * x0 / (4.0 * D * rng.gamma(nu, 1.0, alive.size))
        reset = rng.exponential(1.0 / r, alive.size)
        done = hit < reset
        total[alive[done]] += hit[done]
        total[alive[~done]] += reset[~done]
        alive = alive[~done]
    return total


def _check_simulation():
    rows = []
    # attractive potential: Euler-Maruyama with the drift -D beta U0 / x and bridge-corrected absorption
    x0, D, r, bU0 = 1.0, 1.0, 1.0, 4.0
    res = log_potential_restart(x0, bU0, D, r)
    sim = simulate.first_passage_times(lambda x: -D * bU0 / x, D, x0, n_paths=100_000, dt=1e-4, t_max=60.0,
                                       absorbing=(0.0, None), reset_rate=r, seed=352)
    m, m2 = sim.mean(), sim.second_moment()
    rows.append(("Langevin, mean, beta U0 = 4", res.mean, m.value, m.error))
    rows.append(("Langevin, second moment, beta U0 = 4", res.std**2 + res.mean**2, m2.value, m2.error))
    # repulsive and weakly attractive: exact event-driven sampling. (Euler-Maruyama is not used here: with a
    # repulsive 1/x drift at the absorbing origin its bias falls only like sqrt(dt); measured 22 Nov 2026 at
    # beta U0 = -0.5: 3.81 at dt = 1e-4 and 3.91 at 2.5e-5 against the exact 4.005.)
    for bU0, seed in ((-0.5, 353), (2.5, 354)):
        res = log_potential_restart(x0, bU0, D, r)
        t = _event_driven(x0, bU0, D, r, 200_000, seed)
        n = t.size
        rows.append((f"event-driven, mean, beta U0 = {bU0}", res.mean, float(t.mean()), float(t.std(ddof=1) / np.sqrt(n))))
        t2 = t * t
        rows.append((f"event-driven, second moment, beta U0 = {bU0}", res.std**2 + res.mean**2, float(t2.mean()),
                     float(t2.std(ddof=1) / np.sqrt(n))))
    return statistical(rows)


register(Entry(
    id="K0035",
    slug="log-potential-restart",
    title="Resetting in a logarithmic potential: mean and spread of the time to the origin",
    area="resetting",
    kind="exact",
    statement=(
        "A particle diffusing in the potential U0 ln|x| and reset to x0 at rate r reaches the origin after a mean "
        "time [2^(nu-1) Gamma(nu)/(y^nu K_nu(y)) - 1]/r, with nu = (1 + beta U0)/2 and y = sqrt(r/D) x0. "
        "Resetting shortens the mean only for beta U0 < 5, where the spread of the time without resetting exceeds "
        "its mean."
    ),
    model=(r"\partial_t p=\partial_x\!\left[\frac{U_0}{\zeta x}p\right]+D\,\partial_x^2p-r\,p+r\,\delta(x-x_0)\,Q(t),"
           r"\ p(0,t)=0,\ D=k_BT/\zeta"),
    formula=(r"\langle T_r\rangle=\frac{1}{r}\left[\frac{2^{\nu-1}\Gamma(\nu)}{(\alpha_0x_0)^{\nu}K_\nu(\alpha_0x_0)}-1\right],"
             r"\quad \nu=\frac{1+\beta U_0}{2},\ \alpha_0=\sqrt{r/D}"),
    function=log_potential_restart,
    parameters={"x0": "start and reset position (> 0)", "beta_U0": "strength of the potential U0 ln|x| in units of kT (> -1)",
                "D": "diffusion coefficient (<x^2> = 2Dt for U0 = 0)", "r": "resetting rate"},
    sources=(
        Source(
            citation="S. Ray and S. Reuveni, J. Chem. Phys. 152, 234110 (2020)",
            doi="10.1063/5.0010549",
            locator="Eq. (29) p. 6 (mean); Eq. (31) p. 7 (standard deviation); Eq. (2) (nu); Eq. (25) (transform); "
                    "Eqs. (32)-(34) (no resetting) (arXiv:2004.01898v2 numbering)",
            version_read="arXiv:2004.01898v2 TeX source and PDF, opened 22 Nov 2026 (two readers); journal numbering not checked",
            printed=(r"\left<T_r\right>=\frac{1}{r}\left[\frac{2^{\nu-1}\Gamma(\nu)}{(\alpha_0 x_0)^{\nu}K_{\nu}(\alpha_0 x_0 )}-1\right];\ "
                     r"\sigma(T_r)=\frac{1}{r}\sqrt{\frac{2^{\nu-1}\Gamma(\nu)\left[2^{\nu-1}\Gamma(\nu)-(\alpha_0 x_0)^{1+\nu}"
                     r"K_{\nu-1}(\alpha_0 x_0 )\right]}{(\alpha_0 x_0)^{2\nu}[K_{\nu}(\alpha_0 x_0 )]^2}-1}"),
            notation=("same symbols; beta U0 enters through nu only. The optimal rate is found here by minimising the mean "
                      "numerically, not from the paper's Eq. (37): as printed in arXiv v2, the root of Eq. (37) does not "
                      "match the numerical minimum (at nu = 1/2 it gives z0 = 0.60 against 1.594); a factor K_nu(z0)^2 on "
                      "its first term restores agreement. Read as a question for the authors until the journal version is "
                      "checked"),
        ),
    ),
    validity=("One dimension, the half-line x > 0 with the origin absorbing, potential U0 ln|x| with beta U0 > -1, "
              "Poissonian instantaneous resetting to the start. Exact. Without resetting the mean is finite only for "
              "beta U0 > 1 and the spread only for beta U0 > 3."),
    checks=(
        Check("against the renewal formula", "Reuveni (2016) Eq. (7) with the transform of Eq. (25) (entry K0017)", _check_renewal),
        Check("free diffusion", "entry K0015 at beta U0 = 0", _check_free),
        Check("the limit without resetting", "quadrature solver with the logarithmic potential", _check_no_reset_limit),
        Check("the optimal rate and the transition at beta U0 = 5", "numerical minimisation; entry K0016's condition",
              _check_optimum),
        Check("against simulation", "Euler-Maruyama with the 1/x drift (attractive), and exact event-driven sampling of "
              "the Bessel hitting law with exponential resets (repulsive and weakly attractive)", _check_simulation, slow=True),
    ),
    aliases=("resetting in a log potential", "Bessel process with resetting", "resetting transition", "restart with a central force"),
    related=("K0015", "K0016", "K0017", "K0018"),
    added="2026-11-22",
))
