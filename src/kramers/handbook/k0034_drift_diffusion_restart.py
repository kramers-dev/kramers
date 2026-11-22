"""K0034. Drift-diffusion to a target under Poissonian restart: mean, spread and the Peclet transition (Ray, Mondal, Reuveni)."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import optimize

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


def _s_minus_V(V: float, D: float, r: float) -> tuple[float, float]:
    """sqrt(V^2 + 4 D r) and sqrt(V^2 + 4 D r) - V, the second without cancellation."""
    s = float(np.sqrt(V * V + 4.0 * D * r))
    return s, (4.0 * D * r / (s + V) if V > 0 else s - V)


def drift_diffusion_restart(L: float, V: float, D: float, r: float) -> RestartTime:
    """Mean and standard deviation of the time to reach a target at distance L, with drift V towards it,
    restarting from the start at rate r.

    mean = (e^z - 1)/r and std = sqrt(e^(2z) - 2 e^z L r / sqrt(V^2 + 4 D r) - 1)/r, with
    z = L (sqrt(V^2 + 4 D r) - V) / (2 D). <x^2> = 2 D t without drift; V < 0 is drift away
    from the target. The standard deviation is evaluated in a rearranged form that keeps its
    digits as r -> 0 (where it tends to the spread of the passage time without restart).
    """
    if not (L > 0 and D > 0 and r > 0):
        raise OutsideDomain("need L > 0, D > 0 and r > 0 (for r = 0 the time is the passage time without restart)")
    s, smv = _s_minus_V(V, D, r)
    z = L * smv / (2.0 * D)
    E = float(np.expm1(z))
    a = L * r / s
    z_minus_a = L * r * smv / (s * (s + V)) if s + V > 0 else z - a
    F = z * z / 2 + z**3 / 6 + z**4 / 24 + z**5 / 120 if z < 1e-3 else E - z  # e^z - 1 - z
    r2var = E * E + 2.0 * (z_minus_a + F - a * E)  # = e^(2z) - 2 a e^z - 1, rearranged
    return RestartTime(mean=E / r, std=float(np.sqrt(max(r2var, 0.0)) / r))


def optimal_restart_rate(L: float, V: float, D: float) -> float:
    """The restart rate that minimises the mean time, from the transcendental Eq. (17); 0 when Pe = L V/(2 D) >= 1.

    [(z - 1) e^z + 1] Pe = [(1 - z/2) e^z - 1] z, then r* = (V/L) z* + (D/L^2) z*^2 (Eq. 14).
    """
    if not (L > 0 and D > 0):
        raise OutsideDomain("need L > 0 and D > 0")
    Pe = L * V / (2.0 * D)
    if Pe >= 1.0:
        return 0.0

    def g(z: float) -> float:
        # both sides divided by e^z (no overflow at large z), written with expm1 so the sign is right at small z:
        # (1 - z/2 - e^-z) z - (z - 1 + e^-z) Pe
        em = float(np.expm1(-z))
        return (-em - z / 2) * z - (z + em) * Pe

    hi = 2.0
    while g(hi) > 0:
        hi *= 2.0
    zs = optimize.brentq(g, 1e-8, hi, xtol=1e-15, rtol=1e-15, maxiter=500)
    return float((V / L) * zs + (D / L**2) * zs * zs)


def _transform(L: float, V: float, D: float):
    # Laplace transform of the passage time without restart, Eq. (10)
    return lambda s: float(np.exp(-L * (np.sqrt(V * V + 4.0 * D * s) - V) / (2.0 * D)))


def _check_renewal():
    rows = []
    for L, V, D, r in ((1.0, 0.5, 1.0, 0.7), (1.0, -0.8, 0.5, 2.0), (2.0, 3.0, 0.3, 0.1), (1.0, 0.0, 1.0, 1.3)):
        res = drift_diffusion_restart(L, V, D, r)
        mom = restart_moments(_transform(L, V, D), r)
        rows.append((f"mean, L={L}, V={V}, D={D}, r={r}", res.mean, mom.mean))
        rows.append((f"second moment, L={L}, V={V}, D={D}, r={r}", res.std**2 + res.mean**2, mom.second_moment))
    return relative(rows, 1e-8, ref="Reuveni (2016) Eq. (7) with the drift-diffusion transform (entry K0017)")


def _check_limits():
    free = relative([(f"V = 0, L={L}, D={D}, r={r}", drift_diffusion_restart(L, 0.0, D, r).mean, resetting_mean_time(L, r, D))
                     for L, D, r in ((1.3, 0.7, 0.9), (0.4, 2.0, 5.0))], 1e-12, ref="entry K0015")
    # r -> 0 with drift towards the target: mean and spread of the passage time without restart, from the
    # quadrature solver (a linear potential -V x / D with a natural end at -infinity)
    rows = []
    for L, V, D in ((1.0, 0.5, 1.0), (2.0, 1.7, 0.4)):
        res = drift_diffusion_restart(L, V, D, 1e-9)
        fp = FirstPassage1D(lambda x, V=V, D=D: -(V / D) * x, D=D, domain=(-np.inf, L), left="natural", right="absorbing")
        rows.append((f"mean, L={L}, V={V}, D={D}, r=1e-9", res.mean, float(fp.mean_time(0.0).value)))
        rows.append((f"std, L={L}, V={V}, D={D}, r=1e-9", res.std, float(np.sqrt(fp.variance(0.0).value))))
    return combine(free, relative(rows, 1e-5, ref="r -> 0 limit"))


def _check_optimum():
    rows = []
    for L, V, D in ((1.0, 0.0, 1.0), (1.0, 0.4, 1.0), (2.0, 0.5, 2.0), (1.0, -1.0, 1.0)):
        rstar = optimal_restart_rate(L, V, D)
        res = optimize.minimize_scalar(lambda lr, L=L, V=V, D=D: drift_diffusion_restart(L, V, D, float(np.exp(lr))).mean,
                                       bracket=(np.log(rstar) - 1, np.log(rstar), np.log(rstar) + 1), tol=1e-12)
        rows.append((f"Pe = {L * V / (2 * D):g}", rstar, float(np.exp(res.x))))
    part = relative(rows, 1e-6, ref="numerical minimum of the mean")
    # Pe = 1.5: no restart rate helps; the mean rises from L/V as soon as r > 0
    L, V, D = 1.0, 3.0, 1.0
    m = [drift_diffusion_restart(L, V, D, r).mean for r in (1e-6, 1e-3, 1e-1, 1.0)]
    rising = relative([("Pe = 1.5: optimal rate is 0", optimal_restart_rate(L, V, D) + 1.0, 1.0),
                       ("Pe = 1.5: mean increases with r", float(np.all(np.diff(m) > 0)), 1.0)], 1e-15, ref="expected")
    return combine(part, rising)


def _check_simulation():
    rows = []
    for V, seed in ((0.5, 341), (-0.8, 342)):
        L, D, r = 1.0, 1.0, 0.7
        sim = simulate.first_passage_times(lambda x, V=V: V + 0 * x, D, 0.0, n_paths=100_000, dt=2e-4, t_max=200.0,
                                           absorbing=(None, L), reset_rate=r, seed=seed)
        m, m2 = sim.mean(), sim.second_moment()
        res = drift_diffusion_restart(L, V, D, r)
        rows.append((f"mean, V={V}", res.mean, m.value, m.error))
        rows.append((f"second moment, V={V}", res.std**2 + res.mean**2, m2.value, m2.error))
    return statistical(rows)


register(Entry(
    id="K0034",
    slug="drift-diffusion-restart",
    title="Drift-diffusion to a target under restart: mean, spread and the Peclet transition",
    area="resetting",
    kind="exact",
    statement=(
        "A particle that diffuses with drift V towards a target at distance L, and is returned to its start at "
        "rate r, reaches the target after a mean time (e^z - 1)/r with z = L(sqrt(V^2 + 4Dr) - V)/(2D). Restart "
        "can shorten the mean only when the Peclet number LV/(2D) is below 1; above it the best rate is zero."
    ),
    model=(r"\partial_t p=-V\partial_x p+D\partial_x^2 p-r\,p+r\,\delta(x)\,Q(t),\ x\in(-\infty,L),\ "
           r"p(L,t)=0,\ x(0)=0"),
    formula=(r"\langle T_r\rangle=\frac{e^{z}-1}{r},\quad \sigma(T_r)=\frac{1}{r}\sqrt{e^{2z}-2e^{z}\frac{Lr}{\sqrt{V^2+4Dr}}-1},"
             r"\quad z=\frac{L}{2D}\left(\sqrt{V^2+4Dr}-V\right)"),
    function=drift_diffusion_restart,
    parameters={"L": "distance from the start (and restart point) to the target", "V": "drift towards the target (negative: away)",
                "D": "diffusion coefficient (<x^2> = 2Dt)", "r": "restart rate"},
    sources=(
        Source(
            citation="S. Ray, D. Mondal and S. Reuveni, J. Phys. A: Math. Theor. 52, 255002 (2019)",
            doi="10.1088/1751-8121/ab1fcc",
            locator="Eq. (6) p. 3 (mean); Eq. (23) p. 6 with Eq. (13) (standard deviation); Eq. (17) p. 4 with "
                    "Eqs. (12), (14) (optimal rate); model Eq. (1) (arXiv:1811.08239v4 numbering)",
            version_read="arXiv:1811.08239v4 TeX source and PDF, opened 22 Nov 2026 (two readers); journal numbering not checked",
            printed=(r"\left<T_r\right>=\frac{\exp{\left[{\frac{L}{2D}\left(\sqrt{V^2+4Dr}-V\right)}\right]}-1}{r};\ "
                     r"\sigma(T_r)=\frac{\sqrt{\exp{(2z)}-2\exp{(z)}\frac{Lr}{\sqrt{4 Dr+V^2}}-1}}{r};\ "
                     r"\left[\left(z-1\right)\exp{(z)}+1\right]Pe=\left[\left(1-\frac{z}{2}\right)\exp{(z)}-1\right]z"),
            notation="same symbols; Pe = LV/(2D); the target is at x = L, the start and restart point at x = 0",
        ),
    ),
    validity=("One dimension, constant drift and diffusion, an absorbing target and no other boundary, Poissonian "
              "instantaneous restart to the start. Exact. V < 0 (drift away) is allowed for r > 0."),
    checks=(
        Check("against the renewal formula", "Reuveni (2016) Eq. (7) with the drift-diffusion transform (entry K0017)",
              _check_renewal),
        Check("the limits V = 0 and r -> 0", "entry K0015; the quadrature solver without restart", _check_limits),
        Check("the optimal rate and the transition at Pe = 1", "numerical minimisation of the mean", _check_optimum),
        Check("against simulation", "Euler-Maruyama with Poissonian restart, 100,000 paths, drift towards and away",
              _check_simulation, slow=True),
    ),
    aliases=("restart with drift", "Peclet transition", "stochastic resetting with bias", "first passage under restart with drift"),
    related=("K0015", "K0016", "K0017", "K0018"),
    added="2026-11-22",
))
