"""K0019. Mean completion time under sharp (periodic) restart."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import quad
from scipy.special import erf

from .._checking import relative, statistical
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from ..solvers import simulate


def sharp_restart_mean_time(survival: Callable[[float], float], period: float) -> float:
    """Mean completion time when the process restarts every ``period``: int_0^period S dt / (1 - S(period)).

    ``survival`` is S(t), the probability that the process without restart has not
    completed by t.
    """
    if not period > 0:
        raise OutsideDomain("the period must be positive")
    S_end = float(survival(period))
    if not 0 <= S_end < 1:
        raise OutsideDomain("S(period) must be below 1: some completions must happen within a period")
    area = quad(lambda t: float(survival(t)), 0.0, period, epsabs=0.0, epsrel=1e-12, limit=200)[0]
    return float(area / (1.0 - S_end))


def _free(x0: float, D: float) -> Callable[[float], float]:
    return lambda t: float(erf(x0 / np.sqrt(4.0 * D * t))) if t > 0 else 1.0


def _check_simulation():
    rows = []
    for x0, tau, seed in ((1.0, 0.8, 71), (1.0, 2.0, 72)):
        sim = simulate.first_passage_times(lambda x: 0 * x, 1.0, x0, n_paths=40_000, dt=2e-4, t_max=400.0,
                                           absorbing=(0.0, None), reset_period=tau, seed=seed)
        m = sim.mean()
        rows.append((f"x0={x0}, period={tau}", sharp_restart_mean_time(_free(x0, 1.0), tau), m.value, m.error))
    return statistical(rows)


def _check_optimum_condition():
    # Pal, Kundu and Evans Eq. (66): at the optimal period, mean = -S(t*) / S'(t*)
    x0, D = 1.0, 1.0
    S = _free(x0, D)
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda tau: sharp_restart_mean_time(S, tau), bounds=(0.1, 5.0), method="bounded",
                          options={"xatol": 1e-10})
    t_star = res.x
    dS = -x0 / np.sqrt(4 * np.pi * D * t_star**3) * np.exp(-x0 * x0 / (4 * D * t_star))
    return relative([("mean at the optimal period", sharp_restart_mean_time(S, t_star), -S(t_star) / dS)], 1e-6,
                    ref="Pal, Kundu and Evans (2016) Eq. (66)")


register(Entry(
    id="K0019",
    slug="sharp-restart",
    title="Mean completion time under sharp (periodic) restart",
    area="resetting",
    kind="exact",
    statement=(
        "If a process is restarted deterministically every tau, its mean completion time is the "
        "integral of its survival probability up to tau, divided by the probability of "
        "completing within one period."
    ),
    model=r"\text{any process with survival } S(t),\ \text{restart every } \tau",
    formula=r"\langle T_\tau\rangle=\frac{\int_0^{\tau}S(t)\,dt}{1-S(\tau)}",
    function=sharp_restart_mean_time,
    parameters={"survival": "S(t) without restart", "period": "restart period tau"},
    sources=(
        Source(
            citation="A. Pal, A. Kundu and M. R. Evans, J. Phys. A 49, 225001 (2016)",
            doi="10.1088/1751-8113/49/22/225001",
            locator="Eq. (65); optimality condition Eq. (66) (arXiv:1512.08211v2 numbering)",
            version_read="arXiv:1512.08211v2 HTML, opened 14 Oct 2026",
            printed=r"T(x_0)=\frac{\int_{0}^{t^{*}}dt~q(x_0,t)}{1-q(x_0,t^{*})}",
            notation="q = S, t* = tau",
        ),
        Source(
            citation="A. Pal and S. Reuveni, Phys. Rev. Lett. 118, 030603 (2017)",
            doi="10.1103/PhysRevLett.118.030603",
            locator="inline statement <min(T, t)>/Pr(T < t), and Eq. (4)",
            version_read="arXiv:1607.06048v3 HTML, opened 14 Oct 2026",
            role="restatement",
        ),
    ),
    validity="Any process; instantaneous restart from scratch every tau. Exact.",
    checks=(
        Check("the optimality condition at the best period", "Pal, Kundu and Evans Eq. (66)", _check_optimum_condition),
        Check("free diffusion against simulation", "Euler-Maruyama with periodic resetting, 40,000 paths", _check_simulation, slow=True),
    ),
    aliases=("periodic resetting", "deterministic restart"),
    related=("K0015", "K0017"),
    added="2026-10-14",
))
