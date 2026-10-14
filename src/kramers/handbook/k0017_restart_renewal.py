"""K0017. Mean and second moment of any first-passage process under Poissonian restart."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from .._checking import relative, statistical
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from ..solvers import fokker_planck, simulate


class RestartMoments(NamedTuple):
    mean: float
    second_moment: float


def restart_moments(laplace_fpt: Callable[[float], float], r: float, h: float | None = None) -> RestartMoments:
    """Moments of the completion time under restart at rate r, from the Laplace transform T~(s) of the
    first-passage-time density without restart (Reuveni's Eq. 7).

    mean = (1 - T~(r)) / (r T~(r)); second = 2 (r T~'(r) - T~(r) + 1) / (r^2 T~(r)^2).
    The derivative is taken by a central difference with step ``h`` (default 1e-4 r).
    """
    if not r > 0:
        raise OutsideDomain("the restart rate must be positive")
    Tr = float(laplace_fpt(r))
    if not 0 < Tr <= 1:
        raise OutsideDomain("T~(r) must lie in (0, 1]")
    step = 1e-4 * r if h is None else h
    dT = (8 * (laplace_fpt(r + step) - laplace_fpt(r - step)) - (laplace_fpt(r + 2 * step) - laplace_fpt(r - 2 * step))) / (12 * step)
    mean = (1.0 - Tr) / (r * Tr)
    second = 2.0 * (r * float(dT) - Tr + 1.0) / (r * r * Tr * Tr)
    return RestartMoments(mean=mean, second_moment=second)


def _inverse_gaussian_transform(x0: float, v: float, D: float) -> Callable[[float], float]:
    """Laplace transform of the first passage of drift-diffusion (drift v towards the target)."""
    return lambda s: float(np.exp(x0 * (v - np.sqrt(v * v + 4.0 * D * s)) / (2.0 * D)))


def _check_numerical_transform():
    # T~(r) from the first-passage density computed by the Fokker-Planck solver, in the mean formula
    x0, v, D, r = 1.0, 0.5, 1.0, 0.8
    sol = fokker_planck.solve(lambda x: -v + 0 * x, D, (0.0, 40.0), left="absorbing", right="reflecting", x0=x0,
                              t_end=200.0, n_cells=4000, dt=0.05, dt_min=1e-6)
    f = sol.flux_left
    Tr_num = float(np.trapezoid(np.exp(-r * sol.t) * f, sol.t))
    exact = _inverse_gaussian_transform(x0, v, D)
    rows = [("T~(r) itself", Tr_num, exact(r)),
            ("mean under restart", (1 - Tr_num) / (r * Tr_num), restart_moments(exact, r).mean)]
    return relative(rows, 1e-4, ref="formula with the closed-form transform")


def _check_simulation():
    x0, v, D, r = 1.0, 0.3, 1.0, 0.7
    sim = simulate.first_passage_times(lambda x: -v + 0 * x, D, x0, n_paths=40_000, dt=2e-4, t_max=400.0,
                                       absorbing=(0.0, None), reset_rate=r, seed=61)
    m, m2 = sim.mean(), sim.second_moment()
    pred = restart_moments(_inverse_gaussian_transform(x0, v, D), r)
    return statistical([("mean", pred.mean, m.value, m.error), ("second moment", pred.second_moment, m2.value, m2.error)])


def _check_free_case():
    x0, D = 1.3, 0.6
    rows = []
    for r in (0.2, 1.0, 4.0):
        mom = restart_moments(lambda s: np.exp(-x0 * np.sqrt(s / D)), r)
        rows.append((f"r={r}", mom.mean, float(np.expm1(np.sqrt(r / D) * x0) / r)))
    return relative(rows, 1e-12, ref="Evans and Majumdar, K0015")


register(Entry(
    id="K0017",
    slug="restart-renewal",
    title="Any first-passage process under Poissonian restart: mean and second moment",
    area="resetting",
    kind="exact",
    statement=(
        "If a process that completes after a random time T is restarted from scratch at rate "
        "r, the mean completion time is (1 - T~(r))/(r T~(r)), where T~ is the Laplace "
        "transform of the density of T. The second moment also needs only T~ and its slope."
    ),
    model=r"\text{any process with completion time } T,\ \tilde T(s)=\langle e^{-sT}\rangle,\ \text{restart at rate } r",
    formula=(r"\langle T_r\rangle=\frac{1-\tilde T(r)}{r\,\tilde T(r)},\quad "
             r"\langle T_r^2\rangle=\frac{2\left(r\,\tilde T'(r)-\tilde T(r)+1\right)}{r^2\tilde T(r)^2}"),
    function=restart_moments,
    parameters={"laplace_fpt": "s -> T~(s)", "r": "restart rate", "h": "step for the derivative"},
    sources=(
        Source(
            citation="S. Reuveni, Phys. Rev. Lett. 116, 170601 (2016)",
            doi="10.1103/PhysRevLett.116.170601",
            locator="Eq. (7) (arXiv:1512.01600v1 numbering); transform under restart Eq. (6)",
            version_read="arXiv:1512.01600v1 HTML, opened 14 Oct 2026",
            printed=(r"\langle T_r\rangle=\frac{1-\tilde{T}(r)}{\tilde{T}(r)}\frac{1}{r};\ "
                     r"\langle T_r^2\rangle=\frac{2\left(r\frac{d\tilde{T}(r)}{dr}-\tilde{T}(r)+1\right)}{r^2\tilde{T}(r)^2}"),
            notation="same symbols",
        ),
        Source(
            citation="A. Pal and S. Reuveni, Phys. Rev. Lett. 118, 030603 (2017)",
            doi="10.1103/PhysRevLett.118.030603",
            locator="Eq. (3) and the constant-rate formula after it (arXiv:1607.06048v3)",
            version_read="arXiv:1607.06048v3 HTML, opened 14 Oct 2026",
            role="restatement",
        ),
    ),
    validity="Any first-passage process; restarts at exponentially distributed times, instantaneous, from scratch. Exact.",
    checks=(
        Check("free diffusion reproduces Evans and Majumdar", "K0015", _check_free_case),
        Check("with a transform computed numerically", "first-passage density from the Fokker-Planck solver", _check_numerical_transform),
        Check("drift-diffusion against simulation", "Euler-Maruyama with resetting, 40,000 paths", _check_simulation, slow=True),
    ),
    aliases=("first passage under restart", "renewal formula for resetting", "restart of randomized algorithms"),
    related=("K0015", "K0018", "K0019"),
    added="2026-10-14",
))
