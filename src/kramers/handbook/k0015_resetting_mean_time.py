"""K0015. Mean time to reach a target for diffusion with stochastic resetting (Evans and Majumdar)."""

from __future__ import annotations

import numpy as np

from .._checking import relative, statistical
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from ..solvers import simulate


def resetting_mean_time(x0: float, r: float, D: float) -> float:
    """Mean first-passage time to the origin from x0 > 0, resetting to x0 at rate r: (exp(sqrt(r/D) x0) - 1) / r."""
    if not (x0 > 0 and r > 0 and D > 0):
        raise OutsideDomain("need x0 > 0, r > 0, D > 0 (without resetting the mean is infinite)")
    return float(np.expm1(np.sqrt(r / D) * x0) / r)


def _check_renewal():
    # the same number from the renewal formula with the Levy-Smirnov transform exp(-x0 sqrt(s/D)) (K0017)
    rows = []
    for x0, r, D in ((1.0, 1.0, 1.0), (0.3, 5.0, 0.2), (2.0, 0.01, 3.0)):
        Tt = np.exp(-x0 * np.sqrt(r / D))
        rows.append((f"x0={x0}, r={r}, D={D}", resetting_mean_time(x0, r, D), (1 - Tt) / (r * Tt)))
    return relative(rows, 1e-13, ref="Reuveni (2016) Eq. (7) with the free transform")


def _check_simulation():
    rows = []
    for x0, r, seed in ((1.0, 1.0, 51), (1.0, 2.5, 52)):
        sim = simulate.first_passage_times(lambda x: 0 * x, 1.0, x0, n_paths=40_000, dt=2e-4, t_max=500.0,
                                           absorbing=(0.0, None), reset_rate=r, seed=seed)
        m = sim.mean()
        rows.append((f"x0={x0}, r={r}, D=1", resetting_mean_time(x0, r, 1.0), m.value, m.error))
    return statistical(rows)


register(Entry(
    id="K0015",
    slug="resetting-mean-time",
    title="Diffusion with stochastic resetting: mean time to reach the target",
    area="resetting",
    kind="exact",
    statement=(
        "A Brownian particle that restarts from x0 at random times (rate r) reaches a target "
        "at the origin after a finite mean time, (exp(sqrt(r/D) x0) - 1)/r, although without "
        "resetting the mean is infinite."
    ),
    model=r"\partial_t p = D\,\partial_x^2 p - r\,p + r\,\delta(x-x_0),\ \text{absorbing at } 0",
    formula=r"T(x_0)=\frac{1}{r}\left(e^{\alpha_0x_0}-1\right),\quad \alpha_0=\sqrt{r/D}",
    function=resetting_mean_time,
    parameters={"x0": "start and reset position", "r": "resetting rate", "D": "diffusion coefficient"},
    sources=(
        Source(
            citation="M. R. Evans and S. N. Majumdar, Phys. Rev. Lett. 106, 160601 (2011)",
            doi="10.1103/PhysRevLett.106.160601",
            locator="Eq. (7); model Eq. (1) (arXiv:1102.2704v1 numbering)",
            version_read="arXiv:1102.2704v1 HTML, opened 14 Oct 2026",
            printed=r"T(x_0)=\frac{1}{r}\left(\exp(\alpha_0x_0)-1\right),\ \alpha_0=\sqrt{r/D}",
            notation="same symbols",
        ),
    ),
    validity="One dimension; Poissonian, instantaneous resetting to the starting point. Exact.",
    checks=(
        Check("against the general renewal formula", "Reuveni (2016) Eq. (7), entry K0017", _check_renewal),
        Check("against simulation", "Euler-Maruyama with Poissonian resetting, 40,000 paths", _check_simulation, slow=True),
    ),
    aliases=("stochastic resetting", "diffusion with restart", "search with resetting"),
    related=("K0016", "K0017", "K0018", "K0019"),
    added="2026-10-14",
))
