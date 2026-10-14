"""K0030. Switching of the noisy fold under a linear ramp, solved without the quasi-static approximation."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .._checking import absolute, statistical
from ..registry import Check, Entry, register
from ..result import OutsideDomain
from ..solvers import fokker_planck, simulate
from .k0008_fold_decay_rate import fold_decay_rate


@lru_cache(maxsize=256)
def _solve(sigma: float, u0: float, y_escape: float, n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    left = -np.sqrt(max(u0, 0.0)) - 6.0
    t_end = (u0 + 6.0) / sigma
    sol = fokker_planck.solve(lambda y, s: y * y - (u0 - sigma * s), 1.0, (left, y_escape), left="reflecting",
                              right="absorbing", t_end=t_end, n_cells=n_cells, dt=t_end / 3000.0,
                              p0=lambda y: np.exp(-((y + np.sqrt(u0)) ** 2) * np.sqrt(u0)), time_dependent=True)
    return sol.t, sol.survival


def ramped_fold_switching(sigma: float, levels: tuple[float, ...] = (0.1, 0.5, 0.9), u0: float = 7.0,
                          y_escape: float = 9.0, n_cells: int = 1500) -> np.ndarray:
    """Distance to the fold, u, at which the fractions ``levels`` of paths have escaped.

    Model (scaled): dy = (y^2 - u(s)) ds + sqrt(2) dW with u(s) = u0 - sigma s; paths start
    in the quasi-stationary well (a Gaussian around -sqrt(u0)) and escape when y reaches
    ``y_escape``. Negative values are past the fold. Computed by the time-dependent
    Fokker-Planck solver; no fitted coefficient and no quasi-static assumption.
    """
    if not (sigma > 0 and u0 >= 2.0 and y_escape >= np.sqrt(u0) + 2):
        raise OutsideDomain("need sigma > 0, u0 >= 2 and an escape point well past the saddle")
    t, S = _solve(float(sigma), float(u0), float(y_escape), int(n_cells))
    lv = np.asarray(levels, dtype=float)
    absorbed = 1.0 - S
    if np.any(lv >= absorbed[-1]):
        raise OutsideDomain("not all requested fractions escaped within the run")
    t_q = np.interp(lv, absorbed, t)
    return u0 - sigma * t_q


def _check_simulation():
    sigma, u0, yc = 0.1, 7.0, 9.0
    sim = simulate.first_passage_times(lambda y, s: y * y - (u0 - sigma * s), 1.0, -np.sqrt(u0), n_paths=6000,
                                       dt=5e-4, t_max=(u0 + 6.0) / sigma, absorbing=(None, yc),
                                       time_dependent=True, seed=121)
    times = np.sort(sim.times)
    rows = []
    pred = ramped_fold_switching(sigma, (0.25, 0.5, 0.75), u0, yc)
    for q, p in zip((0.25, 0.5, 0.75), pred, strict=True):
        k = int(q * times.size)
        # standard error of a quantile, from the order-statistics interval of +-1 sigma
        lo = times[max(0, k - int(np.sqrt(times.size * q * (1 - q))))]
        hi = times[min(times.size - 1, k + int(np.sqrt(times.size * q * (1 - q))))]
        u_sim = u0 - sigma * times[k]
        se = sigma * (hi - lo) / 2.0
        rows.append((f"u at {q:.2f} escaped, sigma={sigma}", float(p), float(u_sim), float(se)))
    return statistical(rows, n_sigma=4.0)


def _check_slow_ramp():
    # at a slow ramp the median approaches the quasi-static one (rate K0008, which does not depend on y_escape)
    sigma, u0 = 0.002, 7.0
    med = float(ramped_fold_switching(sigma, (0.5,), u0)[0])
    u = np.linspace(u0, 0.0, 4001)
    lam = fold_decay_rate(u)
    H = np.concatenate([[0.0], np.cumsum(0.5 * (lam[1:] + lam[:-1]) * (-np.diff(u)))]) / sigma
    qs = float(np.interp(np.log(2.0), H, u))
    return absolute([(f"median u, sigma={sigma}", med, qs)], 0.01, ref="quasi-static median from the decay rate (K0008)")


register(Entry(
    id="K0030",
    slug="ramped-fold-switching",
    title="Switching of the noisy fold under a linear ramp, solved exactly (computed reference)",
    area="driven",
    kind="computed-reference",
    statement=(
        "When the control parameter of a noisy saddle-node is ramped towards the fold at any "
        "speed, the distribution of the distance at which paths escape, computed from the "
        "time-dependent Fokker-Planck equation with no quasi-static assumption and no fitted "
        "coefficient. At fast ramps the quasi-static (Kramers or exact-rate) answer is off by "
        "tens of per cent and many paths escape only after the fold."
    ),
    model=r"dy=\left(y^2-u_0+\Sigma s\right)ds+\sqrt2\,dW,\ \text{escape at } y=y_c",
    formula=r"S(s)=\int p(y,s)\,dy,\quad \partial_s p=-\partial_y[(y^2-u(s))p]+\partial_y^2p,\ p(y_c,s)=0",
    function=ramped_fold_switching,
    parameters={"sigma": "scaled ramp rate", "levels": "escaped fractions", "u0": "starting distance",
                "y_escape": "escape point", "n_cells": "grid size"},
    sources=(),
    validity="Scaled ramp rates from about 1e-3 to 1 and beyond; the solver's own error is below 1e-3 in u.",
    checks=(
        Check("quantiles against direct simulation", "Euler-Maruyama, 6,000 paths, the same escape criterion", _check_simulation, slow=True),
        Check("slow ramp: approaches the quasi-static median", "decay rate of the frozen fold (K0008)", _check_slow_ramp, slow=True),
    ),
    aliases=("rate-induced delay of tipping", "dynamic bifurcation with noise", "ramped saddle-node",
             "switching under a fast ramp", "tipping under a ramp"),
    related=("K0008", "K0024"),
    notes=("Replaces the lag coefficient fitted to simulations in geoews 0.3 (dev branch). A "
           "prototype of 12 October 2026 matched the author's simulated medians (0.748, 0.611, "
           "0.487, 0.338 at Sigma = 0.08, 0.10, 0.122, 0.15; statistical errors 0.003 to 0.005) to "
           "within 0.005 after converting between the two escape criteria."),
    added="2026-10-14",
))
