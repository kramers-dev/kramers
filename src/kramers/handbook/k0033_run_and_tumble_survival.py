"""K0033. Universal survival probability of a run-and-tumble particle in d dimensions (Mori, Le Doussal, Majumdar, Schehr)."""

from __future__ import annotations

import numpy as np
from scipy import special

from .._checking import combine, relative, statistical
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain


def run_and_tumble_survival(t: float, gamma: float) -> float:
    """Probability that the x-component of a run-and-tumble particle started at the origin stays positive up to t.

    S(t) = exp(-gamma t/2) [I0(gamma t/2) + I1(gamma t/2)] / 2, the same in every dimension and for any speed
    distribution; evaluated with exponentially scaled Bessel functions.
    """
    if not (t >= 0 and gamma > 0):
        raise OutsideDomain("need t >= 0 and gamma > 0")
    z = gamma * t / 2.0
    return float(0.5 * (special.i0e(z) + special.i1e(z)))


def _x_cosines(rng: np.random.Generator, d: int, n: int) -> np.ndarray:
    # x-component of a direction drawn uniformly on the unit sphere in d dimensions
    if d == 1:
        return rng.choice((-1.0, 1.0), size=n)
    g = rng.standard_normal((n, d))
    return g[:, 0] / np.linalg.norm(g, axis=1)


def _death_times(d: int, gamma: float, v0: float, n: int, t_max: float, seed: int) -> np.ndarray:
    # Event-driven: within a straight run x is linear, so x reaches 0 inside the run exactly when the run would
    # end below 0; no time step is involved.
    rng = np.random.default_rng(seed)
    death = np.full(n, np.inf)
    x = np.zeros(n)
    t = np.zeros(n)
    alive = np.ones(n, dtype=bool)
    first = True
    while alive.any():
        idx = np.flatnonzero(alive)
        c = _x_cosines(rng, d, idx.size)
        tau = rng.exponential(1.0 / gamma, idx.size)
        vx = v0 * c
        if first:                                   # start at x = 0: a particle moving into x < 0 is absorbed at once
            hit = vx < 0
            dt_hit = np.zeros(idx.size)
            first = False
        else:
            end = x[idx] + vx * tau
            hit = end < 0
            dt_hit = np.where(hit, x[idx] / np.where(vx < 0, -vx, 1.0), 0.0)
        death[idx[hit]] = t[idx[hit]] + dt_hit[hit]
        x[idx] += vx * tau
        t[idx] += tau
        alive[idx[hit]] = False
        alive &= t < t_max
    return death


def _check_simulation():
    gamma, v0, times, n = 1.3, 1.0, (0.5, 1.0, 5.0, 20.0), 200_000
    rows = []
    for d, seed in ((1, 331), (2, 332), (3, 333)):
        death = _death_times(d, gamma, v0, n, max(times), seed)
        for tt in times:
            p = float(np.mean(death > tt))
            rows.append((f"d={d}, t={tt}", run_and_tumble_survival(tt, gamma), p, np.sqrt(p * (1 - p) / n)))
    return statistical(rows, ref="event-driven simulation")


def _check_limits():
    start = relative([("S(0)", run_and_tumble_survival(0.0, 1.3), 0.5)], 1e-15, ref="1/2")
    rows = [(f"gamma t = {g * t:g}", run_and_tumble_survival(t, g), 1 / np.sqrt(np.pi * g * t)) for g, t in ((1.3, 1e4), (0.2, 1e5))]
    return combine(start, relative(rows, 1e-3, ref="1/sqrt(pi gamma t)"))


register(Entry(
    id="K0033",
    slug="run-and-tumble-survival",
    title="Run-and-tumble particle in d dimensions: universal survival probability",
    area="active",
    kind="exact",
    statement=(
        "For a run-and-tumble particle that starts at the origin in any dimension, tumbles at constant rate gamma "
        "into an isotropic new direction and runs at any speed distribution, the probability that its x-component "
        "has stayed positive up to t is exp(-gamma t/2)[I0(gamma t/2) + I1(gamma t/2)]/2: independent of d and of "
        "the speed, starting at 1/2 and decaying as 1/sqrt(pi gamma t)."
    ),
    model=(r"\dot{\mathbf{x}}=v\,\mathbf{n}(t),\ \mathbf{n}\ \text{redrawn isotropically at rate}\ \gamma,\ "
           r"\mathbf{x}(0)=0;\ S(t)=\Pr[x_1(t')>0\ \forall t'\le t]"),
    formula=r"S(t)=\frac{1}{2}e^{-\gamma t/2}\left[I_0\!\left(\frac{\gamma t}{2}\right)+I_1\!\left(\frac{\gamma t}{2}\right)\right]",
    function=run_and_tumble_survival,
    parameters={"t": "time", "gamma": "tumble rate (each tumble draws a new isotropic direction)"},
    sources=(
        Source(
            citation="F. Mori, P. Le Doussal, S. N. Majumdar and G. Schehr, Phys. Rev. Lett. 124, 090603 (2020)",
            doi="10.1103/PhysRevLett.124.090603",
            locator="Eq. (1) (arXiv:2001.01492v1 numbering)",
            version_read="arXiv:2001.01492v1 HTML (ar5iv), opened 14 Oct and 13 Nov 2026; journal numbering not checked",
            printed=r"S(t)=\frac{1}{2}e^{-\gamma t/2}\left(I_{0}\left(\gamma t/2\right)+I_{1}\left(\gamma t/2\right)\right)",
            notation="same symbols; in d = 1 gamma is the tumble rate with a fresh random direction, not the reversal rate",
        ),
    ),
    validity=("Exponential run times (constant tumble rate), isotropic new directions, start at the origin with a "
              "random direction, no translational diffusion. Exact in every dimension and for any speed distribution."),
    checks=(
        Check("S(0) = 1/2 and the large-time law", "the limits stated with Eq. (1)", _check_limits),
        Check("against event-driven simulation", "exact piecewise-straight runs in d = 1, 2, 3, 200,000 particles each",
              _check_simulation, slow=True),
    ),
    aliases=("active particle persistence", "run-and-tumble first passage", "universal survival"),
    related=("K0020", "K0023"),
    added="2026-11-13",
))
