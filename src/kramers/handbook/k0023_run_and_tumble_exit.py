"""K0023. Mean exit time of a run-and-tumble particle from an interval."""

from __future__ import annotations

import numpy as np

from .._checking import statistical
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain


def run_and_tumble_exit_time(L: float, v: float, alpha: float) -> float:
    """Mean time for a run-and-tumble particle started at 0 moving right to reach L.

    Reflecting wall at 0, absorbing at L, speed v, tumble rate alpha (new direction drawn
    at random, so the direction reverses at rate alpha / 2), no translational diffusion:
    L / v + L^2 / (2 D0) with D0 = v^2 / alpha (Angelani et al. Eq. 11).
    """
    if not (L > 0 and v > 0 and alpha >= 0):
        raise OutsideDomain("need L > 0, v > 0, alpha >= 0")
    return float(L / v + L * L * alpha / (2.0 * v * v))


def _simulate(L: float, v: float, alpha: float, n: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    s = np.ones(n)  # direction
    t = np.zeros(n)
    done = np.zeros(n, dtype=bool)
    rate = alpha / 2.0  # reversals
    while not done.all():
        idx = np.flatnonzero(~done)
        dt = rng.exponential(1.0 / rate, idx.size) if rate > 0 else np.full(idx.size, np.inf)
        xi, si = x[idx], s[idx]
        right = si > 0
        # moving right: absorbed if it reaches L before the next reversal
        reach = right & (xi + v * dt >= L)
        t[idx[reach]] += (L - xi[reach]) / v
        done[idx[reach]] = True
        go = right & ~reach
        x[idx[go]] = xi[go] + v * dt[go]
        t[idx[go]] += dt[go]
        s[idx[go]] = -1.0
        # moving left: elastic reflection at 0 (zero flux), then onwards to the right
        left = ~right
        xl, dl = xi[left], dt[left]
        bounce = v * dl > xl
        newx = np.where(bounce, v * dl - xl, xl - v * dl)
        news = np.where(bounce, 1.0, 1.0)  # a reversal at the end of the run makes it move right
        # after a bounce it moves right and may reach L before the run ends
        il = idx[left]
        hitL = bounce & (newx >= L)
        t_hit = xl / v + L / v
        t[il[hitL]] += t_hit[hitL]
        done[il[hitL]] = True
        rest = ~hitL
        # a bounce turns it right; the reversal at the end of the run turns it left again
        x[il[rest]] = newx[rest]
        t[il[rest]] += dl[rest]
        s[il[rest]] = np.where(bounce[rest], -1.0, news[rest])
    return float(t.mean()), float(t.std(ddof=1) / np.sqrt(n))


def _check_simulation():
    rows = []
    for L, v, alpha, seed in ((1.0, 1.0, 2.0, 91), (2.0, 0.5, 0.3, 92), (1.0, 3.0, 20.0, 93)):
        m, se = _simulate(L, v, alpha, 100_000, seed)
        rows.append((f"L={L}, v={v}, alpha={alpha}", run_and_tumble_exit_time(L, v, alpha), m, se))
    return statistical(rows)


register(Entry(
    id="K0023",
    slug="run-and-tumble-exit",
    title="Mean exit time of a run-and-tumble particle from an interval",
    area="active",
    kind="exact",
    statement=(
        "A run-and-tumble particle (speed v, tumble rate alpha) starting at a reflecting wall "
        "and moving away from it reaches an absorbing point at distance L after a mean time "
        "L/v + L^2/(2 D0), the ballistic time plus a diffusive time with D0 = v^2/alpha."
    ),
    model=r"\partial_tP_R=-\partial_x(vP_R)-\tfrac{\alpha}{2}P_R+\tfrac{\alpha}{2}P_L,\ \text{mirror for } P_L;\ J(0)=0,\ P_L(L)=0",
    formula=r"\tau=\frac{L}{v}+\frac{L^2}{2D_0},\quad D_0=\frac{v^2}{\alpha}",
    function=run_and_tumble_exit_time,
    parameters={"L": "distance to the absorbing point", "v": "swimming speed", "alpha": "tumble rate"},
    sources=(
        Source(
            citation="L. Angelani, R. Di Leonardo and M. Paoluzzi, Eur. Phys. J. E 37, 59 (2014)",
            doi="10.1140/epje/i2014-14059-4",
            locator="Eq. (11); model Eqs. (1)-(2) (arXiv:1506.08046v1 numbering)",
            version_read="arXiv:1506.08046v1 HTML (posted after publication), opened 14 Oct 2026",
            printed=r"\tau_{free}=\frac{L}{v}+\frac{L^{2}}{2D_{0}},\ D_0=v^2/\alpha",
            notation="alpha: tumble rate with a random new direction (reversal rate alpha/2)",
        ),
    ),
    validity="One dimension, constant speed and tumble rate, no translational diffusion, start at the wall moving inwards. Exact.",
    checks=(
        Check("against event-driven simulation", "exact piecewise-deterministic simulation, 100,000 particles", _check_simulation),
    ),
    aliases=("active particle first passage", "telegraph process", "bacterial run and tumble"),
    related=(),
    added="2026-10-14",
))
