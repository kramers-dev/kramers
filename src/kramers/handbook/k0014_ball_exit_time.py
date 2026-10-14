"""K0014. Mean and second moment of the exit time of Brownian motion from a d-dimensional ball."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .._checking import relative, statistical
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from ..solvers.quadrature import FirstPassage1D


class BallExit(NamedTuple):
    mean: float
    second_moment: float


def ball_exit_time(r: float, R: float, D: float, d: int) -> BallExit:
    """Exit time of free diffusion (coefficient D) from the ball of radius R in d dimensions, from radius r.

    Chen and Huang's Eqs. (16)-(18): mean (R^2 - r^2) / (2 d D), second moment
    (R^2 - r^2) [(d + 4) R^2 - d r^2] / (4 d^2 (d + 2) D^2).
    """
    if not (0 <= r <= R and R > 0 and D > 0 and int(d) == d and d >= 1):
        raise OutsideDomain("need 0 <= r <= R, D > 0 and an integer dimension d >= 1")
    rb = r / R
    scale = R * R / D
    mean = (1.0 - rb**2) / (2.0 * d) * scale
    second = (1.0 - rb**2) * ((d + 4) - d * rb**2) / (4.0 * d**2 * (d + 2)) * scale**2
    return BallExit(mean=mean, second_moment=second)


def _check_radial_quadrature():
    rows = []
    for d in (1, 2, 3, 5):
        for r in (0.0, 0.4, 0.9):
            # the radius is a diffusion with drift D (d - 1) / r: psi = r^(d-1), reflecting at the centre
            eps = 1e-9 if d > 1 else 0.0
            fp = FirstPassage1D(log_psi=lambda x, d=d: (d - 1) * np.log(np.maximum(x, 1e-300)), D=1.3,
                                domain=(eps, 1.0), left="reflecting", right="absorbing")
            m = fp.moments(max(r, eps), n=2).value
            b = ball_exit_time(r, 1.0, 1.3, d)
            rows.append((f"d={d}, r={r}: mean", b.mean, m[0]))
            rows.append((f"d={d}, r={r}: second moment", b.second_moment, m[1]))
    return relative(rows, 1e-7, ref="quadrature on the radial (Bessel) process")


def _check_simulation():
    rng = np.random.default_rng(41)
    d, R, D, dt, n = 3, 1.0, 1.0, 1e-4, 20_000
    pos = np.zeros((n, d))
    pos[:, 0] = 0.5
    times = np.full(n, np.nan)
    alive = np.ones(n, dtype=bool)
    t = 0.0
    while alive.any():
        t += dt
        idx = np.flatnonzero(alive)
        before = R - np.sqrt(np.einsum("ij,ij->i", pos[idx], pos[idx]))
        pos[idx] += np.sqrt(2 * D * dt) * rng.standard_normal((idx.size, d))
        after = R - np.sqrt(np.einsum("ij,ij->i", pos[idx], pos[idx]))
        # crossing within the step: the sphere treated as locally flat (bridge correction)
        bridge = rng.random(idx.size) < np.exp(-np.maximum(before * after, 0.0) / (D * dt))
        out = (after <= 0) | bridge
        times[idx[out]] = t
        alive[idx[out]] = False
    mean, se = float(times.mean()), float(times.std(ddof=1) / np.sqrt(n))
    return statistical([("d=3, r=0.5, R=1", ball_exit_time(0.5, R, D, d).mean, mean, se)], n_sigma=4.0)


register(Entry(
    id="K0014",
    slug="ball-exit-time",
    title="Exit time of Brownian motion from a ball in d dimensions: mean and second moment",
    area="targets",
    kind="exact",
    statement=(
        "A Brownian particle with diffusion coefficient D started at distance r from the "
        "centre of an absorbing sphere of radius R in d dimensions leaves after a mean time "
        "(R^2 - r^2)/(2dD); the second moment is also in closed form."
    ),
    model=r"\partial_t p = D\nabla^2 p\ \text{in}\ |x|<R\subset\mathbb{R}^d,\ p=0\ \text{on}\ |x|=R",
    formula=(r"\langle\tau\rangle=\frac{R^2-r^2}{2dD},\quad \langle\tau^2\rangle=\frac{(R^2-r^2)\left[(d+4)R^2-dr^2\right]}{4d^2(d+2)D^2}"),
    function=ball_exit_time,
    parameters={"r": "start radius", "R": "ball radius", "D": "diffusion coefficient", "d": "dimension"},
    sources=(
        Source(
            citation="H. Chen and F. Huang, Phys. Rev. E 105, 034109 (2022)",
            doi="10.1103/PhysRevE.105.034109",
            locator="Eqs. (16)-(18) (arXiv:2109.11101v2 numbering)",
            version_read="arXiv:2109.11101v2 HTML (posted after journal publication), opened 14 Oct 2026",
            printed=(r"\langle\bar{\tau}_0(\bar r)\rangle=\frac{1-\bar r^{2}}{2d};\ "
                     r"\langle\bar\tau_0^2(\bar r)\rangle=\frac{(1-\bar r^2)[(d+4)-d\bar r^2]}{4d^2(d+2)};\ "
                     r"\bar r=r/R,\ \bar\tau=\tau D/R^{2}"),
            notation="no resetting (gamma = 0)",
        ),
    ),
    validity="Free diffusion, absorbing sphere, no resetting. Exact.",
    checks=(
        Check("against the radial process by quadrature, d = 1, 2, 3, 5", "Green's-function quadrature with psi = r^(d-1)", _check_radial_quadrature),
        Check("against simulation in three dimensions", "random walk with bridge correction, 20,000 paths, dt = 1e-4", _check_simulation, slow=True),
    ),
    aliases=("mean exit time from a sphere", "Brownian motion in a ball", "narrow escape, trivial case"),
    related=("K0002",),
    added="2026-10-14",
))
