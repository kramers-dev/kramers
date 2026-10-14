"""The time-dependent Fokker-Planck equation in one dimension, with absorbing ends.

Model (Ito)::

    dX = A(X, t) dt + sqrt(2 D) dW,        a < X < b,

so that the density obeys dp/dt = -d/dx [A p - D dp/dx]. The drift may change in time
(a ramp of a control parameter), which no stationary formula covers.

Numerics (computed checks in ``tests/test_fokker_planck.py``): finite volumes with
Scharfetter-Gummel fluxes, which are exact for a constant flux through a cell with
linear potential and conserve probability; TR-BDF2 in time (second order, L-stable).
Absorbing ends are Dirichlet conditions on the boundary face; the outward fluxes there
are the first-passage-time densities through each end. The probability absorbed at
each end is taken from the scheme's own discrete balance, so survival plus absorbed
probability is one to rounding error.

Limits. TR-BDF2 does not preserve positivity: with time steps large against h^2 / D
at early times (a point start on a fine grid), the density and the fluxes can dip
below zero for a few steps. The default time grid starts at steps of ``dt / 1000`` to
avoid this; ``be_start`` replaces the first steps with backward Euler, which is
positive but first order. Richardson extrapolation (the default in :func:`solve`) acts
on survival, fluxes and absorbed probabilities only; an under-resolved coarse grid can
then give survival slightly above one or small negative fluxes, and
``info["richardson_change"]`` reports the size of the correction. Density snapshots
are from the fine grid and are not extrapolated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.linalg.lapack import dgtsv

from ..result import OutsideDomain

Array = NDArray[np.float64]

__all__ = ["FPSolution", "solve"]

_GAMMA = 2.0 - np.sqrt(2.0)


def _bernoulli(z: Array) -> Array:
    """B(z) = z / (exp(z) - 1), with B(0) = 1."""
    out = np.ones_like(z)
    small = np.abs(z) < 1e-6
    zs = z[small]
    out[small] = 1.0 - zs / 2.0 + zs * zs / 12.0
    big = ~small
    zb = z[big]
    with np.errstate(over="ignore"):
        out[big] = zb / np.expm1(zb)
    out[big & (z > 700)] = 0.0
    return out


@dataclass(frozen=True)
class FPSolution:
    """Output of :func:`solve`.

    ``survival[k]`` is the probability that the process has not been absorbed by
    ``t[k]``; ``flux_left`` and ``flux_right`` are the outward fluxes (first-passage
    densities through each end); ``absorbed_left`` and ``absorbed_right`` their running
    integrals; ``density`` holds snapshots at ``t[snapshots]`` if requested.
    """

    t: Array
    survival: Array
    flux_left: Array
    flux_right: Array
    absorbed_left: Array
    absorbed_right: Array
    x: Array
    density: Array | None = None
    snapshots: NDArray[np.intp] | None = None
    info: dict = field(default_factory=dict)

    @property
    def density_of_first_passage(self) -> Array:
        return self.flux_left + self.flux_right

    def quantile(self, q: float | Array) -> float | Array:
        """Time by which a fraction q has been absorbed (interpolated)."""
        q_arr = np.atleast_1d(np.asarray(q, dtype=float))
        absorbed = 1.0 - self.survival
        if np.any(q_arr > absorbed[-1]):
            raise OutsideDomain("the run ended before that fraction was absorbed; increase t_end")
        out = np.interp(q_arr, absorbed, self.t)
        return float(out[0]) if np.ndim(q) == 0 else out

    def mean_time(self) -> float:
        """Integral of the survival probability, with an exponential tail beyond t_end."""
        s = self.survival
        area = float(np.trapezoid(s, self.t))
        if s[-1] > 1e-14:
            k = max(4, s.size // 20)
            rate = -np.log(s[-1] / s[-k]) / (self.t[-1] - self.t[-k])
            if not rate > 0:
                raise OutsideDomain("survival did not decay by t_end; increase t_end")
            area += s[-1] / rate
        return area


def _operator(A_faces: Array, D: float, h: float, left: str, right: str, A_left: float, A_right: float) -> tuple[Array, Array, Array]:
    """Tridiagonal M with dp/dt = M p: (sub, diag, sup), plus the boundary coefficients."""
    n = A_faces.size + 1
    pe = A_faces * h / D
    bp = _bernoulli(pe)  # B(P)
    bm = _bernoulli(-pe)  # B(-P)
    c = D / (h * h)
    diag = np.zeros(n)
    diag[:-1] -= c * bm
    diag[1:] -= c * bp
    sup = c * bp  # row i, column i + 1
    sub = c * bm  # row i + 1, column i
    if right == "absorbing":
        pe_r = A_right * (h / 2.0) / D
        diag[-1] -= (2.0 * D / (h * h)) * _bernoulli(np.array([-pe_r]))[0]
    if left == "absorbing":
        pe_l = A_left * (h / 2.0) / D
        diag[0] -= (2.0 * D / (h * h)) * _bernoulli(np.array([pe_l]))[0]
    return sub, diag, sup


def _boundary_coefficients(D: float, h: float, left: str, right: str, A_left: float, A_right: float) -> tuple[float, float]:
    """Outward flux per unit density in the end cells."""
    cl = cr = 0.0
    if left == "absorbing":
        cl = float((2.0 * D / h) * _bernoulli(np.array([A_left * (h / 2.0) / D]))[0])
    if right == "absorbing":
        cr = float((2.0 * D / h) * _bernoulli(np.array([-A_right * (h / 2.0) / D]))[0])
    return cl, cr


def _step_solve(sub: Array, diag: Array, sup: Array, scale: float, rhs: Array) -> Array:
    """Solve (I - scale * M) p = rhs for tridiagonal M."""
    _, _, _, x, info = dgtsv(-scale * sub, 1.0 - scale * diag, -scale * sup, rhs)
    if info != 0:
        raise RuntimeError(f"tridiagonal solve failed (info={info})")
    return x


def _apply(sub: Array, diag: Array, sup: Array, p: Array) -> Array:
    out = diag * p
    out[:-1] += sup * p[1:]
    out[1:] += sub * p[:-1]
    return out


def _time_grid(t_end: float, dt_min: float, dt_max: float, growth: float) -> Array:
    steps = [0.0]
    dt = dt_min
    while steps[-1] < t_end:
        steps.append(min(steps[-1] + dt, t_end))
        dt = min(dt * growth, dt_max)
    return np.asarray(steps)


def _solve_once(
    drift: Callable[..., Array],
    D: float,
    domain: tuple[float, float],
    *,
    left: str,
    right: str,
    t_end: float,
    x0: float | None = None,
    p0: Callable[[Array], Array] | Array | None = None,
    n_cells: int = 2000,
    dt: float | None = None,
    dt_min: float | None = None,
    growth: float = 1.01,
    time_dependent: bool = False,
    n_snapshots: int = 0,
    be_start: int = 0,
) -> FPSolution:
    """Evolve the density from a point ``x0`` (or a density ``p0``) to ``t_end``.

    Parameters
    ----------
    drift
        A(x) or, with ``time_dependent=True``, A(x, t).
    D
        Constant diffusion coefficient.
    domain
        Finite (a, b).
    left, right
        "absorbing" or "reflecting".
    dt
        Largest time step; by default ``t_end / 2000``. Steps start at ``dt_min``
        (default ``dt / 1000``) and grow by the factor ``growth`` per step, so that a
        point start is resolved and long times are cheap.
    """
    a, b = float(domain[0]), float(domain[1])
    if not (np.isfinite(a) and np.isfinite(b) and a < b):
        raise OutsideDomain("the Fokker-Planck solver needs a finite domain a < b")
    for side in (left, right):
        if side not in {"absorbing", "reflecting"}:
            raise OutsideDomain("boundaries are 'absorbing' or 'reflecting'")
    if D <= 0:
        raise OutsideDomain("D must be positive")
    h = (b - a) / n_cells
    x = a + h * (np.arange(n_cells) + 0.5)
    faces = a + h * np.arange(1, n_cells)
    if p0 is None:
        if x0 is None or not a <= x0 <= b:
            raise OutsideDomain("give x0 inside the domain, or p0")
        if (x0 == a and left == "absorbing") or (x0 == b and right == "absorbing"):
            raise OutsideDomain("a start on an absorbing end is absorbed at once; start inside the domain")
        p = np.zeros(n_cells)
        pos = (x0 - x[0]) / h
        i = int(np.clip(np.floor(pos), 0, n_cells - 2))
        theta = float(np.clip(pos - i, 0.0, 1.0))
        p[i] += (1.0 - theta) / h
        p[i + 1] += theta / h
    else:
        p = np.asarray(p0(x) if callable(p0) else p0, dtype=float).copy()
        p /= h * p.sum()
    dt_max = float(dt) if dt is not None else t_end / 2000.0
    dt_min = min(dt_max, float(dt_min) if dt_min is not None else 1e-3 * dt_max)
    t_grid = _time_grid(t_end, dt_min, dt_max, growth)
    n_steps = t_grid.size - 1

    def op(t: float) -> tuple[Array, Array, Array, float, float]:
        if time_dependent:
            Af = np.asarray(drift(faces, t), dtype=float)
            Al = float(drift(np.array([a]), t)[0])
            Ar = float(drift(np.array([b]), t)[0])
        else:
            Af = np.asarray(drift(faces), dtype=float)
            Al = float(drift(np.array([a]))[0])
            Ar = float(drift(np.array([b]))[0])
        sub, diag, sup = _operator(Af, D, h, left, right, Al, Ar)
        cl, cr = _boundary_coefficients(D, h, left, right, Al, Ar)
        return sub, diag, sup, cl, cr

    fixed = None if time_dependent else op(0.0)

    def op_at(t: float) -> tuple[Array, Array, Array, float, float]:
        return fixed if fixed is not None else op(t)

    surv = np.empty(n_steps + 1)
    fl = np.empty(n_steps + 1)
    fr = np.empty(n_steps + 1)
    # probability absorbed at each end during each step, from the scheme's own
    # discrete balance, so that survival + absorbed = 1 to rounding error
    abs_l = np.zeros(n_steps + 1)
    abs_r = np.zeros(n_steps + 1)
    snaps_idx = (np.unique(np.linspace(0, n_steps, n_snapshots).round().astype(np.intp))
                 if n_snapshots > 0 else np.array([], dtype=np.intp))
    snaps: list[Array] = []
    _, _, _, cl, cr = op_at(0.0)
    surv[0], fl[0], fr[0] = h * p.sum(), cl * p[0], cr * p[-1]
    if 0 in snaps_idx:
        snaps.append(p.copy())
    c_bdf = 1.0 / (_GAMMA * (2.0 - _GAMMA))
    self_start = be_start
    w = (1.0 - _GAMMA) / (2.0 - _GAMMA)
    for k in range(1, n_steps + 1):
        t, t1 = t_grid[k - 1], t_grid[k]
        dt_k = t1 - t
        if k <= self_start:  # optional backward-Euler start, in two half steps
            for half in (0.5, 1.0):
                sub, diag, sup, cl_h, cr_h = op_at(t + half * dt_k)
                p = _step_solve(sub, diag, sup, dt_k / 2.0, p)
                abs_l[k] += dt_k / 2.0 * cl_h * p[0]
                abs_r[k] += dt_k / 2.0 * cr_h * p[-1]
        else:
            sub0, diag0, sup0, cl0, cr0 = op_at(t)
            subg, diagg, supg, clg, crg = op_at(t + _GAMMA * dt_k)
            rhs = p + (_GAMMA * dt_k / 2.0) * _apply(sub0, diag0, sup0, p)
            p_star = _step_solve(subg, diagg, supg, _GAMMA * dt_k / 2.0, rhs)
            sub1, diag1, sup1, cl1, cr1 = op_at(t1)
            rhs2 = (p_star - (1.0 - _GAMMA) ** 2 * p) * c_bdf
            p_new = _step_solve(sub1, diag1, sup1, w * dt_k, rhs2)
            tr = _GAMMA * dt_k / 2.0
            abs_l[k] = c_bdf * tr * (cl0 * p[0] + clg * p_star[0]) + w * dt_k * cl1 * p_new[0]
            abs_r[k] = c_bdf * tr * (cr0 * p[-1] + crg * p_star[-1]) + w * dt_k * cr1 * p_new[-1]
            p = p_new
        _, _, _, cl, cr = op_at(t1)
        surv[k] = h * p.sum()
        fl[k], fr[k] = cl * p[0], cr * p[-1]
        if k in snaps_idx:
            snaps.append(p.copy())
    absorbed_l = np.cumsum(abs_l)
    absorbed_r = np.cumsum(abs_r)
    return FPSolution(
        t=t_grid, survival=surv, flux_left=fl, flux_right=fr,
        absorbed_left=absorbed_l, absorbed_right=absorbed_r, x=x,
        density=np.array(snaps) if snaps else None,
        snapshots=snaps_idx if snaps else None,
        info={"n_cells": n_cells, "dt_max": dt_max, "dt_min": dt_min, "steps": n_steps, "h": h,
              "scheme": "Scharfetter-Gummel finite volumes, TR-BDF2"},
    )


def solve(
    drift: Callable[..., Array],
    D: float,
    domain: tuple[float, float],
    *,
    left: str,
    right: str,
    t_end: float,
    x0: float | None = None,
    p0: Callable[[Array], Array] | Array | None = None,
    n_cells: int = 2000,
    dt: float | None = None,
    dt_min: float | None = None,
    growth: float = 1.01,
    time_dependent: bool = False,
    n_snapshots: int = 0,
    richardson: bool = True,
    be_start: int = 0,
) -> FPSolution:
    """Evolve the density from a point ``x0`` (or a density ``p0``) to ``t_end``.

    Parameters
    ----------
    drift
        A(x) or, with ``time_dependent=True``, A(x, t).
    D
        Constant diffusion coefficient.
    domain
        Finite (a, b).
    left, right
        "absorbing" or "reflecting".
    dt
        Largest time step; by default ``t_end / 2000``. Steps start at ``dt_min``
        (default ``dt / 1000``) and grow by the factor ``growth`` per step.
    richardson
        Also solve with twice as many cells and extrapolate. The absorbing-wall
        closure is first order in the cell width; this removes that term. The
        difference between the two grids is returned in ``info["richardson_change"]``.
    """
    kw = dict(left=left, right=right, t_end=t_end, x0=x0, p0=p0, dt=dt, dt_min=dt_min,
              growth=growth, time_dependent=time_dependent, be_start=be_start)
    coarse = _solve_once(drift, D, domain, n_cells=n_cells, n_snapshots=0 if richardson else n_snapshots, **kw)
    if not richardson:
        return coarse
    fine = _solve_once(drift, D, domain, n_cells=2 * n_cells, n_snapshots=n_snapshots, **kw)

    def ex(f: Array, c: Array) -> Array:
        return 2.0 * f - c

    change = float(np.max(np.abs(fine.survival - coarse.survival)))
    info = dict(fine.info)
    info.update({"richardson": True, "richardson_change": change, "n_cells": (n_cells, 2 * n_cells)})
    return FPSolution(
        t=fine.t,
        survival=ex(fine.survival, coarse.survival),
        flux_left=ex(fine.flux_left, coarse.flux_left),
        flux_right=ex(fine.flux_right, coarse.flux_right),
        absorbed_left=ex(fine.absorbed_left, coarse.absorbed_left),
        absorbed_right=ex(fine.absorbed_right, coarse.absorbed_right),
        x=fine.x, density=fine.density, snapshots=fine.snapshots, info=info,
    )
