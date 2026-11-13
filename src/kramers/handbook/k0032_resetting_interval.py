"""K0032. Mean exit time from an interval with two absorbing ends for diffusion with resetting (Pal and Prasad)."""

from __future__ import annotations

import numpy as np

from .._checking import relative, statistical
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from ..solvers import simulate
from .k0012_drift_diffusion_interval import drift_diffusion_interval
from .k0017_restart_renewal import restart_moments


def _log_sinh(z: float) -> float:
    """log(sinh z) for z > 0, without overflow."""
    return float(z + np.log(-np.expm1(-2.0 * z)) - np.log(2.0))


def resetting_interval_mean_time(x0: float, a: float, b: float, r: float, D: float) -> float:
    """Mean time to leave (a, b) through either end, starting at and resetting to x0 at rate r.

    T = [ sinh((b-a) alpha0) / (sinh((b-x0) alpha0) + sinh((x0-a) alpha0)) - 1 ] / r, alpha0 = sqrt(r/D).
    Evaluated through sinh(u+v) - sinh u - sinh v = 4 sinh(u/2) sinh(v/2) sinh((u+v)/2), so that there is no
    cancellation as r -> 0 and no overflow for large r.
    """
    if not (a < x0 < b):
        raise OutsideDomain("need a < x0 < b")
    if not (r > 0 and D > 0):
        raise OutsideDomain("need r > 0 and D > 0 (for r = 0 use K0012)")
    alpha0 = np.sqrt(r / D)
    u, v = (b - x0) * alpha0, (x0 - a) * alpha0
    log_num = np.log(4.0) + _log_sinh((u + v) / 2) + _log_sinh(u / 2) + _log_sinh(v / 2)
    log_den = np.logaddexp(_log_sinh(u), _log_sinh(v))
    return float(np.exp(log_num - log_den) / r)


def _interval_transform(x0: float, a: float, b: float, D: float):
    # Laplace transform of the exit time from (a, b) without resetting
    def T(s: float) -> float:
        al = np.sqrt(s / D)
        return float((np.sinh((b - x0) * al) + np.sinh((x0 - a) * al)) / np.sinh((b - a) * al))
    return T


def _check_renewal():
    rows = []
    for x0, a, b, r, D in ((0.3, -0.5, 1.2, 0.5, 0.5), (0.3, -0.5, 1.2, 3.0, 0.5), (0.9, 0.0, 1.0, 20.0, 1.0),
                           (2.0, 0.0, 10.0, 0.01, 0.3)):
        rows.append((f"x0={x0}, ({a}, {b}), r={r}, D={D}", resetting_interval_mean_time(x0, a, b, r, D),
                     restart_moments(_interval_transform(x0, a, b, D), r).mean))
    return relative(rows, 1e-12, ref="Reuveni (2016) Eq. (7) with the interval transform")


def _check_no_resetting_limit():
    rows = []
    for x0, a, b, D in ((0.3, -0.5, 1.2, 0.7), (0.1, 0.0, 1.0, 2.0)):
        T0 = drift_diffusion_interval(x0 - a, b - a, 0.0, D).mean_time
        rows.append((f"x0={x0}, ({a}, {b}), D={D}, r=1e-9", resetting_interval_mean_time(x0, a, b, 1e-9, D), T0))
    return relative(rows, 1e-6, ref="K0012 without resetting")


def _check_simulation():
    rows = []
    for r, seed in ((0.5, 321), (3.0, 322)):
        sim = simulate.first_passage_times(lambda x: 0 * x, 0.5, 0.3, n_paths=100_000, dt=2e-4, t_max=200.0,
                                           absorbing=(-0.5, 1.2), reset_rate=r, seed=seed)
        m = sim.mean()
        rows.append((f"x0=0.3, (-0.5, 1.2), r={r}, D=0.5", resetting_interval_mean_time(0.3, -0.5, 1.2, r, 0.5), m.value, m.error))
    return statistical(rows)


register(Entry(
    id="K0032",
    slug="resetting-interval",
    title="Diffusion with resetting in an interval: mean exit time through either end",
    area="resetting",
    kind="exact",
    statement=(
        "A Brownian particle in an interval with two absorbing ends, reset to its start x0 at rate r, leaves "
        "after a mean time [sinh((b-a)alpha0)/(sinh((b-x0)alpha0)+sinh((x0-a)alpha0)) - 1]/r. Unlike the "
        "half-line, the mean is finite without resetting, so resetting can raise it or lower it."
    ),
    model=(r"\partial_t Q_r = D\,\partial_x^2 Q_r - r\,Q_r + r\,Q_r(x_0,t),\ Q_r(a,t)=Q_r(b,t)=0"),
    formula=(r"\langle T_r(x_0)\rangle=\frac{1}{r}\left[\frac{\sinh((b-a)\alpha_0)}{\sinh((b-x_0)\alpha_0)"
             r"+\sinh((x_0-a)\alpha_0)}-1\right],\quad \alpha_0=\sqrt{r/D}"),
    function=resetting_interval_mean_time,
    parameters={"x0": "start and reset position", "a": "left absorbing end", "b": "right absorbing end",
                "r": "resetting rate", "D": "diffusion coefficient (<x^2> = 2Dt between resets)"},
    sources=(
        Source(
            citation="A. Pal and V. V. Prasad, Phys. Rev. E 99, 032123 (2019)",
            doi="10.1103/PhysRevE.99.032123",
            locator="Eq. (15), with alpha0 from Eq. (16); model Eq. (1) (arXiv:1812.03009 numbering)",
            version_read="arXiv:1812.03009 HTML (ar5iv), opened 14 Oct and 13 Nov 2026; journal numbering not checked",
            printed=(r"\langle T_{r}(x_{0})\rangle=\frac{1}{r}\left[\frac{\sinh(b-a)\alpha_{0}}{\sinh(b-x_{0})\alpha_{0}"
                     r"+\sinh(x_{0}-a)\alpha_{0}}-1\right]"),
            notation="sinh(b-a)alpha0 is read as sinh((b-a) alpha0); the paper's figures use D = 1/2",
        ),
    ),
    validity="One dimension, free diffusion, Poissonian instantaneous resetting to the start, both ends absorbing. Exact.",
    checks=(
        Check("against the renewal formula", "Reuveni (2016) Eq. (7) with the interval's Laplace transform (entry K0017)",
              _check_renewal),
        Check("the limit without resetting", "entry K0012 with zero drift", _check_no_resetting_limit),
        Check("against simulation", "Euler-Maruyama with Poissonian resetting and crossing correction, 100,000 paths",
              _check_simulation, slow=True),
    ),
    aliases=("resetting in a box", "restart with two exits", "first passage under resetting in an interval"),
    related=("K0012", "K0015", "K0016", "K0017"),
    added="2026-11-13",
))
