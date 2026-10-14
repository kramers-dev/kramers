"""K0021. Mean velocity of an overdamped particle in a tilted periodic potential (Stratonovich's formula)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.special import roots_legendre

from .._checking import relative
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from ..solvers.quadrature import FirstPassage1D

Func = Callable[[np.ndarray], np.ndarray]


def _I_pm(V: Func, F: float, L: float, D0: float, kT: float, n_x: int = 256, panels: int = 24, order: int = 16):
    """I+(x) and I-(x) of Reimann et al., Eq. (9), on a uniform periodic grid of x."""
    x = np.arange(n_x) * L / n_x
    gx, gw = roots_legendre(order)
    edges = np.linspace(0.0, L, panels + 1)
    y = ((edges[:-1, None] + edges[1:, None]) / 2 + (edges[1:, None] - edges[:-1, None]) / 2 * gx[None, :]).ravel()
    w = (np.diff(edges)[:, None] / 2 * gw[None, :]).ravel()
    Vx = V(x)[:, None]
    ip = np.exp((Vx - V(x[:, None] - y[None, :]) - y[None, :] * F) / kT) @ w / D0
    im = np.exp((-Vx + V(x[:, None] + y[None, :]) - y[None, :] * F) / kT) @ w / D0
    return x, ip, im


def tilted_periodic_velocity(V: Func, F: float, L: float, D0: float, kT: float = 1.0) -> float:
    """Long-time mean velocity for eta dx/dt = -V'(x) + F + noise, V of period L, D0 = kT/eta.

    <v> = (1 - exp(-L F / kT)) / [(1/L) integral over one period of I+(x)] (Reimann et al. Eq. 7).
    """
    if not (L > 0 and D0 > 0 and kT > 0):
        raise OutsideDomain("need L, D0, kT > 0")
    if F == 0:
        return 0.0
    if F < 0:  # mirror symmetry: x -> -x, V(x) -> V(-x)
        return -tilted_periodic_velocity(lambda x: V(-x), -F, L, D0, kT)
    _, ip, _ = _I_pm(V, F, L, D0, kT)
    return float(-np.expm1(-L * F / kT) / np.mean(ip))


def _check_free():
    rows = [(f"V=0, F={F}", tilted_periodic_velocity(lambda x: 0 * x, F, 1.0, 0.7, 1.3), 0.7 * F / 1.3) for F in (0.2, 3.0, -2.0)]
    return relative(rows, 1e-12, ref="F / eta = D0 F / kT")


def _check_quadrature():
    # Eq. (4): <v> = L / <t(x0 -> x0 + L)>, the mean time computed by the quadrature solver
    rows = []
    for amp, F in ((2.0, 1.0), (4.0, 3.0), (1.0, 0.5)):
        V = lambda x, amp=amp: amp * np.cos(2 * np.pi * x)  # noqa: E731
        U = lambda x, amp=amp, F=F: amp * np.cos(2 * np.pi * x) - F * x  # noqa: E731
        fp = FirstPassage1D(U, D=1.0, domain=(-np.inf, 1.0), left="natural", right="absorbing")
        rows.append((f"V = {amp} cos 2 pi x, F={F}", tilted_periodic_velocity(V, F, 1.0, 1.0), 1.0 / float(fp.mean_time(0.0).value)))
    return relative(rows, 1e-9, ref="L / mean time over one period (quadrature solver)")


register(Entry(
    id="K0021",
    slug="tilted-periodic-velocity",
    title="Mean velocity in a tilted periodic potential (Stratonovich's formula)",
    area="transport",
    kind="exact",
    statement=(
        "An overdamped Brownian particle in a periodic potential tilted by a constant force F "
        "drifts with a long-time mean velocity given by one double integral over a period. It "
        "covers the whole range from the locked state to the running state, through the "
        "critical tilt where the barriers vanish."
    ),
    model=r"\eta\dot x=-V'(x)+F+\sqrt{2\eta k_BT}\,\xi(t),\quad V(x+L)=V(x)",
    formula=(r"\langle\dot x\rangle=\frac{1-e^{-LF/k_BT}}{\int_{x_0}^{x_0+L}\frac{dx}{L}I_+(x)},\quad "
             r"I_\pm(x)=\int_0^L\frac{dy}{D_0}e^{[\pm V(x)\mp V(x\mp y)-yF]/k_BT}"),
    function=tilted_periodic_velocity,
    parameters={"V": "periodic potential (vectorised)", "F": "tilting force", "L": "period",
                "D0": "free diffusion coefficient kT/eta", "kT": "thermal energy"},
    sources=(
        Source(
            citation="P. Reimann, C. Van den Broeck, H. Linke, P. Haenggi, J. M. Rubi and A. Perez-Madrid, Phys. Rev. Lett. 87, 010602 (2001)",
            doi="10.1103/PhysRevLett.87.010602",
            locator="Eqs. (7) and (9), p. 010602-1; Eq. (4)",
            version_read="journal PDF in the Universitat de Barcelona repository (hdl 2445/13254), page image read 14 Oct 2026",
            printed=(r"\langle\dot x\rangle=\frac{1-e^{-LF/kT}}{\int_{x_0}^{x_0+L}\frac{dx}{L}I_+(x)};\ "
                     r"I_\pm(x):=\int_0^L\frac{dy}{D_0}e^{[\pm V(x)\mp V(x\mp y)-yF]/kT}"),
            notation="same symbols; V is the untilted periodic part",
        ),
        Source(
            citation="R. L. Stratonovich, Radiotekh. Elektron. 3, 497 (1958)",
            doi=None,
            locator="the velocity formula, as credited by Reimann et al. (their ref. 12)",
            version_read="not opened; citation as given by Reimann et al.",
            role="original",
            evidence="citation-only",
        ),
    ),
    validity="One-dimensional overdamped motion, white noise, any periodic V and any F. Exact.",
    checks=(
        Check("no potential: F / eta", "free drift", _check_free),
        Check("against L over the mean time per period", "quadrature solver, natural boundary behind the tilt", _check_quadrature),
    ),
    aliases=("tilted washboard", "Brownian motion in a tilted periodic potential", "overdamped Josephson junction I-V curve",
             "colloid driven over a periodic landscape", "depinning"),
    related=("K0022",),
    notes=("Dagdug and Berezhkovskii (2009) restate I+- with the opposite overall sign in the "
           "exponent; the sign here is the PRL's, confirmed by the free-drift check."),
    added="2026-10-14",
))
