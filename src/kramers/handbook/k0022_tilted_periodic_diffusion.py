"""K0022. Effective diffusion in a tilted periodic potential (Reimann et al.)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .._checking import relative
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from ..solvers.quadrature import FirstPassage1D
from .k0021_tilted_periodic_velocity import _I_pm

Func = Callable[[np.ndarray], np.ndarray]


def tilted_periodic_diffusion(V: Func, F: float, L: float, D0: float, kT: float = 1.0) -> float:
    """Long-time effective diffusion coefficient: D0 <I+^2 I-> / <I+>^3 over one period (Reimann et al. Eq. 8)."""
    if not (L > 0 and D0 > 0 and kT > 0):
        raise OutsideDomain("need L, D0, kT > 0")
    if F < 0:
        return tilted_periodic_diffusion(lambda x: V(-x), -F, L, D0, kT)
    _, ip, im = _I_pm(V, F, L, D0, kT)
    return float(D0 * np.mean(ip * ip * im) / np.mean(ip) ** 3)


def _check_free():
    rows = [(f"V=0, F={F}", tilted_periodic_diffusion(lambda x: 0 * x, F, 1.0, 0.7, 1.3), 0.7) for F in (0.2, 3.0)]
    return relative(rows, 1e-11, ref="D0")


def _check_quadrature():
    # Eq. (5): D = (L^2/2) var(t) / <t>^3 with the passage-time moments over one period (quadrature solver)
    rows = []
    for amp, F in ((2.0, 1.0), (1.0, 0.5), (4.0, 2.0 * np.pi * 4.0 * 0.95)):
        V = lambda x, amp=amp: amp * np.cos(2 * np.pi * x)  # noqa: E731
        U = lambda x, amp=amp, F=F: amp * np.cos(2 * np.pi * x) - F * x  # noqa: E731
        fp = FirstPassage1D(U, D=1.0, domain=(-np.inf, 1.0), left="natural", right="absorbing")
        m = fp.moments(0.0, n=2).value
        rows.append((f"V = {amp} cos 2 pi x, F={F:.4g}", tilted_periodic_diffusion(V, F, 1.0, 1.0),
                     0.5 * (m[1] - m[0] ** 2) / m[0] ** 3))
    return relative(rows, 1e-8, ref="(L^2/2) var t / <t>^3 (quadrature solver)")


register(Entry(
    id="K0022",
    slug="tilted-periodic-diffusion",
    title="Effective diffusion in a tilted periodic potential",
    area="transport",
    kind="exact",
    statement=(
        "The long-time effective diffusion coefficient of an overdamped particle in a tilted "
        "periodic potential, as one integral over a period. Near the critical tilt it can "
        "exceed free diffusion by orders of magnitude (giant acceleration of diffusion)."
    ),
    model=r"\eta\dot x=-V'(x)+F+\sqrt{2\eta k_BT}\,\xi(t),\quad V(x+L)=V(x)",
    formula=r"D=D_0\frac{\int_{x_0}^{x_0+L}\frac{dx}{L}I_+^2(x)I_-(x)}{\left[\int_{x_0}^{x_0+L}\frac{dx}{L}I_+(x)\right]^3}",
    function=tilted_periodic_diffusion,
    parameters={"V": "periodic potential", "F": "tilt", "L": "period", "D0": "kT/eta", "kT": "thermal energy"},
    sources=(
        Source(
            citation="P. Reimann, C. Van den Broeck, H. Linke, P. Haenggi, J. M. Rubi and A. Perez-Madrid, Phys. Rev. Lett. 87, 010602 (2001)",
            doi="10.1103/PhysRevLett.87.010602",
            locator="Eqs. (8), (9), (5), p. 010602-1",
            version_read="journal PDF in the Universitat de Barcelona repository, page image read 14 Oct 2026",
            printed=r"D=D_0\frac{\int_{x_0}^{x_0+L}\frac{dx}{L}I_+^2(x)I_-(x)}{[\int_{x_0}^{x_0+L}\frac{dx}{L}I_+(x)]^3}",
            notation="same symbols",
        ),
        Source(
            citation="P. Reimann, C. Van den Broeck, H. Linke, P. Haenggi, J. M. Rubi and A. Perez-Madrid, Phys. Rev. E 65, 031104 (2002)",
            doi="10.1103/PhysRevE.65.031104",
            locator="the long version",
            version_read="not opened (closed access)",
            role="restatement",
            evidence="citation-only",
        ),
    ),
    validity="One-dimensional overdamped motion, white noise, any periodic V and tilt. Exact.",
    checks=(
        Check("no potential: D0", "free diffusion", _check_free),
        Check("against the passage-time moments over one period", "quadrature solver, Eq. (5)", _check_quadrature),
    ),
    aliases=("giant diffusion", "giant acceleration of free diffusion", "diffusion in a tilted washboard"),
    related=("K0021",),
    added="2026-10-14",
))
