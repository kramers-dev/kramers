"""K0001. Kramers' escape rate in the overdamped (Smoluchowski) limit."""

from __future__ import annotations

import warnings

import numpy as np

from .._checking import combine, relative
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain, ValidityWarning
from ..solvers.quadrature import FirstPassage1D

# Relative overestimate of the formula against the exact decay rate (smallest eigenvalue
# from the quadrature solver), computed 14 October 2026 by the check below.
VALIDITY_MAP = {
    "barrier_kT": (2.0, 5.0, 10.0, 20.0, 40.0, 80.0),
    "cubic well": (0.0792, 0.0700, 0.0319, 0.0148, 0.0072, 0.0035),
    "quartic double well": (0.166, 0.0996, 0.0433, 0.0200, 0.0097, 0.0048),
    "cosine well": (0.226, 0.134, 0.0582, 0.0268, 0.0129, 0.0064),
}


def kramers_rate_overdamped(curvature_well: float, curvature_barrier: float, barrier: float,
                            friction: float, kT: float = 1.0) -> float:
    """k = sqrt(U''(a) |U''(b)|) / (2 pi friction) * exp(-barrier / kT).

    Parameters
    ----------
    curvature_well
        U''(a) > 0 at the bottom of the well (energy / length^2).
    curvature_barrier
        |U''(b)| at the top of the barrier.
    barrier
        U(b) - U(a) > 0.
    friction
        Friction coefficient zeta in zeta dx/dt = -U'(x) + noise (so D = kT / zeta).
        It is M gamma in the notation of Haenggi, Talkner and Borkovec, whose gamma is
        a damping rate.
    kT
        Thermal energy.

    Warns ``ValidityWarning`` when the barrier is below 10 kT, where the formula is
    several per cent high (see ``VALIDITY_MAP``).
    """
    for name, v in (("curvature_well", curvature_well), ("curvature_barrier", curvature_barrier),
                    ("barrier", barrier), ("friction", friction), ("kT", kT)):
        if not v > 0:
            raise OutsideDomain(f"{name} must be positive")
    ratio = barrier / kT
    if ratio < 10.0:
        warnings.warn(f"barrier of {ratio:.3g} kT: Kramers' rate is too high by corrections of order kT/barrier "
                      f"(3 to 6% at 10 kT, 7 to 13% at 5 kT for cubic, quartic and cosine wells; entry K0001)",
                      ValidityWarning, stacklevel=2)
    return float(np.sqrt(curvature_well * curvature_barrier) / (2.0 * np.pi * friction) * np.exp(-ratio))


def _exact_and_formula(shape: str, B: float) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValidityWarning)
        if shape == "cubic well":
            u = (0.75 * B) ** (2.0 / 3.0)
            fp = FirstPassage1D(lambda y: u * y - y**3 / 3.0, D=1.0, left="natural", right="exit",
                                dlog_psi=lambda y: y * y - u)
            k = kramers_rate_overdamped(2 * np.sqrt(u), 2 * np.sqrt(u), B, 1.0)
        elif shape == "quartic double well":
            fp = FirstPassage1D(lambda x: B * (x * x - 1.0) ** 2, D=1.0, domain=(-np.inf, 1.0),
                                left="natural", right="absorbing")
            k = kramers_rate_overdamped(8 * B, 4 * B, B, 1.0)
        else:  # cosine well, two barriers: the rate out of the well is 2k
            fp = FirstPassage1D(lambda x: 0.5 * B * (1.0 - np.cos(x)), D=1.0, domain=(-2 * np.pi, 2 * np.pi),
                                left="absorbing", right="absorbing")
            k = 2 * kramers_rate_overdamped(B / 2, B / 2, B, 1.0)
    return k, float(fp.decay_rate().value)


def _check_convergence():
    parts = []
    for shape in ("cubic well", "quartic double well", "cosine well"):
        rows = []
        for B in (40.0, 80.0, 160.0):
            k, lam = _exact_and_formula(shape, B)
            rows.append((f"{shape}, {B:g} kT", k, lam))
        parts.append(relative(rows, 0.015, ref="exact decay rate (quadrature)"))
        # the error must fall like 1/barrier: compare error x barrier at 80 and 160 kT
        e80 = rows[1][1] / rows[1][2] - 1.0
        e160 = rows[2][1] / rows[2][2] - 1.0
        parts.append(relative([(f"{shape}: error x barrier, 160 vs 80 kT", e160 * 160.0, e80 * 80.0)], 0.03,
                              what="160 kT", ref="80 kT"))
    return combine(*parts)


def _check_map():
    rows = []
    for shape in ("cubic well", "quartic double well", "cosine well"):
        for B, stored in zip(VALIDITY_MAP["barrier_kT"][1:4], VALIDITY_MAP[shape][1:4], strict=True):
            k, lam = _exact_and_formula(shape, B)
            rows.append((f"{shape}, {B:g} kT", k / lam - 1.0, stored))
    return relative(rows, 0.01, what="overestimate now", ref="stored map")


register(Entry(
    id="K0001",
    slug="kramers-rate-overdamped",
    title="Kramers' escape rate, overdamped (Smoluchowski) limit",
    area="escape",
    kind="asymptotic",
    statement=(
        "For strong friction and a barrier high compared with kT, the rate of escape from a "
        "potential well is the geometric mean of the curvatures at the well bottom and the "
        "barrier top, divided by 2 pi times the friction coefficient, times the Arrhenius "
        "factor. It is too high by corrections of relative order kT/barrier: 3 to 6% at 10 kT "
        "for the cubic, quartic double-well and cosine shapes mapped here."
    ),
    model=r"\zeta\dot x = -U'(x) + \sqrt{2\zeta k_BT}\,\xi(t),\quad \Delta U = U(x_b)-U(x_a) \gg k_BT",
    formula=r"k = \frac{\sqrt{U''(x_a)\,|U''(x_b)|}}{2\pi\zeta}\exp\!\left(-\frac{\Delta U}{k_BT}\right)",
    function=kramers_rate_overdamped,
    parameters={"curvature_well": "U''(x_a)", "curvature_barrier": "|U''(x_b)|", "barrier": "Delta U",
                "friction": "friction coefficient zeta (D = kT/zeta)", "kT": "thermal energy"},
    sources=(
        Source(
            citation="P. Haenggi, P. Talkner and M. Borkovec, Rev. Mod. Phys. 62, 251 (1990)",
            doi="10.1103/RevModPhys.62.251",
            locator="Eq. (4.34), p. 272; also (4.54), p. 274, and (7.10), p. 291",
            version_read="scan of the journal pages, Augsburg university repository (opus4 file 41911); p. 291 viewed 14 Oct 2026",
            printed=r"k^{\rm overdamped}_{A\to C}=\frac{\omega_0\,\omega_b}{2\pi\gamma}\exp(-\beta E_b)",
            notation=("M omega_0^2 = U''(x_a), M omega_b^2 = |U''(x_b)|, E_b = Delta U, beta = 1/kT; their gamma "
                      "is a damping rate (Eq. 4.1), so M gamma = zeta, our friction coefficient"),
        ),
        Source(
            citation="H. A. Kramers, Physica 7, 284 (1940)",
            doi="10.1016/S0031-8914(40)90098-2",
            locator="the original derivation",
            version_read="not opened (closed access); citation only",
            role="original",
            evidence="citation-only",
        ),
    ),
    validity=("Overdamped motion (friction large compared with the barrier frequency), barrier "
              "large compared with kT, smooth parabolic well and barrier. Overestimate against the "
              "exact rate (computed check): cubic well 7.0%, 3.2%, 1.5% at 5, 10, 20 kT; quartic "
              "double well 10.0%, 4.3%, 2.0%; cosine well 13.4%, 5.8%, 2.7%."),
    checks=(
        Check("converges to the exact decay rate like 1/barrier", "smallest eigenvalue, quadrature solver", _check_convergence),
        Check("the stored validity map is current", "the same computation", _check_map, slow=True),
    ),
    aliases=("Kramers rate", "Arrhenius law with prefactor", "transition state theory, overdamped",
             "Smoluchowski limit", "Eyring-Kramers formula in one dimension"),
    related=("K0002", "K0006", "K0008"),
    extra={"validity_map": VALIDITY_MAP},
    added="2026-10-14",
))
