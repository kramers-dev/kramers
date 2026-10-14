"""K0002. Exact mean first-passage time of a one-dimensional diffusion in a potential."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from .._checking import combine, relative, statistical
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from ..solvers import fokker_planck, simulate
from ..solvers.quadrature import FirstPassage1D


def mean_first_passage_time(potential: Callable[[np.ndarray], np.ndarray], x0: ArrayLike, a: float, b: float,
                            D: float = 1.0, left: str = "reflecting", right: str = "absorbing") -> np.ndarray | float:
    """Mean time to leave (a, b) from x0 for dx = -D U'(x) dt + sqrt(2D) dW, U in kT.

    ``left`` and ``right`` are "reflecting" or "absorbing" (at least one absorbing);
    infinite ends are allowed when the potential confines ("natural") or the drift
    expels the particle in finite time ("exit"). Reflecting at a and absorbing at b is
    Haenggi, Talkner and Borkovec's Eq. (7.8); both absorbing is their Eq. (7.7) with
    a misprinted sign corrected (see the entry notes).
    """
    if not D > 0:
        raise OutsideDomain("D must be positive")
    fp = FirstPassage1D(potential, D=float(D), domain=(a, b), left=left, right=right)
    value = fp.mean_time(x0).value
    return float(value) if np.ndim(value) == 0 else np.asarray(value)


def _check_closed_forms():
    rows = []
    D, L = 0.7, 2.0
    for x in (0.0, 0.6, 1.9):
        rows.append((f"flat, reflecting/absorbing, x={x}", mean_first_passage_time(lambda y: 0 * y, x, 0.0, L, D),
                     (L * L - x * x) / (2 * D)))
    for x in (0.3, 1.0):
        rows.append((f"flat, both absorbing, x={x}", mean_first_passage_time(lambda y: 0 * y, x, 0.0, L, D, left="absorbing"),
                     x * (L - x) / (2 * D)))
    # constant drift v towards b, both absorbing: T = (L pi_L(x) - x) / v (Redner; Srivastava et al. Eq. 7)
    v = 1.0
    for x in (0.5, 1.0):
        pi = np.expm1(-v * x / D) / np.expm1(-v * L / D)
        rows.append((f"drift v={v}, both absorbing, x={x}",
                     mean_first_passage_time(lambda y: -v * y / D, x, 0.0, L, D, left="absorbing"), (L * pi - x) / v))
    return relative(rows, 1e-12, ref="closed forms")


def _check_fokker_planck():
    # a non-flat potential: the misprinted and the correct two-absorbing-end formulas differ here
    U = lambda x: x**3 / 3.0 - x  # noqa: E731
    rows = []
    for left in ("reflecting", "absorbing"):
        sol = fokker_planck.solve(lambda x: -(x * x - 1.0), 1.0, (-2.0, 1.0), left=left, right="absorbing",
                                  x0=-1.0, t_end=60.0, n_cells=1500, dt=0.01)
        rows.append((f"U = x^3/3 - x on [-2, 1], {left} at -2, x0=-1",
                     mean_first_passage_time(U, -1.0, -2.0, 1.0, left=left), sol.mean_time()))
    return relative(rows, 2e-5, ref="integral of the survival, time-dependent Fokker-Planck")


def _check_simulation():
    U = lambda x: 2.0 * (x * x - 1.0) ** 2  # noqa: E731
    sim = simulate.first_passage_times(lambda x: -8.0 * x * (x * x - 1.0), 1.0, -1.0, n_paths=20_000, dt=2e-4,
                                       t_max=2000.0, absorbing=(None, 0.8), reflecting=(-2.5, None), seed=31)
    m = sim.mean()
    return combine(statistical([("double well, -1 to 0.8", mean_first_passage_time(U, -1.0, -2.5, 0.8), m.value, m.error)]))


register(Entry(
    id="K0002",
    slug="mfpt-1d",
    title="Exact mean first-passage time in one dimension (quadrature formula)",
    area="escape",
    kind="exact",
    statement=(
        "The mean time for an overdamped particle in a potential U to leave an interval is "
        "a double integral of exp(U/kT) and exp(-U/kT): the Pontryagin-Andronov-Vitt "
        "formula. It holds at any barrier height and for any potential shape, and it is "
        "the exact reference against which rate formulas such as Kramers' are judged."
    ),
    model=r"dx = -D\,U'(x)\,dt + \sqrt{2D}\,dW,\ a<x<b,\ U \text{ in } k_BT",
    formula=r"t(x) = \frac{1}{D}\int_x^b dy\, e^{U(y)}\int_a^y dz\, e^{-U(z)}\quad (a \text{ reflecting},\ b \text{ absorbing})",
    function=mean_first_passage_time,
    parameters={"potential": "U(x) in kT", "x0": "start", "a, b": "ends of the interval",
                "D": "diffusion coefficient", "left, right": "boundary types"},
    sources=(
        Source(
            citation="P. Haenggi, P. Talkner and M. Borkovec, Rev. Mod. Phys. 62, 251 (1990)",
            doi="10.1103/RevModPhys.62.251",
            locator="Eq. (7.8), p. 291 (reflecting a, absorbing b); Eq. (7.7) (both absorbing); Eq. (4.56b), p. 275",
            version_read="scan of the journal pages, Augsburg university repository; p. 291 viewed 14 Oct 2026",
            printed=r"t(x)=\frac{M\gamma}{k_BT}\int_x^b dy\,\exp[U(y)/k_BT]\int_a^y dz\,\exp[-U(z)/k_BT]",
            notation="M gamma / kT = 1/D; U/kT = our U",
        ),
        Source(
            citation="A. Szabo, K. Schulten and Z. Schulten, J. Chem. Phys. 72, 4350 (1980)",
            doi="10.1063/1.439715",
            locator="Eq. (2.15), p. 4352 (any dimension, position-dependent D, radiation boundary)",
            version_read="journal scan on the Schulten group site, opened 14 Oct 2026",
            role="restatement",
        ),
        Source(
            citation="L. Pontryagin, A. Andronov and A. Vitt (1933), as cited by Haenggi, Talkner and Borkovec",
            doi=None,
            locator="credited by Haenggi, Talkner and Borkovec for Eq. (7.8)",
            version_read="not opened; journal, volume and page not verified here",
            role="original",
            evidence="citation-only",
        ),
    ),
    validity="One-dimensional overdamped (Fokker-Planck) dynamics with white noise; exact.",
    checks=(
        Check("closed forms: flat potential and constant drift", "Redner (2023) Eqs. (4.16); Srivastava et al. (2017) Eq. (7)", _check_closed_forms),
        Check("a non-flat potential against the time-dependent solver", "Fokker-Planck solver, integral of the survival", _check_fokker_planck),
        Check("double well against simulation", "Euler-Maruyama, 20,000 paths", _check_simulation, slow=True),
    ),
    aliases=("Pontryagin formula", "Kramers time", "mean exit time", "MFPT"),
    related=("K0001", "K0003", "K0012"),
    notes=(
        "Erratum found while implementing this entry. In Eq. (7.7) of Haenggi, Talkner and "
        "Borkovec as printed (journal scan, p. 291), the inner integral in the numerator of the "
        "first term has exp[U(z)/kBT]; it must be exp[-U(z)/kBT]. Computed check: for "
        "U = x^3/3 - x on [-2, 1] with kT = D = 1, the printed form gives t(b) = 2.511 instead "
        "of 0, while the corrected form agrees with this entry to 1e-10. A flat potential "
        "cannot reveal the difference. No erratum was found in Crossref."
    ),
    added="2026-10-14",
))
