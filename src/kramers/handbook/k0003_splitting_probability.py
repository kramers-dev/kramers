"""K0003. Splitting probability of a one-dimensional diffusion between two absorbing ends."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad

from .._checking import relative, statistical
from ..registry import Check, Entry, Source, register
from ..solvers import simulate
from ..solvers.quadrature import FirstPassage1D


def splitting_probability(
    potential: Callable[[np.ndarray], np.ndarray],
    x0: ArrayLike,
    a: float,
    b: float,
    D: float | Callable[[np.ndarray], np.ndarray] = 1.0,
) -> np.ndarray | float:
    """Probability of reaching ``b`` before ``a`` from ``x0``.

    Model: dX = [-D U'(X) + D'(X)] dt + sqrt(2 D(X)) dW (Ito), U in units of kT, so that
    exp(-U) is the equilibrium density whatever D(x) is.

    Parameters
    ----------
    potential
        U(x) in units of kT (vectorised).
    x0
        Starting point(s), a <= x0 <= b.
    a, b
        The two absorbing ends.
    D
        Diffusion coefficient, a number or a function of x.
    """
    if callable(D):
        Dfun = D
        fp = FirstPassage1D(log_psi=lambda x: -potential(x) + np.log(Dfun(x)), D=Dfun,
                            domain=(a, b), left="absorbing", right="absorbing")
    else:
        fp = FirstPassage1D(potential, D=float(D), domain=(a, b), left="absorbing", right="absorbing")
    value = fp.splitting(x0).value
    return float(value) if np.ndim(value) == 0 else np.asarray(value)


def _check_constant_force():
    rows = []
    for F in (-30.0, -2.0, -0.1, 1e-7, 0.5, 4.0, 40.0):
        for x0 in (0.1, 0.5, 0.97):
            closed = np.expm1(-F * x0) / np.expm1(-F * 1.0)
            rows.append((f"F={F:g}, x0={x0}", splitting_probability(lambda x, F=F: -F * x, x0, 0.0, 1.0), closed))
    return relative(rows, 1e-11, ref="Redner (4.12)")


def _check_variable_D():
    Dfun = lambda x: 1.0 + 0.6 * np.sin(3.0 * x)  # noqa: E731
    U = lambda x: 0.8 * x**2  # noqa: E731
    rows = []
    for x0 in (-0.7, 0.0, 0.9):
        num = quad(lambda z: np.exp(U(z)) / Dfun(z), -1.0, x0, epsabs=0, epsrel=1e-13)[0]
        den = quad(lambda z: np.exp(U(z)) / Dfun(z), -1.0, 1.2, epsabs=0, epsrel=1e-13)[0]
        rows.append((f"x0={x0}", splitting_probability(U, x0, -1.0, 1.2, D=Dfun), num / den))
    return relative(rows, 1e-10, ref="adaptive quadrature of Eq. (6)")


def _check_simulation():
    U = lambda x: 3.0 * (x**2 - 1.0) ** 2  # noqa: E731
    rows = []
    for x0, seed in ((-0.3, 11), (0.2, 12)):
        sim = simulate.first_passage_times(lambda x: -12.0 * x * (x**2 - 1.0), 1.0, x0, n_paths=40_000,
                                           dt=2e-4, t_max=200.0, absorbing=(-1.4, 1.4), seed=seed)
        p = sim.probability("right")
        rows.append((f"double well, x0={x0}", splitting_probability(U, x0, -1.4, 1.4), p.value, p.error))
    return statistical(rows, n_sigma=4.0)


register(Entry(
    id="K0003",
    slug="splitting-probability-1d",
    title="Splitting probability between two absorbing ends in one dimension",
    area="targets",
    kind="exact",
    statement=(
        "For a diffusion in a potential U(x) with a position-dependent diffusion coefficient "
        "D(x), started at x0 between two absorbing points a < x0 < b, the probability of "
        "reaching b first is the integral of exp(U)/D from a to x0 divided by the same "
        "integral from a to b."
    ),
    model=r"dX = [-D(X)\,U'(X) + D'(X)]\,dt + \sqrt{2D(X)}\,dW,\quad a < X < b,\ U \text{ in } k_BT",
    formula=r"\pi_b(x_0) = \frac{\int_a^{x_0} e^{U(z)}\,dz/D(z)}{\int_a^{b} e^{U(z)}\,dz/D(z)}",
    function=splitting_probability,
    parameters={"potential": "U(x) in units of kT", "x0": "start", "a, b": "absorbing ends",
                "D": "diffusion coefficient (number or function)"},
    sources=(
        Source(
            citation="L. Dagdug and A. M. Berezhkovskii, J. Chem. Phys. 131, 056101 (2009)",
            doi="10.1063/1.3179679",
            locator="Eq. (6)",
            version_read="PubMed Central PMC2736570 (MathML), opened 14 Oct 2026",
            printed=r"W_R(x_0)=\frac{\int_{x_L}^{x_0}e^{\beta U(z)}\frac{dz}{D(z)}}{\int_{x_L}^{x_R}e^{\beta U(z)}\frac{dz}{D(z)}}",
            notation="x_L = a, x_R = b, beta U = our U (in kT), W_R = pi_b",
        ),
        Source(
            citation="S. Redner, Physica A 631, 128545 (2023)",
            doi="10.1016/j.physa.2023.128545",
            locator="Eq. (4.12), p. 6 (constant drift)",
            version_read="journal PDF on the author's site, opened 14 Oct 2026",
            printed=r"\mathcal{E}_{+}(x_0)=\frac{1-e^{-vx_0/D}}{1-e^{-vL/D}}",
            notation="interval [0, L]; constant drift v = D F, i.e. U = -F x",
            role="restatement",
        ),
    ),
    validity="Markovian one-dimensional diffusion; both ends absorbing; D(x) > 0. Exact.",
    checks=(
        Check("constant force against the printed closed form", "Redner (2023), Eq. (4.12)", _check_constant_force),
        Check("position-dependent D against adaptive quadrature", "scipy.integrate.quad of the printed integrals", _check_variable_D),
        Check("double well against simulation", "Euler-Maruyama simulation, 40,000 paths", _check_simulation, slow=True),
    ),
    aliases=("hitting probability", "committor", "exit probability", "transmission probability"),
    related=("K0011",),
    added="2026-10-14",
))
