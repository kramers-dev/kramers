"""The time-dependent Fokker-Planck solver against closed forms and against quadrature."""

import numpy as np
import pytest

from kramers.solvers import fokker_planck
from kramers.solvers.quadrature import FirstPassage1D


def test_free_diffusion_balance_mean_and_splitting():
    D, L, x0 = 0.7, 2.0, 0.3
    sol = fokker_planck.solve(lambda x: 0 * x, D, (0.0, L), left="absorbing", right="absorbing", x0=x0,
                              t_end=15.0, n_cells=500, dt=5e-3)
    assert 1 - sol.survival[-1] - sol.absorbed_left[-1] - sol.absorbed_right[-1] == pytest.approx(0, abs=1e-10)
    assert sol.absorbed_right[-1] == pytest.approx(x0 / L, abs=1e-10)  # Redner (2023) Eq. (4.12), v = 0
    assert sol.mean_time() == pytest.approx(x0 * (L - x0) / (2 * D), rel=1e-5)  # Eq. (4.16)


def test_inverse_gaussian_density():
    # Krapivsky and Redner (2018) Eq. (1): drift v towards the target from distance x0
    D, v, x0 = 1.0, 0.5, 2.0
    sol = fokker_planck.solve(lambda x: -v + 0 * x, D, (0.0, 40.0), left="absorbing", right="reflecting", x0=x0,
                              t_end=200.0, n_cells=2000, dt=0.05, dt_min=1e-5)
    t = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    exact = x0 / np.sqrt(4 * np.pi * D * t**3) * np.exp(-((x0 - v * t) ** 2) / (4 * D * t))
    assert np.interp(t, sol.t, sol.flux_left) == pytest.approx(exact, rel=5e-4)


def test_against_quadrature_in_a_double_well():
    U = lambda x: 3.0 * (x**2 - 1) ** 2  # noqa: E731
    exact = float(FirstPassage1D(U, D=1.0, domain=(-3.0, 0.8), left="reflecting", right="absorbing").mean_time(-1.0).value)
    sol = fokker_planck.solve(lambda x: -12.0 * x * (x**2 - 1), 1.0, (-3.0, 0.8), left="reflecting", right="absorbing",
                              x0=-1.0, t_end=1500.0, n_cells=1000, dt=0.125)
    assert sol.mean_time() == pytest.approx(exact, rel=1e-5)


def test_time_dependent_interface_agrees_with_static():
    a = fokker_planck.solve(lambda x: -1.0 + 0 * x, 1.0, (0.0, 5.0), left="absorbing", right="reflecting", x0=1.0,
                            t_end=10.0, n_cells=400, richardson=False)
    b = fokker_planck.solve(lambda x, t: -1.0 + 0 * x, 1.0, (0.0, 5.0), left="absorbing", right="reflecting", x0=1.0,
                            t_end=10.0, n_cells=400, richardson=False, time_dependent=True)
    assert b.survival == pytest.approx(a.survival, abs=1e-14)
