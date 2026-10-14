"""Simulation and Laplace inversion against closed forms."""

import numpy as np
import pytest

from kramers.solvers import laplace, simulate


def test_simulation_free_diffusion():
    sim = simulate.first_passage_times(lambda x: 0 * x, 1.0, 0.3, n_paths=20_000, dt=2.5e-4, t_max=50.0,
                                       absorbing=(0.0, 1.0), seed=1)
    m, p = sim.mean(), sim.probability("right")
    assert abs(m.value - 0.105) < 4 * m.error + 1e-3  # x(L - x)/(2D), plus first-order dt bias
    assert abs(p.value - 0.3) < 4 * p.error


def test_simulation_with_resetting():
    sim = simulate.first_passage_times(lambda x: 0 * x, 1.0, 1.0, n_paths=20_000, dt=5e-4, t_max=400.0,
                                       absorbing=(0.0, None), reset_rate=1.0, seed=3)
    m = sim.mean()
    assert abs(m.value - (np.e - 1.0)) < 4 * m.error + 5e-3  # Evans and Majumdar Eq. (7)


@pytest.mark.parametrize("F, f", [
    (lambda s: np.exp(-1.0 * np.sqrt(s / 0.5)), lambda t: 1.0 / np.sqrt(4 * np.pi * 0.5 * t**3) * np.exp(-1.0 / (4 * 0.5 * t))),
    (lambda s: np.exp(1.0 * (0.3 - np.sqrt(0.09 + 2.0 * s))), lambda t: 1.0 / np.sqrt(2 * np.pi * t**3) * np.exp(-((1.0 - 0.3 * t) ** 2) / (2 * t))),
])
def test_laplace_inversion(F, f):
    t = np.array([0.1, 0.5, 1.0, 3.0, 10.0])
    assert laplace.invert(F, t) == pytest.approx(f(t), rel=1e-9)
