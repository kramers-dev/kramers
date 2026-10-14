"""The quadrature solver against closed forms and independent high-precision values.

Every number here is either an exact formula (stated next to it) or a value from an
independent method, named with its source. Tolerances are those the method reaches,
with a margin of about ten.
"""

import numpy as np
import pytest
from scipy.special import airy, gamma

from kramers import OutsideDomain
from kramers.solvers.quadrature import FirstPassage1D

FLAT = lambda x: 0 * x  # noqa: E731


@pytest.mark.parametrize("left, right, x, exact", [
    ("absorbing", "absorbing", [0.3, 1.0, 1.7], lambda x: x * (2 - x) / (2 * 0.7)),
    ("reflecting", "absorbing", [0.0, 0.3, 1.7], lambda x: (4 - x**2) / (2 * 0.7)),
    ("absorbing", "reflecting", [0.3, 2.0], lambda x: (4 - (2 - x) ** 2) / (2 * 0.7)),
])
def test_free_diffusion_mean(left, right, x, exact):
    fp = FirstPassage1D(FLAT, D=0.7, domain=(0.0, 2.0), left=left, right=right)
    x = np.array(x)
    assert fp.mean_time(x).value == pytest.approx(exact(x), rel=1e-13)


def test_free_diffusion_second_moment():
    # reflecting at 0, absorbing at 1, D = 1: <T^2>(x) = (5 - 6 x^2 + x^4) / 12
    fp = FirstPassage1D(FLAT, D=1.0, domain=(0.0, 1.0), left="reflecting", right="absorbing")
    x = np.array([0.0, 0.5])
    assert fp.moments(x, n=2).value[1] == pytest.approx((5 - 6 * x**2 + x**4) / 12, rel=1e-13)


def test_splitting_with_drift_from_potential_and_from_drift():
    v, D, L = 3.0, 0.5, 1.0
    x = np.array([0.1, 0.5, 0.9])
    exact = np.expm1(-v * x / D) / np.expm1(-v * L / D)
    by_potential = FirstPassage1D(lambda y: -v * y / D, D=D, domain=(0.0, L), left="absorbing", right="absorbing")
    by_drift = FirstPassage1D(drift=lambda y: v + 0 * y, D=D, domain=(0.0, L), left="absorbing", right="absorbing")
    assert by_potential.splitting(x).value == pytest.approx(exact, rel=1e-13)
    assert by_drift.splitting(x).value == pytest.approx(exact, rel=1e-13)


@pytest.mark.parametrize("u", [-2.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
def test_noisy_fold_passage_time_is_the_airy_closed_form(u):
    # pi^2 (Ai^2 + Bi^2)(u): Hathcock and Sethna (2021) Eq. (15) in these units
    fp = FirstPassage1D(lambda y: u * y - y**3 / 3, D=1.0, left="natural", right="exit", dlog_psi=lambda y: y * y - u)
    ai, _, bi, _ = airy(u)
    assert float(fp.mean_time(-np.inf).value) == pytest.approx(np.pi**2 * (ai**2 + bi**2), rel=1e-12)


def test_derivative_of_the_potential_is_found_when_not_given():
    fp = FirstPassage1D(lambda y: y - y**3 / 3, D=1.0, left="natural", right="exit")
    ai, _, bi, _ = airy(1.0)
    assert float(fp.mean_time(-np.inf).value) == pytest.approx(np.pi**2 * (ai**2 + bi**2), rel=1e-12)


def test_variance_at_the_fold_is_one_third_of_the_mean_squared():
    # Sigeti and Horsthemke; Lindner et al. (2003) Eqs. (3.4)-(3.5)
    fp = FirstPassage1D(lambda y: -(y**3) / 3, D=1.0, left="natural", right="exit", dlog_psi=lambda y: y * y)
    T1, T2 = fp.moments(-np.inf, n=2).value
    assert T1 == pytest.approx(gamma(1 / 3) ** 2 * (1 / 3) ** (1 / 3), rel=1e-13)
    assert (T2 - T1**2) / T1**2 == pytest.approx(1 / 3, rel=1e-12)


@pytest.mark.parametrize("u, reference", [
    (0.0, 0.3541232444531332932242), (1.0, 0.0820749685574378455751),
    (4.5, 1.956154596511083624017e-06), (8.0, 7.04105074706691e-14),
])
def test_decay_rate_against_65_digit_values(u, reference):
    # Hermite-basis eigenvalues in 65-digit arithmetic (GPT-6 Astra's method, geoews tools/reference_values.py)
    fp = FirstPassage1D(lambda y: u * y - y**3 / 3, D=1.0, left="natural", right="exit", dlog_psi=lambda y: y * y - u)
    assert float(fp.decay_rate().value) == pytest.approx(reference, rel=1e-13)


def test_two_absorbing_ends_with_a_non_flat_potential():
    # the conditional and restricted times must recombine: T = p_b T_b + p_a T_a
    U = lambda x: x**3 / 3 - x  # noqa: E731
    fp = FirstPassage1D(U, D=1.0, domain=(-2.0, 1.0), left="absorbing", right="absorbing")
    x = np.array([-1.0, 0.0, 0.5])
    p = fp.splitting(x).value
    Tb = fp.conditional_mean_time(x, "right").value
    Ta = fp.conditional_mean_time(x, "left").value
    assert fp.mean_time(x).value == pytest.approx(p * Tb + (1 - p) * Ta, rel=1e-12)


def test_refusals():
    with pytest.raises(OutsideDomain):
        FirstPassage1D(FLAT, domain=(0.0, 1.0), left="reflecting", right="reflecting")
    with pytest.raises(OutsideDomain):
        FirstPassage1D(FLAT, domain=(0.0, np.inf), left="absorbing", right="absorbing")
    fp = FirstPassage1D(FLAT, domain=(0.0, 1.0), left="absorbing", right="absorbing")
    with pytest.raises(OutsideDomain):
        fp.mean_time(2.0)
