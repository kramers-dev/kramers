"""Inputs on which the second model's reviews (14 October 2026, two rounds) broke the solvers.

Each test is one of those inputs, with the behaviour the code must now show: the right
answer, or a refusal. Exact values are derived next to each test.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from kramers import OutsideDomain
from kramers.handbook.k0013_first_passage_duality import conditional_time_to_origin
from kramers.handbook.k0024_switching_distribution import switching_distribution
from kramers.handbook.k0027_dhs_rupture_force_moments import dhs_rupture_force_moments
from kramers.solvers import fokker_planck, simulate
from kramers.solvers.quadrature import FirstPassage1D

C = 0.233123  # an irrational-looking position, between any regular sample points


def _gaussian_phi(width):
    return lambda x: -30.0 * np.exp(-(((np.asarray(x) - C) / width) ** 2))


def _exact_splitting(width, x0=0.5):
    # probability of leaving at 1 = integral of exp(-phi) over (0, x0) / over (0, 1); the
    # peak is integrated on its own interval, which a general-purpose quad would miss
    f = lambda y: np.exp(30.0 * np.exp(-(((y - C) / width) ** 2)) - 30.0)  # noqa: E731
    cuts = [0.0, C - 20 * width, C, C + 20 * width, x0]
    left = sum(quad(f, lo, hi, limit=500, epsabs=0, epsrel=1e-13)[0] for lo, hi in zip(cuts[:-1], cuts[1:], strict=True))
    right = quad(f, x0, 1.0, limit=500, epsabs=0, epsrel=1e-13)[0]
    return left / (left + right)


def test_a_narrow_feature_is_found_by_the_probe():
    width = 1e-3  # narrower than the 17-point panels of the first search, wider than the probe spacing
    fp = FirstPassage1D(log_psi=_gaussian_phi(width), D=1.0, domain=(0.0, 1.0), left="absorbing", right="absorbing")
    r = fp.splitting(0.5)
    assert float(r.value) == pytest.approx(_exact_splitting(width), rel=1e-10)


def test_a_feature_narrower_than_the_probe_is_resolved_when_given_as_a_breakpoint():
    width = 1e-5
    fp = FirstPassage1D(log_psi=_gaussian_phi(width), D=1.0, domain=(0.0, 1.0), left="absorbing",
                        right="absorbing", breakpoints=[C])
    assert float(fp.splitting(0.5).value) == pytest.approx(_exact_splitting(width), rel=1e-10)


def test_a_natural_end_is_judged_on_the_speed_density():
    # phi = x, D = exp(alpha x), natural at -infinity, absorbing at 0:
    # T(-1) = int_{-1}^0 e^{-y} int_{-inf}^y e^{(1 - alpha) z} dz dy, finite only for alpha < 1
    alpha = 0.5
    fp = FirstPassage1D(log_psi=lambda x: x, D=lambda x: np.exp(alpha * x), domain=(-np.inf, 0.0),
                        left="natural", right="absorbing")
    exact = (np.exp(alpha) - 1.0) / (alpha * (1.0 - alpha))
    assert float(fp.mean_time(-1.0).value) == pytest.approx(exact, rel=1e-11)
    for alpha in (1.0, 1.1):  # the speed density is not integrable: the time is infinite
        fp = FirstPassage1D(log_psi=lambda x: x, D=lambda x, a=alpha: np.exp(a * x), domain=(-np.inf, 0.0),
                            left="natural", right="absorbing")
        with pytest.raises(OutsideDomain):
            fp.mean_time(-1.0)


def test_a_well_beyond_a_natural_cut_is_brought_into_the_domain():
    # U = x^2 / 2 with a well of depth 800 at x = -30, far beyond where exp(-U) first becomes
    # negligible. T(0) >= int_0^1 e^{U(y)} dy * int_{-30.1}^{-29.9} e^{-U(z)} dz > 0.2 e^{315.6}
    U = lambda x: x**2 / 2 - 800.0 * np.exp(-(((np.asarray(x) + 30.0) / 0.5) ** 2))  # noqa: E731
    fp = FirstPassage1D(U, domain=(-np.inf, 1.0), left="natural", right="absorbing")
    r = fp.mean_time(0.0)
    assert np.log(float(r.value)) > 314.0
    assert r.details["domain"][0] < -30.0


def test_a_periodic_natural_tail_is_not_refused():
    # integrable as x -> -infinity despite rebounds of 40 e-folds; reviewer's second round
    phi = lambda x: -x * x - 20.0 * np.cos(10.0 * (x + 6.59375))  # noqa: E731
    fp = FirstPassage1D(log_psi=phi, domain=(-np.inf, 0.0), left="natural", right="absorbing")
    by_cut = FirstPassage1D(log_psi=phi, domain=(-12.0, 0.0), left="reflecting", right="absorbing")
    assert float(fp.mean_time(-1.0).value) == pytest.approx(float(by_cut.mean_time(-1.0).value), rel=1e-12)


def test_broad_structure_far_beyond_a_cut():
    # reviewer's second round: a broad well at x = -1000 (natural end) and a broad barrier at
    # x = 1000 (exit end), both beyond the old scan; neither may give a small finite answer
    U = lambda x: x * x / 2 - 600000.0 * np.exp(-(((np.asarray(x) + 1000.0) / 20.0) ** 2))  # noqa: E731
    with pytest.raises(OutsideDomain):  # needs more panels than allowed: refused, not wrong
        FirstPassage1D(U, domain=(-np.inf, 1.0), left="natural", right="absorbing").mean_time(0.0)
    phi = lambda x: x**3 / 3 - 4e8 * np.exp(-(((np.asarray(x) - 1000.0) / 20.0) ** 2))  # noqa: E731
    with pytest.raises(OutsideDomain):
        FirstPassage1D(log_psi=phi, domain=(0.0, np.inf), left="reflecting", right="exit").mean_time(0.0)


def test_panels_resolve_a_narrow_feature_of_D():
    # flat phi, D with a narrow deep dip: the speed density has a spike the panels must see.
    # Rayleigh bound with sin(pi x): lambda_1 <= 1.66015758e-29 (reviewer's second round)
    c, w = 0.233123, 1e-4
    D = lambda x: np.exp(-100.0 * np.exp(-(((np.asarray(x) - c) / w) ** 2)))  # noqa: E731
    r = FirstPassage1D(log_psi=lambda x: 0 * x, D=D, domain=(0.0, 1.0), left="absorbing", right="absorbing").decay_rate()
    assert float(r.value) <= 1.66015758e-29
    deeper = lambda x: np.exp(-800.0 * np.exp(-(((np.asarray(x) - c) / w) ** 2)))  # noqa: E731
    with pytest.raises(OutsideDomain):  # D underflows to 0: refused
        FirstPassage1D(log_psi=lambda x: 0 * x, D=deeper, domain=(0.0, 1.0), left="absorbing",
                       right="absorbing", breakpoints=[c]).decay_rate()


@pytest.mark.parametrize("barrier", [100.0, 540.0])
def test_decay_rate_error_covers_two_equal_wells(barrier):
    # an even double well: the ground rate on (-1.5, 1.5) with both ends absorbing equals the
    # rate on (0, 1.5) reflecting at 0, by symmetry; the reported error must cover the difference
    U = lambda x: 10 * (x * x - 1) ** 2 + barrier * np.exp(-((x / 0.2) ** 2))  # noqa: E731
    full = FirstPassage1D(U, domain=(-1.5, 1.5), left="absorbing", right="absorbing").decay_rate()
    half = FirstPassage1D(U, domain=(0.0, 1.5), left="reflecting", right="absorbing").decay_rate()
    assert abs(float(full.value) - float(half.value)) <= float(full.error) + float(half.error)


def test_an_exit_end_with_variable_D_is_refused():
    fp = FirstPassage1D(log_psi=lambda x: x**2 / 2, D=lambda x: 1 + x**2, domain=(0.0, np.inf),
                        left="reflecting", right="exit")
    with pytest.raises(OutsideDomain):
        fp.mean_time(0.5)


def test_a_barrier_beyond_the_exit_cut_is_refused_or_included():
    def phi(width):
        return lambda x: x**3 / 3 - 400.0 * np.exp(-(((np.asarray(x) - 10.0) / width) ** 2))

    def dphi(width):
        return lambda x: x**2 + 800.0 * (np.asarray(x) - 10.0) / width**2 * np.exp(-(((np.asarray(x) - 10.0) / width) ** 2))

    wide = FirstPassage1D(log_psi=phi(0.05), dlog_psi=dphi(0.05), domain=(0.0, np.inf), left="reflecting", right="exit")
    with pytest.raises(OutsideDomain):  # seen by the scan beyond the cut
        wide.mean_time(0.0)
    narrow = FirstPassage1D(log_psi=phi(1e-5), dlog_psi=dphi(1e-5), domain=(0.0, np.inf), left="reflecting",
                            right="exit", breakpoints=[10.0])
    T = float(narrow.mean_time(0.0).value)
    assert np.log(T) > 371.3795  # the reviewer's lower bound from two rectangles of the double integral


def test_decay_rate_with_two_nearly_equal_wells():
    # the reviewer's case: plain power iteration needs about 10,000 steps here; converged value 4.100997190601107e-5
    U = lambda x: 10 * (x**2 - 1) ** 2 + 40 * np.exp(-((x / 0.2) ** 2)) + 0.001 * x  # noqa: E731
    r = FirstPassage1D(U, domain=(-1.5, 1.5), left="absorbing", right="absorbing").decay_rate()
    ref = 4.100997190601107e-5
    assert float(r.value) == pytest.approx(ref, rel=1e-9)
    assert float(r.error) >= abs(float(r.value) - ref) * 0.1  # the error estimate is not falsely tiny


@pytest.mark.parametrize("x0", [0.0, 1.0])
def test_a_start_on_an_absorbing_end_is_refused(x0):
    with pytest.raises(OutsideDomain):
        fokker_planck.solve(lambda x: 0 * x, 1.0, (0.0, 1.0), left="absorbing", right="absorbing", t_end=0.1, x0=x0)
    with pytest.raises(OutsideDomain):
        simulate.first_passage_times(lambda x: 0 * x, 1.0, x0, n_paths=10, dt=1e-3, t_max=1.0, absorbing=(0.0, 1.0))


def test_switching_distribution_refuses_an_incomplete_range():
    with pytest.raises(OutsideDomain):
        switching_distribution(lambda f: np.ones_like(f), 1.0, 0.1)  # only 9.5% switches below F_max
    d = switching_distribution(lambda f: np.ones_like(f), 1.0, 0.1, partial=True)
    assert d.mass == pytest.approx(-np.expm1(-0.1), rel=1e-12)
    full = switching_distribution(lambda f: np.ones_like(f), 1.0, 40.0)
    assert full.mean == pytest.approx(1.0, rel=1e-10)  # exponential law with unit rate


def test_duality_entry_at_extreme_drifts():
    free = (1.0 - 0.7**2) / 6.0  # v = 0: (L^2 - (L - ell)^2) / (6 D), ell = 0.3, L = D = 1
    assert conditional_time_to_origin(0.3, 1.0, 1e-12, 1.0) == pytest.approx(free, rel=1e-12)
    assert np.isfinite(conditional_time_to_origin(0.3, 1.0, -1000.0, 1.0))
    assert conditional_time_to_origin(0.3, 1.0, 1e155, 1.0) == pytest.approx(3e-156, rel=1e-12)  # ell / |v|


def test_rupture_force_moments_refuse_slow_loading():
    with pytest.raises(OutsideDomain):
        dhs_rupture_force_moments(1.0, 1.0, 1.0, 20.0, 2.0 / 3.0)


# --------------------------------------------------------------- third round (15 October)

def test_root_bracket_covers_a_perturbed_double_root():
    # F(x) = 1 - (1 + q) x + q x^2 with q = 1 / (1 + g): roots 1 and 1 + g. Perturbing the
    # coefficients by (1 + eta, 1 - eta, 1 + eta) moves the first root up to 1.0000027639536881
    # (g = 1e-5, eta = 5e-12); the bracket must contain both
    g, eta = 1e-5, 5e-12
    q = 1.0 / (1.0 + g)
    fp = FirstPassage1D(lambda x: 0 * x, domain=(0.0, 1.0), left="absorbing", right="absorbing")
    lower, upper = fp._bracket(1.0, np.log(np.array([1.0, 1.0 + q, q])), np.full(3, eta))
    assert lower <= 1.0 and upper >= 1.0000027639536881


def test_a_tail_that_levels_off_is_refused():
    # phi -> -40 as x -> -infinity: the speed density is not integrable, the mean time infinite
    from scipy.special import expit

    def phi(x):
        s = expit(-(np.asarray(x) + 100.0))
        return (1 - s) * (-np.asarray(x) ** 2) + s * (-40.0)

    with pytest.raises(OutsideDomain):
        FirstPassage1D(log_psi=phi, domain=(-np.inf, 0.0), left="natural", right="absorbing").mean_time(-1.0)


def test_a_zero_of_D_between_samples_is_refused():
    c = np.sqrt(2.0) / 10.0  # D(c) = 0: 1/D is not integrable and the mean time is infinite
    fp = FirstPassage1D(log_psi=lambda x: 0 * x, D=lambda x: (np.asarray(x) - c) ** 2, domain=(0.0, 1.0),
                        left="absorbing", right="absorbing")
    with pytest.raises(OutsideDomain):
        fp.mean_time(0.5)


def test_an_impossible_panel_count_is_refused_at_once():
    import time

    from kramers.solvers import _cheb

    start = time.perf_counter()
    with pytest.raises(RuntimeError):
        _cheb.build_panels(lambda x: 599999.0 * np.asarray(x), 0.0, 1.0)
    assert time.perf_counter() - start < 2.0
