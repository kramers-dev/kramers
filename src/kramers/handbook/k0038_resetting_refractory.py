"""K0038. Diffusion with resetting and a refractory period after each reset: mean time to the target (Evans and Majumdar)."""

from __future__ import annotations

import numpy as np
from scipy import integrate, special

from .._checking import relative, statistical
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from .k0015_resetting_mean_time import resetting_mean_time


def _Q_tilde(L: float, D: float, s: float) -> tuple[float, float]:
    """Laplace transform of the free survival from distance L, Q~0(s) = (1 - e^{-L sqrt(s/D)})/s, and its s-derivative."""
    q = L * np.sqrt(s / D)
    one_minus = -np.expm1(-q)
    Q = one_minus / s
    dQ = (0.5 * q * np.exp(-q) - one_minus) / (s * s)
    return float(Q), float(dQ)


def refractory_resetting_mean_time(L: float, D: float, r: float, a0: float = 0.0, a1: float = 0.0) -> float:
    """Mean time for a diffusing particle to reach a target at distance L, resetting at rate r, with a refractory
    period after each reset whose mean, given the time t since the previous reset, is a0 + a1 t.

    Evans and Majumdar's Eq. (41) with the free-diffusion survival:
        T = [Q~0(r) + r (a0 Q~0(r) - a1 Q~0'(r))] / (1 - r Q~0(r)),  Q~0(s) = (1 - e^{-L sqrt(s/D)})/s.
    The particle is frozen at the reset point during a refractory period; only the conditional mean of
    the period enters. a0 = a1 = 0 is entry K0015; for a1 = 0 the mean is (1 + r a0)(e^{L sqrt(r/D)} - 1)/r
    (derived here, not printed). <x^2> = 2 D t.
    """
    if not (L > 0 and D > 0 and r > 0):
        raise OutsideDomain("need L > 0, D > 0 and r > 0")
    if a0 < 0 or a1 < 0:
        raise OutsideDomain("the mean refractory period a0 + a1 t cannot be negative")
    Q, dQ = _Q_tilde(L, D, r)
    # 1 - r Q~0(r) = e^{-L sqrt(r/D)}: written exactly, so large r L^2/D keeps its digits
    denom = float(np.exp(-L * np.sqrt(r / D)))
    return float((Q + r * (a0 * Q - a1 * dQ)) / denom)


def _eq36(L: float, D: float, r: float, a0: float, a1: float) -> float:
    # Eq. (36) evaluated in the time domain: h(t) = r e^{-rt}, g(t) = e^{-rt}, <tau|t> = a0 + a1 t,
    # Q0(L, t) = erf(L / sqrt(4 D t)) (free survival); quadrature, no transforms
    Q0 = lambda t: special.erf(L / np.sqrt(4 * D * t)) if t > 0 else 1.0  # noqa: E731
    num1 = integrate.quad(lambda t: np.exp(-r * t) * Q0(t), 0, np.inf, epsabs=0, epsrel=1e-12, limit=400)[0]
    num2 = integrate.quad(lambda t: (a0 + a1 * t) * r * np.exp(-r * t) * Q0(t), 0, np.inf, epsabs=0, epsrel=1e-12, limit=400)[0]
    den = 1.0 - integrate.quad(lambda t: r * np.exp(-r * t) * Q0(t), 0, np.inf, epsabs=0, epsrel=1e-12, limit=400)[0]
    return (num1 + num2) / den


def _check_time_domain():
    rows = []
    for L, D, r, a0, a1 in ((1.0, 1.0, 1.0, 0.5, 0.0), (1.0, 1.0, 2.0, 0.0, 0.3), (1.0, 1.0, 0.5, 0.2, 0.4),
                            (0.6, 0.4, 3.0, 1.0, 1.0)):
        rows.append((f"L={L}, D={D}, r={r}, a0={a0}, a1={a1}", refractory_resetting_mean_time(L, D, r, a0, a1),
                     _eq36(L, D, r, a0, a1)))
    return relative(rows, 1e-8, ref="Eq. (36) by quadrature in the time domain")


def _check_no_refractory():
    rows = [(f"L={L}, D={D}, r={r}", refractory_resetting_mean_time(L, D, r), resetting_mean_time(L, r, D))
            for L, D, r in ((1.0, 1.0, 1.0), (0.3, 0.2, 5.0), (2.0, 3.0, 0.01))]
    return relative(rows, 1e-12, ref="entry K0015")


def _check_event_driven():
    # exact sampling, no time step: free passage time L^2 / (2 D Z^2) with Z standard normal, competing with an
    # exponential reset time; after a reset the particle waits a refractory period (exponential or fixed, with
    # mean a0 + a1 t, t the time since the last restart) and starts again from the reset point
    rows = []
    for L, D, r, a0, a1, kind, seed in ((1.0, 1.0, 1.0, 0.5, 0.0, "exponential", 381), (1.0, 1.0, 1.0, 0.5, 0.0, "fixed", 382),
                                        (1.0, 1.0, 2.0, 0.0, 0.3, "exponential", 383), (1.0, 1.0, 0.5, 0.2, 0.4, "exponential", 384)):
        rng = np.random.default_rng(seed)
        n = 200_000
        total = np.zeros(n)
        alive = np.arange(n)
        while alive.size:
            hit = L * L / (2.0 * D * rng.standard_normal(alive.size) ** 2)
            reset = rng.exponential(1.0 / r, alive.size)
            done = hit < reset
            total[alive[done]] += hit[done]
            rest = alive[~done]
            mean_tau = a0 + a1 * reset[~done]
            tau = rng.exponential(1.0, rest.size) * mean_tau if kind == "exponential" else mean_tau
            total[rest] += reset[~done] + tau
            alive = rest
        rows.append((f"L={L}, r={r}, a0={a0}, a1={a1}, {kind} period", refractory_resetting_mean_time(L, D, r, a0, a1),
                     float(total.mean()), float(total.std(ddof=1) / np.sqrt(n))))
    return statistical(rows)


register(Entry(
    id="K0038",
    slug="resetting-refractory",
    title="Diffusion with resetting and a refractory period: mean time to the target",
    area="resetting",
    kind="exact",
    statement=(
        "If after each reset the particle must wait a random refractory period before it moves again, the mean time to "
        "reach the target depends only on the mean of that period (given the time since the previous reset). With "
        "Poissonian resetting at rate r and a constant mean period, it is (1 + r a0) times the mean without the period."
    ),
    model=(r"\text{free diffusion from } X_r=L\ \text{to an absorbing target; resets at rate } r\text{, each followed by a "
           r"frozen period } \tau,\ \langle\tau|t\rangle=a_0+a_1t"),
    formula=(r"T=\frac{\tilde Q_0(r)+r\left[a_0\tilde Q_0(r)-a_1\tilde Q_0'(r)\right]}{1-r\tilde Q_0(r)},\quad "
             r"\tilde Q_0(s)=\frac{1-e^{-L\sqrt{s/D}}}{s}"),
    function=refractory_resetting_mean_time,
    parameters={"L": "distance from the reset point to the target", "D": "diffusion coefficient (<x^2> = 2Dt)",
                "r": "resetting rate", "a0": "mean refractory period (constant part)",
                "a1": "growth of the mean period with the time since the previous reset"},
    sources=(
        Source(
            citation="M. R. Evans and S. N. Majumdar, J. Phys. A: Math. Theor. 52, 01LT01 (2019)",
            doi="10.1088/1751-8121/aaf080",
            locator=("Eq. (41) p. 12 (Poissonian reset); Eq. (36) p. 11 (general); Eq. (23) p. 8 (the correlated example "
                     "<tau|t> = a1 t); definitions Eqs. (20)-(22) (arXiv:1809.01551v2 numbering)"),
            version_read="arXiv:1809.01551v2 PDF pp. 10-12 as page images and the TeX source, opened 22 Nov 2026 (two readers); journal numbering not checked",
            printed=(r"T(X_r)=\frac{\tilde{Q}_0(X_r,r)+r\sum_{n=0}(-1)^na_n\tilde{Q}_0^{(n)}(X_r,r)}{1-r\tilde{Q}_0(X_r,r)}"),
            notation=("<tau|t> = sum_n a_n t^n; this entry keeps n = 0, 1 and takes free diffusion between resets, with the "
                      "survival transform of entry K0015 (the specialisation is done here). The second term of Eq. (38) is "
                      "printed with the integration variable d tau; reading it as dt reproduces Eq. (36)"),
        ),
    ),
    validity=("Free diffusion in one dimension between resets, target absorbing, Poissonian resetting, each reset followed by "
              "a refractory period during which the particle stays at the reset point; the process does not start with a "
              "period. Exact. Only the conditional mean of the period enters."),
    checks=(
        Check("against the time-domain formula", "Eq. (36) by quadrature, no transforms", _check_time_domain),
        Check("without a refractory period", "entry K0015", _check_no_refractory),
        Check("against exact event-driven sampling", "200,000 paths each; exponential and fixed periods; independent and "
              "correlated cases", _check_event_driven, slow=True),
    ),
    aliases=("refractory period", "resetting with a waiting time", "resetting with a home time", "frozen after reset"),
    related=("K0015", "K0017", "K0036"),
    added="2026-11-22",
))
