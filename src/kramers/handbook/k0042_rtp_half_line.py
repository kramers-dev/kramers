"""K0042. Run-and-tumble particle on a half-line: exact first-passage density to the origin (Malakar et al.)."""

from __future__ import annotations

import numpy as np
from scipy import integrate, special

from .._checking import combine, relative, statistical
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain


def rtp_first_passage_density(t: float, x: float, v: float, gamma: float, initial: str = "towards") -> float:
    """Density of the time at which a run-and-tumble particle started at x > 0 first reaches the origin.

    dx/dt = v sigma(t), sigma = +-1 reversing at rate gamma, no translational diffusion. In units tau = gamma t,
    xi = gamma x / v (Eq. 35):
      initial velocity away from the origin ("away", the paper's +v):
        f+ = e^-tau / (tau + xi) [xi I0(sqrt(tau^2 - xi^2)) + sqrt((tau - xi)/(tau + xi)) I1(sqrt(tau^2 - xi^2))] for tau > xi
      initial velocity towards the origin ("towards", the paper's -v):
        f- = e^-tau xi / sqrt(tau^2 - xi^2) I1(sqrt(tau^2 - xi^2)) for tau > xi, plus an atom e^-xi at tau = xi
        (arrival without any reversal), which is not included in the returned density.
    The returned value is per unit time (the factor gamma is included). The density is zero for t < x/v.
    """
    if not (x > 0 and v > 0 and gamma > 0 and t >= 0):
        raise OutsideDomain("need x > 0, v > 0, gamma > 0 and t >= 0")
    if initial not in {"towards", "away"}:
        raise OutsideDomain("initial must be 'towards' or 'away'")
    tau, xi = gamma * t, gamma * x / v
    if tau <= xi:
        return 0.0
    z = float(np.sqrt((tau - xi) * (tau + xi)))
    # e^-tau I_n(z) = ive(n, z) e^(z - tau), with z - tau <= 0
    damp = np.exp(z - tau)
    if initial == "towards":
        val = xi / z * special.ive(1, z) * damp if z > 1e-8 else 0.5 * xi * np.exp(-tau)  # I1(z)/z -> 1/2
    else:
        val = (xi * special.ive(0, z) + np.sqrt((tau - xi) / (tau + xi)) * special.ive(1, z)) * damp / (tau + xi)
    return float(gamma * val)


def _laplace(x: float, s: float, initial: str) -> float:
    # Eq. (32) in reduced units (gamma = v = 1); the "towards" transform includes the atom e^{-(1+s) x}
    lam = np.sqrt(s * s + 2 * s)
    return float(np.exp(-lam * x) / (s + 1 + lam)) if initial == "away" else float(np.exp(-lam * x))


def _numeric_transform(x: float, s: float, initial: str) -> float:
    f = lambda t: rtp_first_passage_density(t, x, 1.0, 1.0, initial) * np.exp(-s * t)  # noqa: E731
    val = sum(integrate.quad(f, a, b, epsabs=0, epsrel=1e-12, limit=400)[0]
              for a, b in ((x, x + 1.0), (x + 1.0, x + 20.0), (x + 20.0, np.inf)))
    return val + (np.exp(-(1 + s) * x) if initial == "towards" else 0.0)


def _check_transform():
    rows = []
    for x, s in ((0.7, 0.3), (3.0, 1.5), (1.0, 0.05)):
        for init in ("away", "towards"):
            rows.append((f"{init}, x={x}, s={s}", _numeric_transform(x, s, init), _laplace(x, s, init)))
    return relative(rows, 1e-9, ref="the printed Laplace transform, Eq. (32)")


def _check_mass_and_tail():
    # total probability 1 (with the atom); the tail ~ x/sqrt(2 pi) t^(-3/2) (towards) and (x+1)/sqrt(2 pi) t^(-3/2) (away)
    rows = []
    for x in (0.7, 3.0):
        for init in ("away", "towards"):
            f = lambda t, x=x, init=init: rtp_first_passage_density(t, x, 1.0, 1.0, init)  # noqa: E731
            T = 4e4
            body = sum(integrate.quad(f, a, b, epsabs=0, epsrel=1e-11, limit=400)[0]
                       for a, b in ((x, x + 1), (x + 1, x + 50), (x + 50, 2000), (2000, T)))
            c = (x + 1 if init == "away" else x) / np.sqrt(2 * np.pi)
            tail = 2 * c / np.sqrt(T)  # int_T^inf c t^-3/2 dt
            atom = np.exp(-x) if init == "towards" else 0.0
            rows.append((f"total mass, {init}, x={x}", body + tail + atom, 1.0))
    mass = relative(rows, 1e-5, ref="1")
    # corrections to the tail are of relative order (x + 1)^2 / t, about 2e-4 at t = 1e5
    tail_rows = [(f"tail at t = 1e5, {init}, x={x}", rtp_first_passage_density(1e5, x, 1.0, 1.0, init),
                  (x + 1 if init == "away" else x) / np.sqrt(2 * np.pi) * 1e5 ** -1.5) for x in (0.7, 3.0) for init in ("away", "towards")]
    return combine(mass, relative(tail_rows, 1e-3, ref="Eq. (36)"))


def _check_event_driven():
    # exact runs, no time step: within a run the position is linear, so arrival is detected exactly
    rows = []
    for x0, seed in ((0.5, 421), (1.0, 422), (3.0, 423)):
        for init, sign in (("away", 1.0), ("towards", -1.0)):
            rng = np.random.default_rng(seed + (0 if sign > 0 else 100))
            n = 200_000
            x = np.full(n, x0)
            vel = np.full(n, sign)
            t = np.zeros(n)
            arrive = np.full(n, np.inf)
            alive = np.arange(n)
            while alive.size:
                run = rng.exponential(1.0, alive.size)
                v = vel[alive]
                hit = (v < 0) & (x[alive] + v * run <= 0)
                arrive[alive[hit]] = t[alive[hit]] + x[alive[hit]]
                x[alive] += v * run
                t[alive] += run
                vel[alive] = -v
                alive = alive[~hit & (t[alive] < 25.0)]
            for tt in (2.0, 5.0, 20.0):
                s = float(np.mean(arrive > tt))
                f = lambda u, init=init, x0=x0: rtp_first_passage_density(u, x0, 1.0, 1.0, init)  # noqa: E731
                absorbed = integrate.quad(f, x0, tt, epsabs=0, epsrel=1e-10, limit=400)[0] if tt > x0 else 0.0
                atom = np.exp(-x0) if (init == "towards" and tt > x0) else 0.0
                rows.append((f"survival, {init}, x={x0}, t={tt}", 1.0 - absorbed - atom, s, np.sqrt(s * (1 - s) / n)))
    return statistical(rows)


register(Entry(
    id="K0042",
    slug="rtp-half-line-first-passage",
    title="Run-and-tumble particle on a half-line: exact first-passage density to the origin",
    area="active",
    kind="exact",
    statement=(
        "A particle that moves at speed v and reverses direction at rate gamma, started at x > 0, first reaches the origin "
        "with a density given by modified Bessel functions of sqrt(t^2 - x^2) (in units 1/gamma and v/gamma). If it starts "
        "towards the origin it arrives at t = x/v without reversing with probability e^(-gamma x/v); at long times the "
        "density decays as t^(-3/2), like Brownian motion."
    ),
    model=r"\dot x=v\,\sigma(t),\ \sigma=\pm1\ \text{reversing at rate}\ \gamma,\ x(0)=x>0,\ \text{absorbing at}\ 0",
    formula=(r"f_+=\frac{e^{-\tau}}{\tau+\xi}\left[\xi I_0(\sqrt{\tau^2-\xi^2})+\sqrt{\tfrac{\tau-\xi}{\tau+\xi}}I_1(\sqrt{\tau^2-\xi^2})\right],\ "
             r"f_-=e^{-\tau}\frac{\xi}{\sqrt{\tau^2-\xi^2}}I_1(\sqrt{\tau^2-\xi^2})+e^{-\tau}\delta(\tau-\xi)"),
    function=rtp_first_passage_density,
    parameters={"t": "time", "x": "start distance from the origin", "v": "speed",
                "gamma": "reversal rate (the velocity flips sign at this rate)",
                "initial": "'towards' (the paper's -v) or 'away' (+v) from the origin"},
    sources=(
        Source(
            citation=("K. Malakar, V. Jemseena, A. Kundu, K. V. Kumar, S. Sabhapandit, S. N. Majumdar, S. Redner and A. Dhar, "
                      "J. Stat. Mech. (2018) 043215"),
            doi="10.1088/1742-5468/aab84f",
            locator="Eq. (35) p. 15 (densities); Eq. (32) p. 14 (transforms); Eq. (36) (tails); model Eq. (1) "
                    "(arXiv:1711.08474v3 numbering)",
            version_read="arXiv:1711.08474v3 PDF pp. 15-16 as page images and the TeX source, opened 22 Nov 2026 (two readers); "
                         "journal numbering not checked",
            printed=(r"f_+(x,t)=\frac{e^{-t}}{t+x}\left[ x ~I_0\big( \sqrt{t^2-x^2} \big) + \frac{\sqrt{t-x}}{\sqrt{t+x}}~I_1\big( "
                     r"\sqrt{t^2-x^2} \big)\right]\theta(t-x);\ f_-(x,t)=e^{- t} \frac{ x}{\sqrt{t^2-x^2}} \,I_1 \big( \sqrt{t^2-x^2} "
                     r"\big)~ \theta(t-x)\,+ e^{- t} \delta(t-x)"),
            notation=("their t and x are gamma t and gamma x / v; +v (moving away) and -v (towards) are the initial velocities; gamma "
                      "is the reversal rate, half the tumble rate of K0033's convention in one dimension. The diffusive extension "
                      "(their Eqs. 38-43) is not included"),
        ),
    ),
    validity="One dimension, speed v, Poissonian reversals, no translational diffusion, the origin absorbing. Exact.",
    checks=(
        Check("against the printed Laplace transforms", "Eq. (32), with the atom for a start towards the origin", _check_transform),
        Check("total probability and the long-time tail", "1; Eq. (36)", _check_mass_and_tail),
        Check("against exact event-driven simulation", "piecewise-linear runs, 200,000 particles per case", _check_event_driven, slow=True),
    ),
    aliases=("persistent random walk first passage", "telegraph process first passage", "active particle first passage", "RTP half-line"),
    related=("K0023", "K0033"),
    added="2026-11-22",
))
