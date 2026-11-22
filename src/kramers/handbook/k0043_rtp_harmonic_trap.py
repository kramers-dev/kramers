"""K0043. Run-and-tumble particle in a harmonic trap, absorbing wall at the centre: survival transform and decay exponent (Dhar et al.)."""

from __future__ import annotations

import numpy as np
from scipy import optimize

from .._checking import combine, relative, statistical
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain


def _F(a: float, b: float, c: float, z: float) -> float:
    """Gauss hypergeometric 2F1(a, b; c; z) for 0 <= z <= 1/2 by its series, summed term by term.

    scipy.special.hyp2f1 returns values off by thirty orders of magnitude at some of the parameters needed here
    (for example b = 2c near 200 at z = 1/2, checked 22 Nov 2026 with scipy 1.18.1 against mpmath), so it is not
    used. For z <= 1/2 the series converges at least geometrically; when a is a small negative number (the root
    search for theta) all terms after the first have one sign, so there is no cancellation. A running scale keeps
    large intermediate terms from overflowing.
    """
    if not 0.0 <= z <= 0.5:
        raise OutsideDomain("the series is used only for 0 <= z <= 1/2")
    if c <= 0 and float(c).is_integer():
        raise OutsideDomain("c must not be zero or a negative integer")
    term, total, scale = 1.0, 1.0, 0.0
    n = 0
    while True:
        term *= (a + n) * (b + n) / ((c + n) * (n + 1.0)) * z
        total += term
        n += 1
        if abs(total) > 1e250:
            term, total, scale = term * 1e-250, total * 1e-250, scale + 250.0
        if (abs(term) <= 1e-17 * abs(total) and n > abs(a) + abs(b) + 10) or n > 200_000 or term == 0.0:
            break
    return float(total * 10.0**scale) if scale else float(total)


def _D(s: float, gamma: float, mu: float) -> float:
    # Eq. (83)
    return float((gamma + s) * _F(s / mu, (s + 2 * gamma) / mu, (s + gamma + mu) / mu, 0.5)
                 + 0.5 * s * (s + 2 * gamma) / (s + gamma + mu)
                 * _F((s + mu) / mu, (s + 2 * gamma + mu) / mu, (s + gamma + 2 * mu) / mu, 0.5))


def rtp_harmonic_survival_transform(s: float, x0: float, v0: float, mu: float, gamma: float, initial: str = "towards") -> float:
    """Laplace transform Q~(x0, s) = int_0^inf e^{-st} Q(x0, t) dt of the survival probability of a run-and-tumble
    particle dx/dt = -mu x + v0 sigma(t) (sigma = +-1 reversing at rate gamma), absorbed at the trap centre x = 0.

    initial = "towards" (the paper's -v0), Eq. (84):
        Q~- = (1/s) [1 - F(s/mu, (s+2g)/mu, (s+g)/mu, z) / F(s/mu, (s+2g)/mu, (s+g)/mu, 1/2)],  z = (1 - mu x0/v0)/2;
    initial = "away" (+v0), from Eqs. (82), (83) and (67):
        Q~+ = 1/s - (g/s) F(s/mu, (s+2g)/mu, (s+g+mu)/mu, z) / D(s, g, mu).
    F is the Gauss hypergeometric function. Requires 0 < x0 < v0/mu (inside the region the particle can reach).
    E[e^{-sT}] = 1 - s Q~(s).
    """
    if not (v0 > 0 and mu > 0 and gamma > 0 and s > 0):
        raise OutsideDomain("need v0, mu, gamma and s positive")
    if not 0 < x0 < v0 / mu:
        raise OutsideDomain("need 0 < x0 < v0/mu")
    if initial not in {"towards", "away"}:
        raise OutsideDomain("initial must be 'towards' or 'away'")
    z = 0.5 * (1.0 - mu * x0 / v0)
    a, b = s / mu, (s + 2 * gamma) / mu
    if initial == "towards":
        c = (s + gamma) / mu
        return float((1.0 - _F(a, b, c, z) / _F(a, b, c, 0.5)) / s)
    return float(1.0 / s - (gamma / s) * _F(a, b, (s + gamma + mu) / mu, z) / _D(s, gamma, mu))


def rtp_harmonic_mean_time(x0: float, v0: float, mu: float, gamma: float, initial: str = "towards") -> float:
    """Mean time to reach the centre, Q~(x0, s -> 0), by a symmetric difference around s = 0 (Q~ is analytic there)."""
    h = 1e-4 * min(mu, gamma)
    f = lambda s: _q_any_s(s, x0, v0, mu, gamma, initial)  # noqa: E731
    return float(0.5 * (f(h) + f(-h)))


def _q_any_s(s: float, x0: float, v0: float, mu: float, gamma: float, initial: str) -> float:
    # the same formulas, for small negative s as well (used only to centre the difference at s = 0)
    z = 0.5 * (1.0 - mu * x0 / v0)
    a, b = s / mu, (s + 2 * gamma) / mu
    if initial == "towards":
        c = (s + gamma) / mu
        return (1.0 - _F(a, b, c, z) / _F(a, b, c, 0.5)) / s
    return 1.0 / s - (gamma / s) * _F(a, b, (s + gamma + mu) / mu, z) / _D(s, gamma, mu)


def rtp_harmonic_decay_exponent(mu: float, gamma: float) -> float:
    """theta, the decay rate of the survival at late times: the lowest positive root of
    F(-theta/mu, (2 gamma - theta)/mu, (gamma - theta)/mu, 1/2) = 0 (Eq. 88). It lies below min(gamma, mu)."""
    if not (mu > 0 and gamma > 0):
        raise OutsideDomain("need mu and gamma positive")
    top = min(gamma, mu) * (1.0 - 1e-9)
    g = lambda th: _F(-th / mu, (2 * gamma - th) / mu, (gamma - th) / mu, 0.5)  # noqa: E731
    grid = np.linspace(top * 1e-6, top, 401)
    vals = np.array([g(t) for t in grid])
    idx = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
    if idx.size == 0:
        raise OutsideDomain("no root found below min(gamma, mu)")
    i = int(idx[0])
    return float(optimize.brentq(g, grid[i], grid[i + 1], xtol=1e-14, rtol=1e-14))


def _sample(x0: float, v0: float, mu: float, gamma: float, sign: float, n: int, seed: int, t_cap: float) -> np.ndarray:
    # exact event-driven paths: between reversals x(t) = sigma v0/mu + (x - sigma v0/mu) e^{-mu t}; with sigma = -1 the
    # centre is reached after (1/mu) ln(1 + mu x / v0), which is tested against the length of the run
    rng = np.random.default_rng(seed)
    x = np.full(n, x0)
    sig = np.full(n, sign)
    t = np.zeros(n)
    arrive = np.full(n, np.inf)
    alive = np.arange(n)
    while alive.size:
        run = rng.exponential(1.0 / gamma, alive.size)
        xs, ss = x[alive], sig[alive]
        reach = np.where(ss < 0, np.log1p(mu * xs / v0) / mu, np.inf)
        hit = reach <= run
        arrive[alive[hit]] = t[alive[hit]] + reach[hit]
        fixed = ss * v0 / mu
        x[alive] = fixed + (xs - fixed) * np.exp(-mu * run)
        t[alive] += run
        sig[alive] = -ss
        alive = alive[~hit & (t[alive] < t_cap)]
    return arrive


def _check_event_driven():
    rows = []
    mu = v0 = 1.0
    for gamma, x0, s, seed in ((0.5, 0.3, 0.5, 431), (2.0, 0.8, 1.0, 432)):
        for init, sign in (("towards", -1.0), ("away", 1.0)):
            T = _sample(x0, v0, mu, gamma, sign, 200_000, seed + (0 if sign < 0 else 50), t_cap=200.0)
            e = np.exp(-s * T)
            rows.append((f"E exp(-sT), {init}, gamma={gamma}, x0={x0}, s={s}",
                         1.0 - s * rtp_harmonic_survival_transform(s, x0, v0, mu, gamma, init), float(e.mean()),
                         float(e.std(ddof=1) / np.sqrt(e.size))))
            fin = np.isfinite(T)
            rows.append((f"mean time, {init}, gamma={gamma}, x0={x0}", rtp_harmonic_mean_time(x0, v0, mu, gamma, init),
                         float(T[fin].mean()), float(T[fin].std(ddof=1) / np.sqrt(fin.sum()))))
    # theta: the late-time decay rate of the simulated survival, by maximum likelihood on the times beyond t0 (the
    # residual times are exponential with rate theta once faster modes have died out), with censoring at the cap;
    # t0 = 5/theta, and 2,000,000 paths leave about 10^4 events, a standard error near 1%
    for gamma, seed in ((0.5, 441), (1.0, 442), (3.0, 443)):
        theta = rtp_harmonic_decay_exponent(1.0, gamma)
        t0, cap = 5.0 / theta, 14.0 / theta
        T = _sample(0.5, 1.0, 1.0, gamma, -1.0, 2_000_000, seed, t_cap=cap)
        late = T > t0
        events = int(np.sum(late & (T <= cap)))
        exposure = float(np.sum(np.minimum(T[late], cap) - t0))
        rate = events / exposure
        rows.append((f"theta, gamma={gamma} (rate of the survival beyond t = 5/theta)", theta, rate, rate / np.sqrt(events)))
    return statistical(rows)


def _check_limits():
    # theta = mu U(gamma/mu) with U(x) -> x for slow reversals and U -> 1 for fast ones (Eqs. 89-90). Both limits
    # are approached slowly (1 - theta/mu is still 0.010 at gamma/mu = 3000), so they are tested where they are
    # reached to 2%, and the approach to 1 is tested for monotonicity
    th = [rtp_harmonic_decay_exponent(1.0, g) for g in (10.0, 100.0, 3000.0)]
    rows = [("slow reversals: theta/gamma at gamma/mu = 0.001", rtp_harmonic_decay_exponent(1.0, 0.001) / 0.001, 1.0),
            ("fast reversals: theta/mu at gamma/mu = 3000", th[2], 1.0),
            ("theta/mu rises towards 1 (gamma/mu = 10, 100, 3000)", float(th[0] < th[1] < th[2] < 1.0), 1.0)]
    lim = relative(rows, 0.02, ref="Eq. (90)")
    # gamma = mu: the root of Eq. (88) is exactly mu/2 (the paper's value 0.5 at gamma = mu = 1, Fig. 12)
    half = relative([("gamma = mu = 1", rtp_harmonic_decay_exponent(1.0, 1.0), 0.5)], 1e-8, ref="1/2")
    # the transform tends to the mean as s -> 0 (finite: the particle always reaches the centre)
    cont = relative([(f"Q~(s = 1e-6) against the mean, {init}", rtp_harmonic_survival_transform(1e-6, 0.3, 1.0, 1.0, 0.5, init),
                      rtp_harmonic_mean_time(0.3, 1.0, 1.0, 0.5, init)) for init in ("towards", "away")], 1e-4, ref="s -> 0")
    return combine(lim, half, cont)


def _check_hypergeometric():
    # the series against mpmath's 2F1 at 30 digits, over the parameters this entry uses (mpmath is in the
    # "reference" and "dev" extras)
    import mpmath as mp

    mp.mp.dps = 30
    rows = []
    for a, b, c, z in ((0.5, 1.5, 1.0, 0.35), (1.0, 5.0, 3.0, 0.5), (-0.00155, 199.99845, 99.99845, 0.5), (-0.9, 199.1, 99.1, 0.5),
                       (0.3, 4.3, 2.3, 0.0), (2.0, 202.0, 103.0, 0.5), (-0.3, 0.7, 0.2, 0.5), (8.0, 12.0, 11.0, 0.25)):
        rows.append((f"2F1({a}, {b}; {c}; {z})", _F(a, b, c, z), float(mp.hyp2f1(a, b, c, z))))
    return relative(rows, 1e-12, ref="mpmath, 30 digits")


register(Entry(
    id="K0043",
    slug="rtp-harmonic-trap",
    title="Run-and-tumble particle in a harmonic trap, absorbed at the centre: survival transform and decay exponent",
    area="active",
    kind="exact",
    statement=(
        "A run-and-tumble particle in a harmonic trap, dx/dt = -mu x + v0 sigma(t), absorbed when it reaches the trap centre, "
        "has a survival probability whose Laplace transform is a ratio of Gauss hypergeometric functions. At late times it "
        "decays as exp(-theta t), where theta solves a hypergeometric equation and lies between the reversal rate (slow "
        "reversals) and mu (fast reversals, the passive limit)."
    ),
    model=r"\dot x=-\mu x+v_0\sigma(t),\ \sigma=\pm1\ \text{reversing at rate}\ \gamma,\ 0<x_0<v_0/\mu,\ \text{absorbed at}\ x=0",
    formula=(r"\tilde Q_-(x_0,s)=\frac1s\left[1-\frac{F\left(\frac s\mu,\frac{s+2\gamma}\mu,\frac{s+\gamma}\mu,\frac12(1-\frac{\mu x_0}{v_0})\right)}"
             r"{F\left(\frac s\mu,\frac{s+2\gamma}\mu,\frac{s+\gamma}\mu,\frac12\right)}\right],\quad "
             r"F\left(-\tfrac\theta\mu,\tfrac{2\gamma-\theta}\mu,\tfrac{\gamma-\theta}\mu,\tfrac12\right)=0"),
    function=rtp_harmonic_survival_transform,
    parameters={"s": "Laplace variable", "x0": "start position, 0 < x0 < v0/mu", "v0": "self-propulsion speed",
                "mu": "trap stiffness over friction (relaxation rate)", "gamma": "reversal rate",
                "initial": "'towards' (the paper's -v0) or 'away' (+v0) from the centre"},
    sources=(
        Source(
            citation="A. Dhar, A. Kundu, S. N. Majumdar, S. Sabhapandit and G. Schehr, Phys. Rev. E 99, 032132 (2019)",
            doi="10.1103/PhysRevE.99.032132",
            locator=("Eq. (84) p. 12 (start towards the centre); Eqs. (82), (83) and the shift (67) (start away); Eq. (88) "
                     "(decay exponent); Eqs. (89)-(90) (limits) (arXiv:1811.03808v2 numbering)"),
            version_read=("arXiv:1811.03808v2 PDF p. 12 as a page image and the TeX source, opened 22 Nov 2026 (two readers); "
                          "v2 was revised in June 2022, after publication; journal text not checked"),
            printed=(r"\tilde{Q}_- (x_0,s) =\frac{1}{s} \left[1- \frac{F\left( \frac{s}{\mu}, \frac{s+2\gamma}{\mu}, \frac{s+\gamma}{\mu},"
                     r"\frac{1}{2}(1-\frac{\mu x_0}{v_0})\right)}{F\left( \frac{s}{\mu}, \frac{s+2\gamma}{\mu}, \frac{s+\gamma}{\mu},\frac{1}{2} "
                     r"\right)}\right];\ \tilde{q}_+(z,s)= -\frac{\gamma}{s} \frac{F\left( \frac{s}{\mu}, \frac{s+2\gamma}{\mu}, "
                     r"\frac{s+\gamma+\mu}{\mu},z \right)}{\mathcal{D}(s,\gamma,\mu)}"),
            notation=("same symbols; F = 2F1; gamma is the reversal rate. For a start away from the centre this entry uses Eqs. (82) "
                      "and (67), Q~+ = 1/s - (gamma/s) F/D. Eq. (85) as printed in arXiv v2, (gamma/s)[1 - F/D], diverges as s -> 0 "
                      "for gamma != 1 (a finite mean needs a finite limit) and misses the simulation; read as a question for "
                      "the authors until the journal version is checked"),
        ),
    ),
    validity=("One dimension, harmonic trap, constant self-propulsion speed with Poissonian reversals, no translational "
              "diffusion, absorbing wall at the trap centre, start inside the reachable region 0 < x0 < v0/mu. Exact."),
    checks=(
        Check("the hypergeometric series", "mpmath's 2F1 at 30 digits", _check_hypergeometric),
        Check("limits of the decay exponent and of the transform", "Eq. (90); theta = 1/2 at gamma = mu = 1; s -> 0", _check_limits),
        Check("against exact event-driven simulation", "piecewise-exponential runs, 200,000 to 400,000 particles; transform, "
              "mean and decay exponent", _check_event_driven, slow=True),
    ),
    aliases=("active particle in a harmonic trap", "run-and-tumble in a potential", "persistence exponent of an active particle",
             "first passage of a confined active particle"),
    related=("K0037", "K0042", "K0033"),
    added="2026-11-22",
))
