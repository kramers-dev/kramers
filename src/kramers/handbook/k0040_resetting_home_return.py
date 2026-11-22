"""K0040. Diffusive search with resetting and a return home at constant speed: mean time to the target (Pal, Kusmierz and Reuveni)."""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize

from .._checking import combine, relative, statistical
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from .k0015_resetting_mean_time import resetting_mean_time


def home_return_mean_time(L: float, D: float, v_return: float, r: float) -> float:
    """Mean time for a searcher diffusing from home (x = 0) to find a target at distance L, when it gives up at
    rate r and walks back home at speed v_return (blind to the target on the way), then searches again.

    <T_r> = (e^z - 1)/r + (L/v_return) [2 sinh(z)/z - 1],  z = sqrt(r L^2 / D)   (Eqs. 9-10).
    The first term is the search (entry K0015); the second is the time spent walking home. <x^2> = 2 D t.
    """
    if not (L > 0 and D > 0 and r > 0 and v_return > 0):
        raise OutsideDomain("need L, D, r and v_return all positive (v_return = infinity is entry K0015)")
    z = float(np.sqrt(r * L * L / D))
    shz = np.sinh(z) / z if z > 1e-4 else 1.0 + z * z / 6.0 + z**4 / 120.0
    return float(np.expm1(z) / r + (L / v_return) * (2.0 * shz - 1.0))


def optimal_rate_home_return(L: float, D: float, v_return: float) -> float:
    """The rate that minimises the mean, r* = z*^2 D / L^2, with z* the root of Eq. (11):
    (2/z^2) (1 - e^-z - z/2) / [(1 - 1/z) + (1 + 1/z) e^-2z] = tau_b/tau_d, tau_b = L/v_return, tau_d = L^2/D."""
    if not (L > 0 and D > 0 and v_return > 0):
        raise OutsideDomain("need L, D and v_return positive")
    ratio = (L / v_return) / (L * L / D)

    def F(z: float) -> float:
        # Eq. (11) with its denominator written as [z - 1 + (z + 1) e^{-2z}]/z; near z = 0 that bracket is of order
        # z^3 and is formed from order-one terms, so z is kept above 1e-3, where about ten digits survive
        num = -np.expm1(-z) - z / 2.0
        den = z - 1.0 + (z + 1.0) * np.exp(-2.0 * z)
        return float(2.0 / z * num / den - ratio)

    # F falls from about 3/(2 z^3) at small z to 0 at the free optimum z = 1.5936 (instant returns)
    z_free = optimize.brentq(lambda z: z / 2 - 1 + np.exp(-z), 0.5, 3.0, xtol=1e-15)
    if F(1e-3) < 0:
        raise OutsideDomain("returns so slow (tau_b/tau_d above about 1.5e9) that the optimal z is below 1e-3; "
                            "use z* = (3 tau_d / (2 tau_b))^(1/3) from the small-z form of Eq. (11)")
    zs = optimize.brentq(F, 1e-3, z_free - 1e-12, xtol=1e-15, rtol=1e-15)
    return float(zs * zs * D / (L * L))


def _eq6(L: float, D: float, v: float, r: float) -> float:
    # the general Eq. (6) with <W> = 0, evaluated by quadrature from the free-range propagator Eq. (8):
    # T~(r) = e^{-L sqrt(r/D)} (free first passage from distance L), and <tau>_r = int |x|/v phi_r(x) dx with
    # phi_r proportional to the Laplace transform of G0(x, t) at s = r, on x < L
    a = np.sqrt(r / D)
    Tt = np.exp(-a * L)
    G = lambda x: (np.exp(-a * abs(x)) - np.exp(-a * abs(2 * L - x))) / (2 * np.sqrt(r * D))  # noqa: E731
    norm = integrate.quad(G, -np.inf, 0, epsabs=0, epsrel=1e-12)[0] + integrate.quad(G, 0, L, epsabs=0, epsrel=1e-12)[0]
    first = integrate.quad(lambda x: abs(x) / v * G(x), -np.inf, 0, epsabs=0, epsrel=1e-12)[0]
    second = integrate.quad(lambda x: abs(x) / v * G(x), 0, L, epsabs=0, epsrel=1e-12)[0]
    tau_r = (first + second) / norm
    return float((1 - Tt) / (r * Tt) + (1 - Tt) / Tt * tau_r)


def _check_general_formula():
    rows = [(f"L={L}, D={D}, v={v}, r={r}", home_return_mean_time(L, D, v, r), _eq6(L, D, v, r))
            for L, D, v, r in ((1.0, 0.5, 1.0, 2.0), (1.0, 1.0, 0.3, 0.5), (2.0, 0.7, 5.0, 1.2))]
    return relative(rows, 1e-9, ref="Eq. (6) by quadrature with the propagator of Eq. (8)")


def _check_limits_and_optimum():
    fast = relative([(f"v = 1e12, L={L}, D={D}, r={r}", home_return_mean_time(L, D, 1e12, r), resetting_mean_time(L, r, D))
                     for L, D, r in ((1.0, 0.5, 2.0), (0.7, 2.0, 0.3))], 1e-10, ref="entry K0015")
    rows = []
    for L, D, v in ((1.0, 0.5, 1.0), (1.0, 1.0, 0.25), (2.0, 0.3, 3.0)):
        rs = optimal_rate_home_return(L, D, v)
        res = optimize.minimize_scalar(lambda lr, L=L, D=D, v=v: home_return_mean_time(L, D, v, float(np.exp(lr))),
                                       bracket=(np.log(rs) - 1, np.log(rs), np.log(rs) + 1), tol=1e-12)
        rows.append((f"L={L}, D={D}, v={v}", rs, float(np.exp(res.x))))
    return combine(fast, relative(rows, 1e-6, ref="numerical minimum of the mean"))


def _check_simulation():
    # Euler-Maruyama from home with a bridge-corrected absorbing target at L; at each reset the clock gains the
    # walk home |x|/v and the searcher restarts at 0 (blind on the way back)
    rows = []
    for L, D, v, r, seed in ((1.0, 0.5, 1.0, 2.0, 401), (1.0, 1.0, 0.3, 0.5, 402)):
        rng = np.random.default_rng(seed)
        n, dt = 100_000, 2e-4
        x = np.zeros(n)
        clock = np.zeros(n)
        done_at = np.full(n, np.nan)
        alive = np.arange(n)
        sig = np.sqrt(2 * D * dt)
        p_reset = -np.expm1(-r * dt)
        while alive.size:
            xa = x[alive]
            xn = xa + sig * rng.standard_normal(alive.size)
            gap = np.maximum((L - xa) * (L - xn), 0.0)
            hit = (xn >= L) | (rng.random(alive.size) < np.exp(-gap / (D * dt)))
            clock[alive] += dt
            done_at[alive[hit]] = clock[alive[hit]]
            keep = ~hit
            alive, xn = alive[keep], xn[keep]
            reset = rng.random(alive.size) < p_reset
            clock[alive[reset]] += np.abs(xn[reset]) / v
            xn[reset] = 0.0
            x[alive] = xn
        rows.append((f"L={L}, D={D}, v={v}, r={r}", home_return_mean_time(L, D, v, r), float(done_at.mean()),
                     float(done_at.std(ddof=1) / np.sqrt(n))))
    return statistical(rows)


register(Entry(
    id="K0040",
    slug="resetting-home-return",
    title="Search with resetting and a return home at constant speed: mean time to the target",
    area="resetting",
    kind="exact",
    statement=(
        "A searcher diffuses from home towards a target at distance L, gives up at rate r and walks home at speed v "
        "before searching again. The mean time is the search term of instantaneous resetting, (e^z - 1)/r, plus a "
        "return term (L/v)[2 sinh(z)/z - 1], z = L sqrt(r/D); the optimal rate falls as returns get slower."
    ),
    model=(r"\text{diffusion from } x=0 \text{ to an absorbing target at } L;\ \text{at rate } r\text{, return to } 0 "
           r"\text{ at speed } v_r \text{ (time } |x|/v_r\text{), blind to the target}"),
    formula=(r"\langle T_r\rangle=\frac{1}{r}\left(e^{\sqrt{\tau_d r}}-1\right)+\tau_b\left[\frac{2\sinh\sqrt{\tau_d r}}"
             r"{\sqrt{\tau_d r}}-1\right],\quad \tau_d=\frac{L^2}{D},\ \tau_b=\frac{L}{v_r}"),
    function=home_return_mean_time,
    parameters={"L": "distance from home to the target", "D": "diffusion coefficient (<x^2> = 2Dt)",
                "v_return": "speed of the walk home", "r": "rate of giving up and returning"},
    sources=(
        Source(
            citation="A. Pal, L. Kusmierz and S. Reuveni, Phys. Rev. Research 2, 043174 (2020)",
            doi="10.1103/PhysRevResearch.2.043174",
            locator="Eq. (9) p. 3 with Eq. (10) (mean); Eq. (6) (general form); Eq. (8) (propagator); Eq. (11) p. 4 (optimal rate) "
                    "(arXiv:1906.06987v3 numbering)",
            version_read="arXiv:1906.06987v3 PDF p. 3 as a page image and the TeX source, opened 22 Nov 2026 (two readers); journal numbering not checked",
            printed=(r"\langle T_r  \rangle =\frac{1}{r}\left( e^{\sqrt{\tau_d r}} -1 \right)+ \tau_b~\left[ \frac{2 \sinh(\sqrt{\tau_d r})}"
                     r"{\sqrt{\tau_d r}} -1 \right]"),
            notation="same symbols; the time spent at home is neglected (<W> = 0 in their Eq. 6)",
        ),
    ),
    validity=("One dimension, free diffusion from home to an absorbing target, Poissonian giving-up, a ballistic return at "
              "constant speed that cannot find the target, no waiting at home. Exact."),
    checks=(
        Check("against the general formula", "Eq. (6) by quadrature with the propagator of Eq. (8)", _check_general_formula),
        Check("instant returns, and the optimal rate", "entry K0015; numerical minimisation", _check_limits_and_optimum),
        Check("against simulation", "Euler-Maruyama with a bridge-corrected target and timed returns home, 100,000 paths",
              _check_simulation, slow=True),
    ),
    aliases=("home returns", "space-time coupled returns", "resetting with a return time", "returning searcher"),
    related=("K0015", "K0016", "K0038"),
    added="2026-11-22",
))
