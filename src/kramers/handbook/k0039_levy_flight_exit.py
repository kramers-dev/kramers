"""K0039. Symmetric Levy flight: mean exit time from an interval (Getoor's formula, as printed by Dybiec et al.)."""

from __future__ import annotations

import numpy as np
from scipy import integrate, special

from .._checking import combine, relative, statistical
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from .k0012_drift_diffusion_interval import drift_diffusion_interval


def levy_flight_exit_time(x0: float, L: float, alpha: float, sigma0: float = 1.0) -> float:
    """Mean time for a symmetric alpha-stable Levy flight started at x0 to leave the interval (-L, L).

    <tau(x0)> = (L^2 - x0^2)^(alpha/2) / (Gamma(1 + alpha) sigma0^alpha)   (0 < alpha <= 2).
    The increments over a time dt have the characteristic function exp(-sigma0^alpha |k|^alpha dt), so
    at alpha = 2 the motion is Brownian with D = sigma0^2 (<x^2> = 2 D t). Leaving means landing
    anywhere outside the interval: a jump over an end counts.
    """
    if not (0 < alpha <= 2):
        raise OutsideDomain("need 0 < alpha <= 2")
    if not (L > 0 and sigma0 > 0 and abs(x0) < L):
        raise OutsideDomain("need L > 0, sigma0 > 0 and |x0| < L")
    return float(np.exp(0.5 * alpha * np.log((L - x0) * (L + x0)) - special.gammaln(1.0 + alpha) - alpha * np.log(sigma0)))


def levy_flight_exit_time_uniform(L: float, alpha: float, sigma0: float = 1.0) -> float:
    """The same averaged over a start uniform on (-L, L): pi L^alpha / (2^(1+alpha) Gamma((1+alpha)/2) Gamma((3+alpha)/2) sigma0^alpha)."""
    if not (0 < alpha <= 2 and L > 0 and sigma0 > 0):
        raise OutsideDomain("need 0 < alpha <= 2, L > 0, sigma0 > 0")
    return float(np.pi * (L / sigma0) ** alpha / (2 ** (1 + alpha) * special.gamma(0.5 * (1 + alpha)) * special.gamma(0.5 * (3 + alpha))))


def _check_brownian_and_average():
    # alpha = 2 is Brownian motion with D = sigma0^2 in an interval of width 2L (entry K0012, no drift)
    rows = []
    for x0, L, s0 in ((0.3, 1.0, 0.7), (-0.9, 2.0, 1.3), (0.0, 0.5, 0.2)):
        rows.append((f"alpha = 2, x0={x0}, L={L}, sigma0={s0}", levy_flight_exit_time(x0, L, 2.0, s0),
                     drift_diffusion_interval(x0 + L, 2 * L, 0.0, s0 * s0).mean_time))
    brown = relative(rows, 1e-12, ref="entry K0012")
    avg = []
    for a in (0.5, 1.0, 1.3, 2.0):
        q = integrate.quad(lambda x, a=a: levy_flight_exit_time(x, 1.0, a), -1, 1, epsabs=0, epsrel=1e-12, limit=200)[0] / 2
        avg.append((f"uniform start, alpha = {a}", levy_flight_exit_time_uniform(1.0, a), q))
    return combine(brown, relative(avg, 1e-10, ref="quadrature average of Eq. (8)"))


def _stable(rng: np.random.Generator, alpha: float, n: int) -> np.ndarray:
    # Chambers-Mallows-Stuck sampler for the symmetric stable law with E exp(ikX) = exp(-|k|^alpha)
    V = rng.uniform(-np.pi / 2, np.pi / 2, n)
    if alpha == 1.0:
        return np.tan(V)
    W = rng.exponential(1.0, n)
    return np.sin(alpha * V) / np.cos(V) ** (1 / alpha) * (np.cos(V - alpha * V) / W) ** ((1 - alpha) / alpha)


def _exit_times(alpha: float, x0: float, L: float, dt: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, float(x0))
    t = np.zeros(n)
    alive = np.arange(n)
    step = dt ** (1 / alpha)
    k = 0
    while alive.size:
        k += 1
        x[alive] += step * _stable(rng, alpha, alive.size)
        out = np.abs(x[alive]) >= L
        t[alive[out]] = k * dt
        alive = alive[~out]
    return t


def _check_simulation():
    # Euler steps of the alpha-stable motion; leaving is tested at the end of each step, so the simulated time is
    # slightly too long (a path can leave and come back within a step). The step is small enough here that the
    # bias is below the statistical error; it is shown falling with the step at alpha = 1.5.
    rows = []
    for alpha, x0, dt, seed in ((0.7, 0.0, 2.5e-4, 391), (1.0, 0.5, 2.5e-4, 392), (1.5, 0.0, 6.25e-5, 393)):
        t = _exit_times(alpha, x0, 1.0, dt, 100_000, seed)
        rows.append((f"alpha={alpha}, x0={x0}, dt={dt:g}", levy_flight_exit_time(x0, 1.0, alpha), float(t.mean()),
                     float(t.std(ddof=1) / np.sqrt(t.size))))
    stat = statistical(rows)
    exact = levy_flight_exit_time(0.0, 1.0, 1.5)
    coarse = float(_exit_times(1.5, 0.0, 1.0, 1e-3, 100_000, 394).mean()) - exact
    fine = rows[2][2] - exact
    falling = relative([("alpha = 1.5: the step bias falls from dt = 1e-3 to 6.25e-5", float(abs(fine) < abs(coarse)), 1.0)],
                       1e-15, ref="expected")
    return combine(stat, falling)


register(Entry(
    id="K0039",
    slug="levy-flight-exit",
    title="Symmetric Levy flight: mean exit time from an interval",
    area="active",
    kind="exact",
    statement=(
        "A symmetric alpha-stable Levy flight started at x0 leaves the interval (-L, L) after a mean time "
        "(L^2 - x0^2)^(alpha/2) / (Gamma(1 + alpha) sigma0^alpha). At alpha = 2 this is the Brownian result "
        "(L^2 - x0^2)/(2D); for alpha < 2 the flight can leave by jumping over an end."
    ),
    model=r"\dot x=\zeta_\alpha(t),\ \langle e^{ik\,\Delta x}\rangle=e^{-\sigma_0^\alpha|k|^\alpha\Delta t},\ \tau=\min\{t:|x(t)|\ge L\}",
    formula=r"\langle\tau(x_0)\rangle=\frac{(L^2-x_0^2)^{\alpha/2}}{\Gamma(1+\alpha)\,\sigma_0^{\alpha}}",
    function=levy_flight_exit_time,
    parameters={"x0": "start, -L < x0 < L", "L": "half-width of the interval", "alpha": "stability index, 0 < alpha <= 2",
                "sigma0": "scale: increments have characteristic function exp(-sigma0^alpha |k|^alpha dt)"},
    sources=(
        Source(
            citation="B. Dybiec, E. Gudowska-Nowak, E. Barkai and A. A. Dubkov, Phys. Rev. E 95, 052102 (2017)",
            doi="10.1103/PhysRevE.95.052102",
            locator="Eq. (8) p. 3 (mean exit time), Eq. (9) (uniform start), Eqs. (1)-(3), (6), (7) (model and definitions) "
                    "(arXiv:1703.10199v1 numbering)",
            version_read="arXiv:1703.10199v1 PDF p. 3 as a page image and the TeX source, opened 22 Nov 2026 (two readers); journal numbering not checked",
            printed=r"\langle \tau (x_0) \rangle =\frac{1}{\Gamma(1+\alpha)} \frac{(L^2-|x_0|^2)^{\alpha/2}}{\sigma_0^\alpha}",
            notation="same symbols; K_alpha = sigma0^alpha in their fractional equation; at alpha = 2, D = sigma0^2",
        ),
        Source(
            citation="R. K. Getoor, Trans. Amer. Math. Soc. 101, 75 (1961)",
            doi="10.1090/s0002-9947-1961-0137148-5",
            locator="the original result, cited as ref. [16] by Dybiec et al.",
            version_read="not opened; citation only",
            role="original",
            evidence="citation-only",
        ),
    ),
    validity=("Symmetric alpha-stable Levy flight (no finite speed), 0 < alpha <= 2, the whole exterior of the interval "
              "absorbing. Exact. Not a Levy walk, and not the local condition p(-L) = p(L) = 0 of older work."),
    checks=(
        Check("the Brownian case and the uniform-start average", "entry K0012 at alpha = 2; quadrature of Eq. (8)",
              _check_brownian_and_average),
        Check("against simulation", "alpha-stable steps by the Chambers-Mallows-Stuck method, 100,000 paths; the step bias "
              "shown falling", _check_simulation, slow=True),
    ),
    aliases=("Levy flight in an interval", "Getoor formula", "stable process exit time", "fractional Laplacian mean exit time"),
    related=("K0012", "K0020"),
    added="2026-11-22",
))
