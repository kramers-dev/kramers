"""K0041. Brownian motion in a planar wedge of any angle: exact survival probability (Chupeau, Benichou and Majumdar)."""

from __future__ import annotations

import numpy as np
from scipy import special

from .._checking import relative, statistical
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain


def _integral_term(y: float, phi0: float, alpha: float) -> float:
    # (2/pi)^(3/2) sqrt(y) e^-y / 2 int_0^inf e^{-y cosh u} sinh(u/2) [arctan(..) + arctan(..)] du,
    # with the two exponentials combined, e^{-y (1 + cosh u)}, and the range cut where that is below e^-745
    a1 = np.sin(np.pi / alpha * (phi0 + np.pi / 2))
    a2 = np.sin(np.pi / alpha * (phi0 - np.pi / 2))
    if abs(a1 + a2) <= 1e-14 * max(abs(a1), abs(a2), 1.0):
        return 0.0  # arctan is odd: the two terms cancel for every u (e.g. a right angle, a half-plane)

    def f(u: np.ndarray) -> np.ndarray:
        s = np.sinh(np.pi * u / (2 * alpha))
        return np.exp(-y * (1 + np.cosh(u))) * np.sinh(u / 2) * (np.arctan2(a1, s) + np.arctan2(a2, s))

    # the integrand is analytic on [0, u_max] (arctan(a/s) = sign(a) pi/2 - arctan(s/a) near s = 0), so a fixed
    # composite Gauss-Legendre rule converges fast; panels are at most 0.1 long, finer near the arctan scales
    u_max = float(np.arccosh(max(745.0 / y - 1.0, 1.0))) + 1.0
    scales = [2 * alpha / np.pi * np.arcsinh(abs(a)) * fac for a in (a1, a2) if abs(a) > 0 for fac in (0.1, 1.0)]
    knots = sorted({0.0, u_max, *[k for k in scales if 0 < k < u_max]})
    edges = [knots[0]]
    for a, b in zip(knots[:-1], knots[1:], strict=True):
        n = max(int(np.ceil((b - a) / 0.1)), 8 if a == 0.0 else 1)
        edges.extend(np.linspace(a, b, n + 1)[1:])
    xg, wg = np.polynomial.legendre.leggauss(24)
    e = np.asarray(edges)
    lo, hi = e[:-1, None], e[1:, None]
    u = 0.5 * (hi - lo) * xg[None, :] + 0.5 * (hi + lo)
    val = float(np.sum(0.5 * (hi - lo) * wg[None, :] * f(u)))
    return float((2 / np.pi) ** 1.5 * np.sqrt(y) / 2 * val)


def wedge_survival(t: float, r0: float, phi0: float, alpha: float, D: float = 1.0) -> float:
    """Probability that a Brownian particle started at polar position (r0, phi0) inside the wedge 0 < phi < alpha
    has not touched either edge by time t (both edges absorbing; any 0 < alpha < 2 pi).

    Eq. (27): with y = r0^2/(8 D t) and k = floor(pi/(2 alpha) - 1/2),
    S = erf(sqrt(2y) sin(min(phi0, pi/2))) + sum_{j=1}^{k+1} (-1)^j [erf(sqrt(2y) sin(min(j alpha + phi0, pi/2)))
        - erf(sqrt(2y) sin(min(j alpha - phi0, pi/2)))] + an integral term. <x^2> = 2 D t per coordinate.
    """
    if not (0 < alpha < 2 * np.pi and 0 < phi0 < alpha):
        raise OutsideDomain("need 0 < alpha < 2 pi and 0 < phi0 < alpha")
    if not (t > 0 and r0 > 0 and D > 0):
        raise OutsideDomain("need t > 0, r0 > 0 and D > 0")
    y = r0 * r0 / (8.0 * D * t)
    q = np.sqrt(2.0 * y)
    k = int(np.floor(np.pi / (2 * alpha) - 0.5))
    S = special.erf(q * np.sin(min(phi0, np.pi / 2)))
    for j in range(1, k + 2):
        S += (-1) ** j * (special.erf(q * np.sin(min(j * alpha + phi0, np.pi / 2)))
                          - special.erf(q * np.sin(min(j * alpha - phi0, np.pi / 2))))
    return float(S + _integral_term(y, phi0, alpha))


def _series(y: float, phi0: float, alpha: float) -> float:
    # Eq. (5): 2 sqrt(2y/pi) e^-y sum_m sin((2m+1) phi0 pi/alpha)/(2m+1) [I_nu(y) + I_{nu+1}(y)], nu = (2m+1) pi/(2 alpha) - 1/2
    total = 0.0
    for m in range(20000):
        nu = (2 * m + 1) * np.pi / (2 * alpha) - 0.5
        term = np.sin((2 * m + 1) * phi0 * np.pi / alpha) / (2 * m + 1) * (special.ive(nu, y) + special.ive(nu + 1, y))
        total += term
        if nu > y + 40 and abs(term) < 1e-18:
            break
    return float(2.0 * np.sqrt(2.0 * y / np.pi) * total)


def _check_two_forms():
    rows = []
    for alpha in (np.pi / 3, 0.9, 2.0, 3 * np.pi / 2):
        for frac in (0.3, 0.5):
            for y in (0.2, 1.0, 5.0):
                phi0 = frac * alpha
                rows.append((f"alpha={alpha:.4f}, phi0={phi0:.4f}, y={y}", wedge_survival(1.0, np.sqrt(8 * y), phi0, alpha),
                             _series(y, phi0, alpha)))
    return relative(rows, 1e-10, ref="the Bessel series, Eq. (5)")


def _check_special_angles():
    rows = []
    for phi0, y in ((0.4712, 1.0), (1.1, 0.3), (0.2, 4.0)):
        q = np.sqrt(2 * y)
        rows.append((f"right angle, phi0={phi0}, y={y}", wedge_survival(1.0, np.sqrt(8 * y), phi0, np.pi / 2),
                     float(special.erf(q * np.sin(phi0)) * special.erf(q * np.cos(phi0)))))
    for phi0, y in ((0.4, 1.0), (2.0, 0.3), (1.5707, 4.0)):
        rows.append((f"half-plane, phi0={phi0}, y={y}", wedge_survival(1.0, np.sqrt(8 * y), phi0, np.pi),
                     float(special.erf(np.sqrt(2 * y) * np.sin(phi0)))))
    return relative(rows, 1e-11, ref="product of half-line survivals; distance to a line")


def _check_simulation():
    # two-dimensional Euler-Maruyama in convex wedges (alpha < pi), each edge a line through the apex with a
    # Brownian-bridge crossing correction exp(-d_before d_after / (D dt)); r0 = D = 1
    rows = []
    for alpha, phi0, seed in ((0.9, 0.35, 411), (2.0, 0.6, 412)):
        rng = np.random.default_rng(seed)
        n, dt, D = 100_000, 2.5e-5, 1.0
        pos = np.tile([np.cos(phi0), np.sin(phi0)], (n, 1))
        dead_at = np.full(n, np.inf)
        alive = np.arange(n)
        normal2 = np.array([np.sin(alpha), -np.cos(alpha)])  # distance to the edge phi = alpha is pos . normal2
        steps = int(round(1.0 / dt))
        for i in range(1, steps + 1):
            p = pos[alive]
            pn = p + np.sqrt(2 * D * dt) * rng.standard_normal(p.shape)
            d0a, d0b = p[:, 1], pn[:, 1]
            d1a, d1b = p @ normal2, pn @ normal2
            hit = (d0b <= 0) | (d1b <= 0)
            hit |= rng.random(alive.size) < np.exp(-np.maximum(d0a * d0b, 0) / (D * dt))
            hit |= rng.random(alive.size) < np.exp(-np.maximum(d1a * d1b, 0) / (D * dt))
            dead_at[alive[hit]] = i * dt
            pos[alive] = pn
            alive = alive[~hit]
        for t in (0.25, 1.0):
            s = float(np.mean(dead_at > t))
            rows.append((f"alpha={alpha}, phi0={phi0}, t={t}", wedge_survival(t, 1.0, phi0, alpha, D), s, np.sqrt(s * (1 - s) / n)))
    return statistical(rows)


register(Entry(
    id="K0041",
    slug="wedge-survival",
    title="Brownian motion in a planar wedge of any angle: exact survival probability",
    area="targets",
    kind="exact",
    statement=(
        "A Brownian particle in a wedge of opening angle alpha with absorbing edges survives up to time t with a probability "
        "that depends only on y = r0^2/(8Dt) and the starting angle. For any alpha it is a finite sum of error functions plus "
        "one integral; at long times it decays as t^(-pi/(2 alpha))."
    ),
    model=r"\partial_tP=D\Delta P,\ P|_{\varphi=0}=P|_{\varphi=\alpha}=0,\ \text{start at }(r_0,\varphi_0)",
    formula=(r"S=\mathrm{erf}(\sqrt{2y}\sin\min(\varphi_0,\tfrac\pi2))+\sum_{j=1}^{k+1}(-1)^j[\mathrm{erf}(\sqrt{2y}\sin\min(j\alpha+\varphi_0,\tfrac\pi2))"
             r"-\mathrm{erf}(\sqrt{2y}\sin\min(j\alpha-\varphi_0,\tfrac\pi2))]+I(y,\varphi_0),\ y=\frac{r_0^2}{8Dt}"),
    function=wedge_survival,
    parameters={"t": "time", "r0": "start distance from the apex", "phi0": "start angle, 0 < phi0 < alpha",
                "alpha": "opening angle of the wedge, 0 < alpha < 2 pi", "D": "diffusion coefficient (<x^2> = 2Dt per coordinate)"},
    sources=(
        Source(
            citation="M. Chupeau, O. Benichou and S. N. Majumdar, Phys. Rev. E 91, 032106 (2015)",
            doi="10.1103/PhysRevE.91.032106",
            locator="Eq. (27) p. 4 (any angle; condenses Eqs. 25 and 26); Eq. (5) p. 2 (Bessel series); Eq. (31) (right angle) "
                    "(arXiv:1501.02274v1 numbering)",
            version_read="arXiv:1501.02274v1 PDF p. 4 as a page image and the TeX source, opened 22 Nov 2026 (two readers); "
                         "journal numbering not checked",
            printed=(r"S(y,\varphi_0) = \mathrm{erf}\left(\sqrt{2y}\sin\left(\min\left(\varphi_0,\frac{\pi}{2}\right) \right)\right) "
                     r"+\sum\limits_{j=1}^{k+1} (-1)^j \left[\ldots\right] +\left(\frac{2}{\pi}\right)^{3/2} \sqrt{y} \; \frac{\mathrm{e}^{-y}}{2} "
                     r"\int_0^{+\infty} \mathrm{d}u \; \mathrm{e}^{-y\cosh u} \sinh\frac{u}{2} \times \left[\arctan(\ldots)+\arctan(\ldots)\right]"),
            notation="same symbols; k = floor(pi/(2 alpha) - 1/2), which is -1 (no sum) for alpha > pi",
        ),
    ),
    validity="Free two-dimensional diffusion, both edges of the wedge absorbing, any opening angle below 2 pi. Exact.",
    checks=(
        Check("two printed forms", "the Bessel series Eq. (5), acute and obtuse wedges", _check_two_forms),
        Check("right angle and half-plane", "product of two half-line survivals; erf of the distance to a line", _check_special_angles),
        Check("against simulation", "two-dimensional Euler-Maruyama with a bridge correction on each edge, 100,000 paths",
              _check_simulation, slow=True),
    ),
    aliases=("wedge survival", "Brownian motion in a cone (planar)", "corner first passage", "survival between two walls meeting at an angle"),
    related=("K0014", "K0020"),
    added="2026-11-22",
))
