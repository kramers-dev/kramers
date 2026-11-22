"""K0037. Ornstein-Uhlenbeck process with resetting, target at the centre of the trap: mean time and the vanishing of the optimal rate (Ahmad et al.; Dubey and Pal)."""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize, special

from .._checking import combine, relative, statistical
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from ..solvers import simulate
from ..solvers.quadrature import FirstPassage1D
from .k0015_resetting_mean_time import resetting_mean_time


def _log_J(nu: float, z: float) -> float:
    """log of J(nu, z) = int_0^inf t^nu (z + t) exp(-z t - t^2/2) dt, by quadrature around its peak."""
    def phi(t: float) -> float:
        return nu * np.log(t) + np.log(z + t) - z * t - 0.5 * t * t if t > 0 else -np.inf

    # the exponent has a single maximum: phi'(t) = nu/t + 1/(z+t) - z - t
    dphi = lambda t: nu / t + 1.0 / (z + t) - z - t  # noqa: E731
    lo, hi = 1e-300, max(1.0, np.sqrt(nu) + 1.0)
    while dphi(hi) > 0:
        hi *= 2.0
    tstar = optimize.brentq(dphi, lo, hi, xtol=1e-14) if dphi(1e-300) > 0 else 0.0
    pm = phi(tstar) if tstar > 0 else np.log(z) if z > 0 else 0.0
    # phi'' = -nu/t^2 - 1/(z+t)^2 - 1 <= -1, so phi(t) <= phi(t*) - (t - t*)^2/2: beyond 14 of t* the
    # integrand is below e^-98 of its peak, and the window [t* - 14, t* + 14] holds the whole integral
    lo, hi = max(0.0, tstar - 14.0), tstar + 14.0
    edges = [lo, tstar, hi] if tstar > lo else [lo, hi]
    if tstar > 0:
        # phi(t) - phi(t*) written in the offset d = t - t*, so large nu does not cancel digits
        def rel(t: float) -> float:
            d = t - tstar
            if t <= 0:
                return 0.0
            return float(np.exp(nu * np.log1p(d / tstar) + np.log1p(d / (z + tstar)) - d * (z + tstar + 0.5 * d)))
    else:
        def rel(t: float) -> float:
            return float(np.exp(phi(t) - pm)) if t > 0 else 0.0
    total = sum(integrate.quad(rel, a, b, epsabs=0.0, epsrel=1e-12, limit=400)[0]
                for a, b in zip(edges[:-1], edges[1:], strict=True) if b > a)
    return float(np.log(total) + pm)


def ou_resetting_mean_time(x0: float, kappa: float, D: float, r: float) -> float:
    """Mean time to reach the centre x = 0 of the trap from x0, for dx = -kappa x dt + sqrt(2D) dW,
    resetting to x0 at rate r.

    Dubey and Pal's Eq. (41), with D restored (z = x0 sqrt(kappa/D), nu = r/kappa):
        <T_r> = sqrt(pi) 2^(-nu/2) e^(-z^2/4) / (r Gamma((nu+1)/2) D_(-nu)(z)) - 1/r,
    D_(-nu) the parabolic cylinder function. Evaluated through the integral representation of D_(-nu),
    integrated by parts so that nothing diverges as nu -> 0: 1/T~(r) = sqrt(pi) 2^(-nu/2)
    Gamma(nu+1) / (Gamma((nu+1)/2) J(nu, z)). Ahmad et al. write the same mean with the potential
    k x^2 (so kappa = 2k) in Kummer functions (Eqs. 21-22). Relative accuracy falls as about
    1e-16 kappa/r when r is tiny compared with kappa.
    """
    if not (x0 > 0 and kappa > 0 and D > 0 and r > 0):
        raise OutsideDomain("need x0 > 0, kappa > 0, D > 0 and r > 0")
    nu = r / kappa
    z = x0 * np.sqrt(kappa / D)
    log_inv_T = (0.5 * np.log(np.pi) - 0.5 * nu * np.log(2.0) + special.gammaln(nu + 1.0)
                 - special.gammaln(0.5 * (nu + 1.0)) - _log_J(nu, z))
    return float(np.expm1(log_inv_T) / r)


def _ahmad_form(x0: float, k: float, D: float, r: float) -> float:
    # Eqs. (21)-(22) as printed, potential k x^2 (drift -2 k x), with alpha^2 = (r + s)/D at s = 0
    gam = k / D
    a = (r / D) / (4.0 * gam)
    u = k * x0 * x0 / D
    G = special.hyp1f1(a, 0.5, u) - 2.0 * np.exp(special.gammaln(a + 0.5) - special.gammaln(a)) * x0 * np.sqrt(gam) \
        * special.hyp1f1(a + 0.5, 1.5, u)
    return float((1.0 / G - 1.0) / r)


def _dubey_pal_form(x0: float, kappa: float, r: float) -> float:
    # Eq. (41) as printed (D = 1), with scipy's parabolic cylinder function
    nu = r / kappa
    Dv = special.pbdv(-nu, np.sqrt(kappa) * x0)[0]
    return float(np.sqrt(np.pi) * 2 ** (-nu / 2) * np.exp(-kappa * x0 * x0 / 4) / (r * special.gamma((nu + 1) / 2) * Dv) - 1 / r)


def _check_printed_forms():
    rows = []
    for x0, k, r in ((2.5, 0.1, 0.3), (1.0, 0.5, 1.0), (0.5, 1.0, 4.0)):
        mine = ou_resetting_mean_time(x0, 2 * k, 1.0, r)
        rows.append((f"Ahmad Eqs. (21)-(22), x0={x0}, k={k}, r={r}", mine, _ahmad_form(x0, k, 1.0, r)))
        rows.append((f"Dubey-Pal Eq. (41), x0={x0}, kappa={2 * k}, r={r}", mine, _dubey_pal_form(x0, 2 * k, r)))
    # D != 1: Ahmad's form with alpha^2 = (r + s)/D against this entry
    rows.append(("Ahmad, x0=0.5, k=0.3, D=0.5, r=2", ou_resetting_mean_time(0.5, 0.6, 0.5, 2.0), _ahmad_form(0.5, 0.3, 0.5, 2.0)))
    return relative(rows, 1e-10, ref="the printed closed forms")


def _check_limits():
    # r -> 0: the mean time to the centre without resetting, from the quadrature solver
    rows = []
    for x0, kappa, D in ((1.0, 1.0, 1.0), (2.5, 0.35, 1.0), (0.5, 0.6, 0.5)):
        fp = FirstPassage1D(lambda x, kk=kappa, DD=D: kk * x * x / (2 * DD), D=D, domain=(0.0, np.inf),
                            left="absorbing", right="natural")
        rows.append((f"r -> 0, x0={x0}, kappa={kappa}, D={D}", ou_resetting_mean_time(x0, kappa, D, 1e-6 * kappa),
                     float(fp.mean_time(x0).value)))
    # at r = 1e-6 kappa the mean differs from its r = 0 value by about r <T>, a few times 1e-6
    lim = relative(rows, 1e-5, ref="quadrature solver without resetting")
    # kappa -> 0: free diffusion with resetting (K0015)
    free = relative([(f"kappa = 1e-6, x0={x0}, D={D}, r={r}", ou_resetting_mean_time(x0, 1e-6, D, r), resetting_mean_time(x0, r, D))
                     for x0, D, r in ((1.0, 1.0, 1.0), (0.4, 0.3, 2.0))], 1e-5, ref="entry K0015")
    return combine(lim, free)


def _check_transition():
    # resetting stops helping where the spread of the time without it equals its mean; with the quadrature
    # solver alone, z = x0 sqrt(kappa/D) at that point must match both papers' printed values:
    # Ahmad et al. K_c = 0.7393 (K = x0 sqrt(k/D), kappa = 2k), Dubey and Pal k* = 0.174895 at x_R = 2.5, D = 1
    def excess(z: float) -> float:
        fp = FirstPassage1D(lambda x: x * x / 2, D=1.0, domain=(0.0, np.inf), left="absorbing", right="natural")
        return float(np.sqrt(fp.variance(z).value) / fp.mean_time(z).value - 1.0)

    zc = optimize.brentq(excess, 0.5, 2.0, xtol=1e-12)
    # the solver gives k* = 0.174898 at x_R = 2.5 against the printed 0.174895: agreement to 2e-5 in k*, 9e-6 in z
    part = relative([("z_c against Dubey and Pal (printed to 6 digits)", zc, 2.5 * np.sqrt(0.174895))], 2e-5, ref="printed k*")
    part2 = relative([("z_c against Ahmad et al. (4 digits)", zc / np.sqrt(2), 0.7393)], 1.5e-4, ref="printed K_c")
    # and the mean under resetting: falls with small r below z_c, rises above
    signs = []
    for z in (zc * 0.95, zc * 1.05):
        d = ou_resetting_mean_time(z, 1.0, 1.0, 1e-2) - ou_resetting_mean_time(z, 1.0, 1.0, 1e-3)
        signs.append((f"z = {z:.3f}: sign of the change of the mean with small r", float(np.sign(d)), -1.0 if z < zc else 1.0))
    return combine(part, part2, relative(signs, 1e-15, ref="expected"))


def _check_simulation():
    rows = []
    for x0, kappa, D, r, seed in ((0.5, 0.6, 0.5, 2.0, 371), (1.0, 1.0, 1.0, 1.0, 372)):
        sim = simulate.first_passage_times(lambda x, kk=kappa: -kk * x, D, x0, n_paths=100_000, dt=1e-4, t_max=60.0,
                                           absorbing=(0.0, None), reset_rate=r, seed=seed)
        m = sim.mean()
        rows.append((f"x0={x0}, kappa={kappa}, D={D}, r={r}", ou_resetting_mean_time(x0, kappa, D, r), m.value, m.error))
    return statistical(rows)


register(Entry(
    id="K0037",
    slug="ou-resetting-centre",
    title="Ornstein-Uhlenbeck process with resetting, target at the trap centre: mean time and the vanishing optimal rate",
    area="resetting",
    kind="exact",
    statement=(
        "A particle in a harmonic trap, dx = -kappa x dt + sqrt(2D) dW, reset to x0 at rate r, reaches the centre of the "
        "trap after a mean time given in closed form by a parabolic cylinder function. Resetting shortens the mean only "
        "when the start is close enough to the centre: x0 sqrt(kappa/D) < 1.0455, where the spread of the time without "
        "resetting equals its mean."
    ),
    model=r"dx=-\kappa x\,dt+\sqrt{2D}\,dW,\ \text{absorbing at } x=0,\ \text{reset to } x_0 \text{ at rate } r",
    formula=(r"\langle T_r\rangle=\frac{\sqrt{\pi}\,2^{-\nu/2}e^{-z^2/4}}{r\,\Gamma\!\left(\frac{\nu+1}{2}\right)D_{-\nu}(z)}-\frac1r,"
             r"\quad \nu=\frac{r}{\kappa},\ z=x_0\sqrt{\kappa/D}"),
    function=ou_resetting_mean_time,
    parameters={"x0": "start and reset position (> 0); the target is the trap centre", "kappa": "trap stiffness (drift -kappa x)",
                "D": "diffusion coefficient", "r": "resetting rate"},
    sources=(
        Source(
            citation="A. Dubey and A. Pal, J. Phys. A: Math. Theor. 56, 435002 (2023)",
            doi="10.1088/1751-8121/acf748",
            locator="Eq. (41) p. 12; critical stiffness k* = 0.174895 at x_R = 2.5 (Fig. 4 caption) (arXiv:2304.05226v1 numbering)",
            version_read="arXiv:2304.05226v1 TeX source and PDF text layer, opened 22 Nov 2026; journal numbering not checked",
            printed=(r"\langle t_f(r) \rangle = \frac{\sqrt{\pi}2^{-\frac{r}{2k}}e^{-kx_R^2/4}}{r\Gamma(\frac{r+k}{2k})"
                     r"D_{-\frac{r}{k}}(\sqrt{k}x_R)}-\frac{1}{r}"),
            notation=("their k is kappa and D = 1; for other D replace x_R by x0/sqrt(D) (a rescaling, not printed). The quadrature "
                      "solver puts the critical stiffness at k* = 0.174898 for x_R = 2.5, against the printed 0.174895"),
        ),
        Source(
            citation="S. Ahmad, I. Nayak, A. Bansal, A. Nandi and D. Das, Phys. Rev. E 99, 022130 (2019)",
            doi="10.1103/PhysRevE.99.022130",
            locator="Eqs. (21)-(22) p. 4 (mean, Kummer functions); K_c = 0.7393 (p. 4, Fig. 2 caption); Eq. (6) (transition)",
            version_read="arXiv:1902.02955v1 TeX source and PDF p. 4, opened 22 Nov 2026 (two readers); journal numbering not checked",
            printed=(r"\langle T_r \rangle= \frac{1}{r}\bigg[\frac{1}{G(x_{0},s=0)}-1\bigg],\ G =F_{1}\Big(\frac{\alpha^2}{4\gamma};"
                     r"\frac{1}{2};\frac{kx^{2}}{D}\Big)-2\frac{\Gamma(\frac{\alpha^2}{4\gamma}+\frac{1}{2})}{\Gamma(\frac{\alpha^2}"
                     r"{4\gamma})} x \sqrt{\gamma} F_{1}\Big(\frac{\alpha^2}{4\gamma}+\frac{1}{2};\frac{3}{2};\frac{kx^{2}}{D}\Big)"),
            notation=("their potential is k x^2, so kappa = 2k and K = x0 sqrt(k/D) = z/sqrt(2); gamma = k/D. The text after "
                      "their Eq. (2) prints alpha = sqrt(r + s/D); the equation follows with alpha^2 = (r + s)/D, which is what "
                      "is used here (the two agree at D = 1; checked against this entry at D = 0.5)"),
            role="restatement",
        ),
    ),
    validity=("One dimension, harmonic trap with the target at its centre, start and reset point on one side, Poissonian "
              "instantaneous resetting. Exact. A target away from the centre is a different problem."),
    checks=(
        Check("two printed forms", "Ahmad et al. Eqs. (21)-(22) and Dubey and Pal Eq. (41), evaluated as printed", _check_printed_forms),
        Check("the limits r -> 0 and kappa -> 0", "quadrature solver without resetting; entry K0015", _check_limits),
        Check("the transition where resetting stops helping", "quadrature solver: spread = mean; both papers' printed values",
              _check_transition),
        Check("against simulation", "Euler-Maruyama with resetting, 100,000 paths", _check_simulation, slow=True),
    ),
    aliases=("Ornstein-Uhlenbeck with resetting", "harmonic trap with resetting", "resetting transition in a potential",
             "optical trap with stochastic resetting"),
    related=("K0010", "K0015", "K0017", "K0036"),
    added="2026-11-22",
))
