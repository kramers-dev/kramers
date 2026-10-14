"""K0012. Drift-diffusion on an interval with absorbing ends: exit probability and mean times."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .._checking import relative
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from ..solvers.quadrature import FirstPassage1D


class IntervalExit(NamedTuple):
    p_right: float  # probability of leaving through L
    mean_time: float  # unconditional mean exit time
    mean_time_right: float  # mean exit time given exit through L
    mean_time_left: float  # mean exit time given exit through 0


def _zcoth_minus_one(z: np.ndarray) -> np.ndarray:
    """g(z) = z coth z - 1, without cancellation near 0."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    small = np.abs(z) < 0.05
    zs = z[small] ** 2
    out[small] = zs / 3.0 - zs**2 / 45.0 + 2.0 * zs**3 / 945.0 - zs**4 / 4725.0
    zb = z[~small]
    out[~small] = zb / np.tanh(zb) - 1.0
    return out


def drift_diffusion_interval(x: float, L: float, v: float, D: float) -> IntervalExit:
    """Exit statistics of dX = v dt + sqrt(2D) dW from x in (0, L), both ends absorbing.

    p_right is Redner's Eq. (4.12). The conditional means are Srivastava et al.'s
    Eqs. (11)-(12) moved to [0, L] (thresholds +-zeta = +-L/2 around the midpoint):
    T_L = [L coth(vL/2D) - x coth(vx/2D)] / v and T_0 the same with x -> L - x.
    Evaluated through g(z) = z coth z - 1 so that small drifts lose no digits.
    """
    if not (0.0 <= x <= L and L > 0 and D > 0):
        raise OutsideDomain("need 0 <= x <= L, L > 0, D > 0")
    a = v / (2.0 * D)
    if v == 0.0:
        p = x / L
    elif v > 0:
        p = float(np.expm1(-v * x / D) / np.expm1(-v * L / D))
    else:  # the same ratio multiplied through by exp(vL/D), so nothing overflows
        p = float(np.exp(v * (L - x) / D) * np.expm1(v * x / D) / np.expm1(v * L / D))
    if abs(a * L) < 1e-12:  # no drift: Redner Eqs. (4.16), (4.19)
        t_right = (L * L - x * x) / (6.0 * D)
        t_left = (L * L - (L - x) ** 2) / (6.0 * D)
    else:
        g = _zcoth_minus_one
        t_right = float((g(np.array(a * L)) - g(np.array(a * x))) / (2.0 * D * a * a)) if x > 0 else float(g(np.array(a * L)) / (2.0 * D * a * a))
        y = L - x
        t_left = float((g(np.array(a * L)) - g(np.array(a * y))) / (2.0 * D * a * a)) if y > 0 else float(g(np.array(a * L)) / (2.0 * D * a * a))
    mean = p * t_right + (1.0 - p) * t_left
    return IntervalExit(p_right=p, mean_time=mean, mean_time_right=t_right, mean_time_left=t_left)


def _check_quadrature():
    rows = []
    for x, L, v, D in ((0.5, 2.0, 1.0, 1.0), (1.0, 2.0, 1.0, 1.0), (0.3, 1.0, -4.0, 0.5),
                       (0.7, 1.0, 1e-7, 1.0), (0.2, 3.0, 25.0, 1.0)):
        r = drift_diffusion_interval(x, L, v, D)
        fp = FirstPassage1D(drift=lambda y, v=v: v + 0.0 * y, D=D, domain=(0.0, L), left="absorbing", right="absorbing")
        rows += [
            (f"p_right, x={x}, v={v}", r.p_right, float(fp.splitting(x).value)),
            (f"mean, x={x}, v={v}", r.mean_time, float(fp.mean_time(x).value)),
            (f"mean | right, x={x}, v={v}", r.mean_time_right, float(fp.conditional_mean_time(x, "right").value)),
            (f"mean | left, x={x}, v={v}", r.mean_time_left, float(fp.conditional_mean_time(x, "left").value)),
        ]
    return relative(rows, 1e-9, ref="quadrature solver")


def _check_printed():
    # Srivastava et al. Eqs. (6), (7), (11), (12) as printed, thresholds +-zeta, start x0 from the middle
    zeta, x0, mu, sigma = 1.0, -0.5, 1.0, np.sqrt(2.0)
    s = mu / sigma**2
    P_minus = (np.exp(-2 * s * x0) - np.exp(-2 * s * zeta)) / (np.exp(2 * s * zeta) - np.exp(-2 * s * zeta))
    mDT = ((1 - 2 * P_minus) * zeta - x0) / mu
    mDT_plus = 2 * zeta / mu / np.tanh(2 * s * zeta) - (zeta + x0) / mu / np.tanh(s * (zeta + x0))
    mDT_minus = 2 * zeta / mu / np.tanh(2 * s * zeta) - (zeta - x0) / mu / np.tanh(s * (zeta - x0))
    r = drift_diffusion_interval(zeta + x0, 2 * zeta, mu, sigma**2 / 2)
    rows = [("P(lower) = 1 - p_right", 1 - r.p_right, P_minus), ("mean", r.mean_time, mDT),
            ("mean | upper", r.mean_time_right, mDT_plus), ("mean | lower", r.mean_time_left, mDT_minus)]
    return relative(rows, 1e-13, ref="Srivastava et al. (2017), Eqs. (6), (7), (11), (12)")


register(Entry(
    id="K0012",
    slug="drift-diffusion-interval",
    title="Drift-diffusion on an interval: exit probability, mean and conditional exit times",
    area="targets",
    kind="exact",
    statement=(
        "For Brownian motion with drift v and diffusion D started at x in (0, L) with both "
        "ends absorbing: the probability of leaving through L, the mean exit time, and the "
        "mean exit time conditioned on each end, in closed form, including the zero-drift "
        "limits."
    ),
    model=r"dX = v\,dt + \sqrt{2D}\,dW,\quad 0<X<L,\ \text{both ends absorbing}",
    formula=(r"\pi_L=\frac{1-e^{-vx/D}}{1-e^{-vL/D}},\ T_L=\frac{1}{v}\left[L\coth\frac{vL}{2D}-x\coth\frac{vx}{2D}\right],\ "
             r"T_0=T_L|_{x\to L-x},\ T=\pi_LT_L+(1-\pi_L)T_0"),
    function=drift_diffusion_interval,
    parameters={"x": "start", "L": "length", "v": "drift (towards L if positive)", "D": "diffusion coefficient"},
    sources=(
        Source(
            citation="S. Redner, Physica A 631, 128545 (2023)",
            doi="10.1016/j.physa.2023.128545",
            locator="Eqs. (4.12) p. 6; (4.16), (4.19) p. 7",
            version_read="journal PDF on the author's site, opened 14 Oct 2026",
            printed=(r"\mathcal{E}_{+}(x_0)=\frac{1-e^{-vx_0/D}}{1-e^{-vL/D}};\ t(x_0)=\frac{x_0(L-x_0)}{2D};\ "
                     r"t_{+}(x_0)=\frac{L^{2}-x_0^{2}}{6D}"),
            notation="E_+ = p_right; zero-drift t and t_+",
        ),
        Source(
            citation="V. Srivastava, S. F. Feng, J. D. Cohen, N. E. Leonard and A. Shenhav, J. Math. Psychol. 77, 94 (2017)",
            doi="10.1016/j.jmp.2016.10.001",
            locator="Eqs. (6), (7), (11), (12), Sec. 3.3",
            version_read="PubMed Central author manuscript PMC5473348 and arXiv:1508.03373v3, opened 14 Oct 2026",
            printed=(r"\mathrm{mDT}^{+}=\frac{2\zeta}{\mu}\coth(2s\zeta)-\frac{\zeta+x_0}{\mu}\coth(s(\zeta+x_0)),\ s=\mu/\sigma^2"),
            notation="thresholds +-zeta = 0 and L moved by L/2; mu = v; sigma^2 = 2D; their x0 = our x - L/2",
        ),
        Source(
            citation="P. L. Krapivsky and S. Redner, J. Stat. Mech. (2018) 093208",
            doi="10.1088/1742-5468/aaddb3",
            locator="Eqs. (28a), (28b), p. 11",
            version_read="NSF public-access copy of the journal PDF, page image read 14 Oct 2026",
            role="restatement",
        ),
    ),
    validity="Constant drift and diffusion, both ends absorbing, start inside. Exact.",
    checks=(
        Check("all four quantities against the quadrature solver", "Green's-function quadrature", _check_quadrature),
        Check("against the formulas as printed", "Srivastava et al. (2017)", _check_printed),
    ),
    aliases=("gambler's ruin, continuous", "drift-diffusion model of decision making", "Wiener first passage with two barriers"),
    related=("K0002", "K0003", "K0013"),
    added="2026-10-14",
))
