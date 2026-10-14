"""K0024. Distribution of the switching point under a ramp, in the quasi-static (adiabatic) approximation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .._checking import relative
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from ..solvers import _cheb


@dataclass(frozen=True)
class SwitchingDistribution:
    """The switching law on Chebyshev panels, with its moments; pdf, survival and quantiles."""

    panels: _cheb.Panels
    log_hazard_density: np.ndarray  # log(k / Fdot) at the panel nodes
    hazard: np.ndarray  # cumulative hazard H(F) at the panel nodes
    F0: float
    mean: float
    variance: float
    mode: float

    def _eval(self, F: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        F = np.atleast_1d(np.asarray(F, dtype=float))
        inside = (F >= self.panels.edges[0]) & (F <= self.panels.edges[-1])
        lh = np.full_like(F, -np.inf)
        H = np.where(F < self.panels.edges[0], 0.0, np.inf)
        if np.any(inside):
            lh[inside] = _cheb.interpolate(self.panels, self.log_hazard_density, F[inside])
            H[inside] = _cheb.interpolate(self.panels, self.hazard, F[inside])
        return F, lh, H

    def pdf(self, F: ArrayLike) -> np.ndarray:
        _, lh, H = self._eval(F)
        with np.errstate(over="ignore", invalid="ignore"):
            out = np.exp(lh - H)
        return np.where(np.isfinite(out), out, 0.0)

    def survival(self, F: ArrayLike) -> np.ndarray:
        _, _, H = self._eval(F)
        return np.exp(-H)

    def quantile(self, q: float) -> float:
        Hq = -np.log1p(-q)
        if not 0 < q < 1 or Hq > self.hazard[-1, -1]:
            raise OutsideDomain("quantile outside the computed range; widen F_max")
        x = self.panels.x.ravel()
        h = self.hazard.ravel()
        order = np.argsort(x, kind="stable")
        x0 = float(np.interp(Hq, h[order], x[order]))
        from scipy.optimize import brentq
        lo, hi = max(self.panels.edges[0], x0 - 1e-3 * (1 + abs(x0))), min(self.panels.edges[-1], x0 + 1e-3 * (1 + abs(x0)))
        f = lambda v: float(_cheb.interpolate(self.panels, self.hazard, np.array([v]))[0]) - Hq  # noqa: E731
        if f(lo) * f(hi) < 0:
            return float(brentq(f, lo, hi, xtol=1e-14))
        return x0


def switching_distribution(rate: Callable[[np.ndarray], np.ndarray], loading_rate: float | Callable[[np.ndarray], np.ndarray],
                           F_max: float, F0: float = 0.0, n: int = 25) -> SwitchingDistribution:
    """Distribution of the control value F at which the system switches while F rises from F0.

    p(F) = k(F) / Fdot(F) * exp(-integral from F0 to F of k / Fdot), the quasi-static
    survival law (Dudko, Hummer and Szabo, Eq. 2; Kurkijarvi; Garg). ``rate`` is k(F)
    (vectorised, positive); ``loading_rate`` is dF/dt, a number or a function of F.
    The integral is accumulated in logarithms on Chebyshev panels, so rates spanning
    hundreds of orders of magnitude are handled.
    """
    if not F_max > F0:
        raise OutsideDomain("need F_max > F0")
    if callable(loading_rate):
        Fdot = loading_rate
    else:
        if not loading_rate > 0:
            raise OutsideDomain("the loading rate must be positive")
        Fdot = lambda f: np.full_like(np.asarray(f, dtype=float), float(loading_rate))  # noqa: E731

    def log_h(f: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return np.log(rate(f)) - np.log(Fdot(f))

    P = _cheb.build_panels(log_h, F0, F_max, n=n)
    x = P.x
    lh = log_h(x)
    logH = _cheb.log_cumulative(P, lh)
    H = np.exp(logH)
    shifted = np.maximum(x - F0, 1e-300)
    log_p = lh - H
    w_mean = _cheb.log_integral(P, log_p + np.log(shifted))
    w_sq = _cheb.log_integral(P, log_p + 2 * np.log(shifted))
    mass = -float(np.expm1(-H[-1, -1]))
    mean = F0 + float(np.exp(w_mean)) / mass
    var = float(np.exp(w_sq)) / mass - (mean - F0) ** 2
    flat = np.argmax(log_p.ravel())
    dist = SwitchingDistribution(panels=P, log_hazard_density=lh, hazard=H, F0=F0, mean=mean, variance=var,
                                 mode=float(x.ravel()[flat]))
    # refine the mode on a fine grid around the best node
    span = P.edges[-1] - P.edges[0]
    fine = np.linspace(max(P.edges[0], dist.mode - 1e-2 * span), min(P.edges[-1], dist.mode + 1e-2 * span), 4001)
    mode = float(fine[np.argmax(dist.pdf(fine))])
    return SwitchingDistribution(panels=P, log_hazard_density=lh, hazard=H, F0=F0, mean=mean, variance=var, mode=mode)


def _dhs_rate(F, k0, x, dG, nu):
    return k0 * (1 - nu * F * x / dG) ** (1 / nu - 1) * np.exp(dG * (1 - (1 - nu * F * x / dG) ** (1 / nu)))


def _check_bell_closed_form():
    k0, x, r = 1e-3, 0.5, 20.0  # kT = 1
    dist = switching_distribution(lambda f: k0 * np.exp(f * x), r, 40.0)
    F = np.array([5.0, 10.0, 14.0, 18.0])
    H = k0 / (x * r) * np.expm1(F * x)
    exact = k0 * np.exp(F * x) / r * np.exp(-H)
    rows = [(f"p(F={f:g})", float(dist.pdf(np.array([f]))[0]), float(e)) for f, e in zip(F, exact, strict=True)]
    return relative(rows, 1e-6, ref="closed form for the Bell rate")


def _check_dhs_density():
    # Dudko, Hummer and Szabo Eq. (4), printed closed form for the same model (kT = 1)
    rows = []
    for nu in (2.0 / 3.0, 0.5):
        k0, x, dG, KV = 1e-4, 0.4, 20.0, 1e3
        Fc = dG / (nu * x)
        dist = switching_distribution(lambda f, nu=nu, k0=k0, x=x, dG=dG: _dhs_rate(f, k0, x, dG, nu), KV, 0.999 * Fc)
        for frac in (0.5, 0.7):
            F = frac * Fc
            kF = _dhs_rate(F, k0, x, dG, nu)
            printed = kF / KV * np.exp(k0 / (x * KV)) * np.exp(-kF / (x * KV) * (1 - nu * F * x / dG) ** (1 - 1 / nu))
            rows.append((f"nu={nu:.3f}, F={frac} Fc", float(dist.pdf(np.array([F]))[0]), float(printed)))
    return relative(rows, 1e-6, ref="Dudko, Hummer and Szabo (2006), Eq. (4)")


def _check_gumbel_moments():
    # for k(F) = A exp(F/w), F/w + ln(A w / r) is the logarithm of a unit exponential variable, so
    # (Gumbel law for minima) the mean is w (ln(r/(A w)) - gamma_E) and the variance (pi^2/6) w^2,
    # variance (pi^2/6) w^2, up to corrections of order A w / r (here 1e-12)
    A, w, r = 1e-9, 0.7, 700.0
    dist = switching_distribution(lambda f: A * np.exp(f / w), r, 60.0, n=33)
    rows = [("mean", dist.mean, w * (np.log(r / (A * w)) - 0.5772156649015329)),
            ("variance", dist.variance, np.pi**2 / 6.0 * w * w)]
    return relative(rows, 1e-9, ref="Gumbel law (exact for an exponential rate)")


register(Entry(
    id="K0024",
    slug="switching-distribution-ramp",
    title="Distribution of the switching point under a ramp (quasi-static approximation)",
    area="driven",
    kind="exact",
    statement=(
        "When a control (a force, a current, a field) is ramped and the system escapes with "
        "the rate k(F) of the frozen value F, the switching value has density "
        "k(F)/Fdot exp(-integral of k/Fdot). This turns any rate law into switching-current, "
        "rupture-force or switching-field histograms, and is exact for a rate process; the "
        "rate picture itself fails for ramps that are fast compared with relaxation in the well."
    ),
    model=r"\dot S(t)=-k(F(t))\,S(t),\quad \dot F = \dot F(F) > 0",
    formula=r"p(F)=\frac{k(F)}{\dot F(F)}\exp\!\left(-\int_{F_0}^{F}\frac{k(F')}{\dot F(F')}\,dF'\right)",
    function=switching_distribution,
    parameters={"rate": "k(F)", "loading_rate": "dF/dt (number or function of F)", "F_max": "upper end of the grid",
                "F0": "starting value", "n": "Chebyshev order per panel"},
    sources=(
        Source(
            citation="O. K. Dudko, G. Hummer and A. Szabo, Phys. Rev. Lett. 96, 108101 (2006)",
            doi="10.1103/PhysRevLett.96.108101",
            locator="Eqs. (1)-(2), p. 108101-2",
            version_read="published PDF in Zenodo record 1233953 (CC0), opened 14 Oct 2026",
            printed=r"p(F)=\frac{k(F)}{\dot F}\,e^{-\int_0^F[k(F')/\dot F']dF'}",
            notation="same symbols",
        ),
        Source(
            citation="O. K. Dudko, G. Hummer and A. Szabo, Proc. Natl. Acad. Sci. USA 105, 15755 (2008)",
            doi="10.1073/pnas.0806085105",
            locator="Eq. [1], p. 15755",
            version_read="PubMed Central PMC2572921 equation images, opened 14 Oct 2026",
            role="restatement",
        ),
        Source(
            citation="C. Herbert and F. Bouchet, Phys. Rev. E 96, 030201 (2017)",
            doi="10.1103/PhysRevE.96.030201",
            locator="unnumbered survival law above Eq. (4) (arXiv:1703.01450v2, p. 4): the same law for the noisy saddle-node",
            version_read="arXiv:1703.01450v2, opened 14 Oct 2026 (journal version not opened)",
            role="restatement",
        ),
        Source(
            citation="J. Kurkijarvi, Phys. Rev. B 6, 832 (1972); T. A. Fulton and L. N. Dunkleberger, Phys. Rev. B 9, 4760 (1974); A. Garg, Phys. Rev. B 51, 15592 (1995)",
            doi="10.1103/PhysRevB.6.832",
            locator="the switching-current problem; Garg for general rate laws",
            version_read="abstracts only (closed access)",
            role="original",
            evidence="citation-only",
        ),
    ),
    validity=("Exact for a process with the frozen rate k(F). The frozen-rate picture needs the "
              "ramp to be slow compared with relaxation in the well; for the ramped noisy fold "
              "its error is mapped in K0030."),
    checks=(
        Check("Bell rate: closed form", "integral of an exponential", _check_bell_closed_form),
        Check("against the density printed for the Dudko-Hummer-Szabo model", "Dudko, Hummer and Szabo (2006), Eq. (4)", _check_dhs_density),
        Check("mean and variance for an exponential rate", "Gumbel law", _check_gumbel_moments),
    ),
    aliases=("switching-current distribution", "rupture-force distribution", "switching-field distribution",
             "Kurkijarvi distribution", "escape under a linear ramp", "dynamic force spectroscopy"),
    related=("K0025", "K0026", "K0027", "K0028", "K0029", "K0030"),
    notes=("Herbert and Bouchet's Eq. (5) (arXiv v2), the large-noise moments, is inconsistent with "
           "their Eq. (4) by a factor 1/(2 pi) (found by a research agent with its own numerics; not "
           "re-derived here, and the journal version was not opened)."),
    added="2026-10-14",
))
