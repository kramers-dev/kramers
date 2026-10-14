"""K0009. Firing rate of the quadratic integrate-and-fire neuron driven by white noise."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma

from .._checking import relative
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from ..solvers.quadrature import FirstPassage1D


def qif_firing_rate(mu: float, sigma: float, tau_m: float) -> float:
    """Stationary rate of tau_m dv/dt = v^2 + mu + tau_m^(1/2) sigma eta(t), spikes at +inf, reset to -inf.

    Brunel and Latham's single-integral form, Eq. (3.18a): nu = 1 / (pi tau_m) / I with
    I = integral over xi of exp(-mu xi^2 - sigma^4 xi^6 / 48) / sqrt(pi).
    """
    if not tau_m > 0 or sigma < 0:
        raise OutsideDomain("tau_m must be positive and sigma non-negative")
    if sigma == 0:
        if mu <= 0:
            return 0.0
        return float(np.sqrt(mu) / (np.pi * tau_m))
    c = sigma**4 / 48.0
    # exponent g(xi) = -mu xi^2 - c xi^6 has its maximum at xi^2 = -mu / (3c) when mu < 0
    if mu < 0:
        x2 = -mu / (3.0 * c)
        peak = -mu * x2 - c * x2**3
        xs = np.sqrt(x2)
    else:
        peak, xs = 0.0, 0.0
    upper = max(4.0 * xs, (40.0 / c) ** (1.0 / 6.0) + xs + 1.0)
    pts = [xs] if xs > 0 else None
    val = quad(lambda x: np.exp(-mu * x * x - c * x**6 - peak), 0.0, upper, points=pts,
               epsabs=0.0, epsrel=1e-13, limit=200)[0]
    integral = 2.0 * val / np.sqrt(np.pi)
    return float(np.exp(-peak) / (np.pi * tau_m * integral))


def _check_quadrature():
    rows = []
    for mu, sigma, tau in ((-1.0, 1.0, 10.0), (0.0, 0.5, 1.0), (0.7, 2.0, 5.0), (-3.0, 1.5, 2.0)):
        D = sigma**2 / (2.0 * tau)  # diffusion coefficient of v in the white-noise limit
        fp = FirstPassage1D(log_psi=lambda v, mu=mu, D=D, tau=tau: (v**3 / 3.0 + mu * v) / (tau * D), D=D,
                            left="natural", right="exit", dlog_psi=lambda v, mu=mu, D=D, tau=tau: (v * v + mu) / (tau * D))
        rows.append((f"mu={mu}, sigma={sigma}, tau_m={tau}", qif_firing_rate(mu, sigma, tau),
                     1.0 / float(fp.mean_time(-np.inf).value)))
    return relative(rows, 1e-10, ref="quadrature solver, passage -inf to +inf")


def _check_printed_limits():
    tau = 10.0
    rows = [("sigma=0, mu=0.49", qif_firing_rate(0.49, 0.0, tau), 0.7 / (np.pi * tau))]
    sigma = 0.8
    at_zero = 3.0 ** (5.0 / 6.0) * sigma ** (2.0 / 3.0) / (2.0 ** (2.0 / 3.0) * np.sqrt(np.pi) * gamma(1.0 / 6.0) * tau)
    rows.append(("mu=0, sigma=0.8", qif_firing_rate(0.0, sigma, tau), at_zero))
    return relative(rows, 1e-11, ref="limits printed on pp. 2288-2289")


register(Entry(
    id="K0009",
    slug="qif-firing-rate",
    title="Firing rate of the quadratic integrate-and-fire neuron with white noise",
    area="neurons",
    kind="exact",
    statement=(
        "The stationary firing rate of the quadratic integrate-and-fire (theta) neuron driven "
        "by white noise, for any mean input mu (below or above threshold), as one integral. "
        "It is the reciprocal of the passage time through a noisy saddle-node (K0004)."
    ),
    model=r"\tau_m\dot v = v^2 + \mu + \tau_m^{1/2}\sigma\,\eta(t),\ \text{spike at } v=+\infty,\ \text{reset to } -\infty",
    formula=(r"\nu = \frac{1}{\pi\tau_m}\left[\int_{-\infty}^{\infty}\frac{d\xi}{\pi^{1/2}}"
             r"\exp(-\mu\xi^2-\sigma^4\xi^6/48)\right]^{-1}"),
    function=qif_firing_rate,
    parameters={"mu": "mean input (dimensionless current)", "sigma": "noise amplitude",
                "tau_m": "membrane time constant"},
    sources=(
        Source(
            citation="N. Brunel and P. E. Latham, Neural Comput. 15, 2281 (2003)",
            doi="10.1162/089976603322362365",
            locator="Eq. (3.18a), p. 2288; model Eqs. (2.1a), (2.1b), (2.2), p. 2283",
            version_read="publisher's PDF hosted by the Nirenberg lab, page images read 14 Oct 2026",
            printed=(r"\nu_{0S}=\frac{1}{\pi\tau_m}\left[\int_{-\infty}^{\infty}\frac{d\xi}{\pi^{1/2}}"
                     r"\exp[-\mu\xi^2-\sigma^4\xi^6/48]\right]^{-1}"),
            notation="nu_0S = nu; zeroth order in tau_s/tau_m (white noise)",
        ),
        Source(
            citation="B. Lindner, A. Longtin and A. Bulsara, Neural Comput. 15, 1761 (2003)",
            doi="10.1162/08997660360675035",
            locator="Eq. (3.1), p. 1770 (double-integral form)",
            version_read="author's PDF, page image read 12 Oct 2026",
            role="restatement",
        ),
    ),
    validity="White noise (synaptic time constant negligible against tau_m); any mu. Exact.",
    checks=(
        Check("against the quadrature solver", "Green's-function quadrature, entrance and exit tails", _check_quadrature),
        Check("the printed limits sigma = 0 and mu = 0", "Brunel and Latham (2003), pp. 2288-2289", _check_printed_limits),
    ),
    aliases=("theta neuron firing rate", "type I neuron", "saddle-node on invariant circle"),
    related=("K0004", "K0005", "K0010"),
    added="2026-10-14",
))
