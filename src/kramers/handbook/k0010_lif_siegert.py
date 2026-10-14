"""K0010. Firing rate of the leaky integrate-and-fire neuron with white noise (the Siegert formula)."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import erfcx

from .._checking import relative, statistical
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from ..solvers import simulate
from ..solvers.quadrature import FirstPassage1D


def lif_firing_rate(mu: float, sigma: float, theta: float, V_r: float, tau: float, tau_rp: float = 0.0) -> float:
    """Stationary rate of tau dV/dt = -V + mu + sigma sqrt(tau) eta(t), threshold theta, reset V_r.

    Brunel's Eq. (21): 1/nu = tau_rp + tau sqrt(pi) * integral from (V_r - mu)/sigma to
    (theta - mu)/sigma of exp(u^2) (1 + erf u) du.

    Note the convention: the free membrane potential has standard deviation
    sigma / sqrt(2). Pass ``sigma = sqrt(2) * sd`` if you hold the standard deviation.
    """
    if not (tau > 0 and sigma > 0 and theta > V_r and tau_rp >= 0):
        raise OutsideDomain("need tau > 0, sigma > 0, theta > V_r, tau_rp >= 0")
    lo, hi = (V_r - mu) / sigma, (theta - mu) / sigma
    if hi > 26.0:
        raise OutsideDomain("threshold more than 26 sigma above the mean input: rate below 1e-290")
    # exp(u^2) (1 + erf u) = erfcx(-u), which neither overflows nor cancels
    val = quad(lambda u: erfcx(-u), lo, hi, epsabs=0.0, epsrel=1e-13, limit=200)[0]
    return float(1.0 / (tau_rp + tau * np.sqrt(np.pi) * val))


def _check_quadrature():
    rows = []
    for mu, sigma, theta, Vr, tau in ((15.0, 5.0, 20.0, 10.0, 20.0), (25.0, 2.0, 20.0, 10.0, 20.0),
                                      (5.0, 3.0, 20.0, 10.0, 10.0)):
        D = sigma**2 / (2.0 * tau)
        fp = FirstPassage1D(lambda V, mu=mu, sigma=sigma: (V - mu) ** 2 / sigma**2, D=D,
                            domain=(-np.inf, theta), left="natural", right="absorbing")
        T = float(fp.mean_time(Vr).value)
        rows.append((f"mu={mu}, sigma={sigma}", lif_firing_rate(mu, sigma, theta, Vr, tau), 1.0 / T))
    return relative(rows, 1e-10, ref="quadrature solver, Ornstein-Uhlenbeck to threshold")


def _check_alternative_form():
    # Brunel and Hakim (1999), Eq. (14): 1/(nu tau) = int_0^inf exp(-u^2) (exp(2 y_t u) - exp(2 y_r u)) / u du
    rows = []
    for mu, sigma, theta, Vr, tau in ((15.0, 5.0, 20.0, 10.0, 20.0), (18.0, 1.0, 20.0, 10.0, 20.0)):
        yt, yr = (theta - mu) / sigma, (Vr - mu) / sigma
        def integrand(u: float, yt: float = yt, yr: float = yr) -> float:
            if u < 1e-8:  # the limit u -> 0
                return 2.0 * (yt - yr)
            return (np.exp(-u * u + 2 * yt * u) - np.exp(-u * u + 2 * yr * u)) / u

        upper = max(yt, 0.0) + 40.0
        val = quad(integrand, 0.0, upper, points=[max(yt, 1e-3)], epsabs=0.0, epsrel=1e-12, limit=400)[0]
        rows.append((f"mu={mu}, sigma={sigma}", lif_firing_rate(mu, sigma, theta, Vr, tau), 1.0 / (tau * val)))
    return relative(rows, 1e-9, ref="Brunel and Hakim (1999), Eq. (14)")


def _check_simulation():
    mu, sigma, theta, Vr, tau = 15.0, 5.0, 20.0, 10.0, 20.0
    sim = simulate.first_passage_times(lambda V: (mu - V) / tau, sigma**2 / (2 * tau), Vr, n_paths=20_000,
                                       dt=0.01, t_max=5000.0, absorbing=(None, theta), seed=21)
    m = sim.mean()
    return statistical([("mean interspike interval (ms)", 1.0 / lif_firing_rate(mu, sigma, theta, Vr, tau), m.value, m.error)])


register(Entry(
    id="K0010",
    slug="lif-siegert-rate",
    title="Firing rate of the leaky integrate-and-fire neuron with white noise (Siegert formula)",
    area="neurons",
    kind="exact",
    statement=(
        "The stationary firing rate of the leaky integrate-and-fire neuron with white-noise "
        "input is the reciprocal of the refractory period plus the mean first-passage time of "
        "an Ornstein-Uhlenbeck process from reset to threshold, written as one integral of "
        "exp(u^2)(1 + erf u)."
    ),
    model=r"\tau\dot V = -V + \mu + \sigma\sqrt{\tau}\,\eta(t),\ V=\theta \to \text{spike, reset to } V_r",
    formula=(r"\frac{1}{\nu}=\tau_{rp}+\tau\sqrt{\pi}\int_{(V_r-\mu)/\sigma}^{(\theta-\mu)/\sigma}"
             r"du\,e^{u^2}\left(1+\operatorname{erf}u\right)"),
    function=lif_firing_rate,
    parameters={"mu": "mean input", "sigma": "noise amplitude (free SD is sigma/sqrt 2)",
                "theta": "threshold", "V_r": "reset", "tau": "membrane time constant",
                "tau_rp": "refractory period"},
    sources=(
        Source(
            citation="N. Brunel, J. Comput. Neurosci. 8, 183 (2000)",
            doi="10.1023/A:1008925309027",
            locator="Eq. (21), p. 188; model Eqs. (1), (3), (6), pp. 185-186",
            version_read="publisher's PDF on a third-party university server, page image read 14 Oct 2026",
            printed=(r"\frac{1}{\nu_0}=\tau_{rp}+\tau\sqrt{\pi}\int_{\frac{V_r-\mu_0}{\sigma_0}}"
                     r"^{\frac{\theta-\mu_0}{\sigma_0}}du\,e^{u^2}(1+\mathrm{erf}(u))"),
            notation="mu_0 = mu, sigma_0 = sigma; RI = mu + sigma sqrt(tau) eta (Eq. 3)",
        ),
        Source(
            citation="N. Brunel and V. Hakim, Neural Comput. 11, 1621 (1999)",
            doi="10.1162/089976699300016179",
            locator="Eq. (14) (arXiv cond-mat/9904278 numbering)",
            version_read="arXiv HTML, opened 14 Oct 2026",
            role="restatement",
        ),
        Source(
            citation="A. J. F. Siegert, Phys. Rev. 81, 617 (1951)",
            doi="10.1103/PhysRev.81.617",
            locator="the first-passage method behind the formula",
            version_read="not opened; citation only",
            role="original",
            evidence="citation-only",
        ),
    ),
    validity=("Diffusion approximation of the input (white noise); stationary state. Exact for "
              "that model. Threshold at most 26 sigma above the mean input."),
    checks=(
        Check("against the quadrature solver", "Ornstein-Uhlenbeck mean first-passage time by quadrature", _check_quadrature),
        Check("against Brunel and Hakim's alternative integral", "Brunel and Hakim (1999), Eq. (14)", _check_alternative_form),
        Check("against simulation", "Euler-Maruyama, 20,000 interspike intervals", _check_simulation, slow=True),
    ),
    aliases=("Siegert formula", "Ricciardi formula", "LIF transfer function", "Ornstein-Uhlenbeck first passage"),
    related=("K0009",),
    added="2026-10-14",
))
