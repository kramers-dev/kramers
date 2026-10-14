"""K0028. From rupture-force histograms to force-dependent lifetimes (Dudko, Hummer and Szabo, 2008)."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from .._checking import relative, statistical
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain


class LifetimeEstimate(NamedTuple):
    centres: np.ndarray
    lifetime: np.ndarray
    stderr: np.ndarray


def histogram_to_lifetime(counts: ArrayLike, edges: ArrayLike, loading_rate: float | Callable[[np.ndarray], np.ndarray]) -> LifetimeEstimate:
    """tau(F) at bin centres: (h_k/2 + sum_{i>k} h_i) dF / (h_k Fdot(F_k)) (Eq. [10] of DHS 2008).

    ``loading_rate`` may depend on force (a linker). ``stderr`` is a delta-method error
    from multinomial counts, a computed addition. Empty bins, and bins with no ruptures
    above them, give NaN.
    """
    n = np.asarray(counts, dtype=float)
    e = np.asarray(edges, dtype=float)
    if e.size != n.size + 1 or np.any(np.diff(e) <= 0):
        raise OutsideDomain("edges must be increasing with one more entry than counts")
    width = np.diff(e)
    centres = 0.5 * (e[:-1] + e[1:])
    Fdot = loading_rate(centres) if callable(loading_rate) else np.full_like(centres, float(loading_rate))
    total = n.sum()
    h = n / (total * width)
    above = np.concatenate([np.cumsum((h * width)[::-1])[::-1][1:], [0.0]])  # probability in higher bins
    with np.errstate(divide="ignore", invalid="ignore"):
        surv = h * width / 2.0 + above
        tau = surv / (h * Fdot)
        # error: tau ~ (n_k/2 + A) / n_k * dF / Fdot, with A = counts above; Var from Poisson counts
        A = above * total
        se = width / Fdot * np.sqrt(A**2 / n**3 + A / n**2)
    undefined = (n == 0) | (above <= 0)  # empty bin, or no ruptures above it (the last bin)
    tau[undefined] = np.nan
    se[undefined] = np.nan
    return LifetimeEstimate(centres=centres, lifetime=tau, stderr=se)


def _check_synthetic():
    rng = np.random.default_rng(111)
    k0, x, rate = 1e-3, 0.5, 5.0  # Bell rate, kT = 1
    k = lambda F: k0 * np.exp(F * x)  # noqa: E731
    u = -np.log(rng.random(200_000))
    # inverse of the cumulative hazard H(F) = k0/(x rate) (e^{F x} - 1)
    F_sw = np.log1p(u * x * rate / k0) / x
    # the estimator needs every rupture inside the histogram
    edges = np.linspace(F_sw.min(), F_sw.max() + 1e-9, 41)
    counts, _ = np.histogram(F_sw, bins=edges)
    est = histogram_to_lifetime(counts, edges, rate)
    at_or_above = np.cumsum(counts[::-1])[::-1]
    rows = []
    for c, tau, s, n_k, N_k in zip(est.centres, est.lifetime, est.stderr, counts, at_or_above, strict=True):
        # within the stated validity: fewer than 10% of the survivors fail inside the bin
        if np.isfinite(tau) and s / tau < 0.05 and n_k / N_k < 0.1:
            rows.append((f"F = {c:.3f}", float(1.0 / k(c)), float(tau), float(s)))
    return statistical(rows, n_sigma=4.0, what="true lifetime", ref="recovered")


def _check_exact_histogram():
    # on an exact (infinite-sample) histogram the printed estimator equals (dF/Fdot) / (2 tanh(lambda/2)),
    # lambda = ln(S_k / S_k+1); the true mean lifetime over the bin is (dF/Fdot)/lambda
    k0, x, rate = 1e-3, 0.5, 5.0
    edges = np.linspace(4.0, 24.0, 41)
    S = np.exp(-k0 / (x * rate) * np.expm1(edges * x))
    counts = (S[:-1] - S[1:]) * 1e12
    est = histogram_to_lifetime(counts, edges, rate)
    lam = np.log(S[:-1] / S[1:])
    dF = np.diff(edges)
    rows = []
    for i in range(0, 39, 6):
        rows.append((f"bin {i}, {100 * (1 - np.exp(-lam[i])):.1f}% fail in it", float(est.lifetime[i]),
                     float(dF[i] / rate / (2.0 * np.tanh(lam[i] / 2.0)))))
    return relative(rows, 1e-9, ref="(dF/Fdot) / (2 tanh(lambda/2))")


register(Entry(
    id="K0028",
    slug="dhs-lifetime-transform",
    title="Force-dependent lifetimes from rupture-force histograms (Dudko, Hummer and Szabo, 2008)",
    area="driven",
    kind="estimator",
    statement=(
        "A histogram of rupture forces at one loading rate gives the lifetime of the bond at "
        "each force: the probability of surviving to that force divided by the density of "
        "ruptures there and the loading rate. Histograms from different loading rates must "
        "collapse onto one curve if the rate picture holds, which is a test of the model."
    ),
    model=r"\tau(F)=\frac{\int_F^\infty p(f)\,df}{\dot F(F)\,p(F)}",
    formula=r"\tau\big(F_0+(k-\tfrac12)\Delta F\big)=\frac{\left(h_k/2+\sum_{i=k+1}^{N}h_i\right)\Delta F}{h_k\,\dot F\big(F_0+(k-\tfrac12)\Delta F\big)}",
    function=histogram_to_lifetime,
    parameters={"counts": "ruptures per force bin", "edges": "bin edges", "loading_rate": "dF/dt, number or function of F"},
    sources=(
        Source(
            citation="O. K. Dudko, G. Hummer and A. Szabo, Proc. Natl. Acad. Sci. USA 105, 15755 (2008)",
            doi="10.1073/pnas.0806085105",
            locator="Eq. [10], p. 15759 (Methods); continuous form Eq. [2], p. 15756",
            version_read="PubMed Central PMC2572921 equation image of Eq. [10], opened 14 Oct 2026",
            printed=(r"\tau\big(F_0+(k-\tfrac12)\Delta F\big)=\frac{\left(h_k/2+\sum_{i=k+1}^{N}h_i\right)\Delta F}"
                     r"{h_k\,\dot F\big(F_0+(k-\tfrac12)\Delta F\big)}"),
            notation="h_i = C_i / (N_tot Delta F); bin k spans [F0 + (k-1) dF, F0 + k dF]",
        ),
        Source(
            citation="Z. Liu, H. Liu, A. M. Vera, R. C. Bernardi, P. Tinnefeld and M. A. Nash, Nat. Commun. 11, 4321 (2020)",
            doi="10.1038/s41467-020-18063-x",
            locator="Eq. (3), Methods (the same estimator as k_off)",
            version_read="Europe PMC full-text XML PMC7456326, opened 14 Oct 2026",
            role="restatement",
        ),
    ),
    validity=("Rate process with single-exponential kinetics at constant force; quasi-adiabatic "
              "loading; irreversible rupture; every rupture inside the histogram. On an exact "
              "histogram the estimator is too high by a factor lambda / (2 tanh(lambda/2)), about "
              "1 + lambda^2/12, where lambda = ln(S_k/S_k+1) (computed check): 0.09% when 10% of the "
              "survivors fail in a bin, 0.7% at 25%, 4% at 50%. The logarithmic form (K0025) has no "
              "such bias."),
    checks=(
        Check("on an exact histogram it equals its known discretisation", "(dF/Fdot) / (2 tanh(lambda/2))", _check_exact_histogram),
        Check("recovers a known lifetime from 200,000 synthetic ruptures", "exact sampling from the Bell rate", _check_synthetic),
    ),
    aliases=("force spectroscopy histogram transformation", "lifetime from rupture forces", "k_off from histograms"),
    related=("K0024", "K0025", "K0026"),
    added="2026-10-14",
))
