"""K0025. From a switching histogram to escape rates (the Fulton-Dunkleberger transformation)."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from .._checking import statistical
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain


class RateEstimate(NamedTuple):
    centres: np.ndarray
    rate: np.ndarray
    stderr: np.ndarray


def histogram_to_rate(counts: ArrayLike, edges: ArrayLike, ramp_rate: float) -> RateEstimate:
    """Escape rate at each bin centre from switching counts under a linear ramp.

    Gamma(I_k) = |dI/dt| / Delta I_k * ln( sum_{j >= k} n_j / sum_{j >= k+1} n_j )
    (Kaemmerer et al. Eq. 13, after Fulton and Dunkleberger), assigned to bin centres,
    where it is second order in the bin width. ``stderr`` is the delta-method standard
    error from binomial counting, |dI/dt| / Delta I_k * sqrt(n_k / (N_k N_{k+1})), a
    computed addition, not in the source. Bins with no later switches give NaN.
    """
    n = np.asarray(counts, dtype=float)
    e = np.asarray(edges, dtype=float)
    if e.size != n.size + 1 or np.any(np.diff(e) <= 0):
        raise OutsideDomain("edges must be increasing with one more entry than counts")
    if np.any(n < 0) or not ramp_rate != 0:
        raise OutsideDomain("counts must be non-negative and the ramp rate non-zero")
    width = np.diff(e)
    N_k = np.cumsum(n[::-1])[::-1]  # switches in bins k and above
    N_next = np.concatenate([N_k[1:], [0.0]])
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = abs(ramp_rate) / width * np.log(N_k / N_next)
        se = abs(ramp_rate) / width * np.sqrt(n / (N_k * N_next))
    bad = N_next <= 0
    rate[bad] = np.nan
    se[bad] = np.nan
    return RateEstimate(centres=0.5 * (e[:-1] + e[1:]), rate=rate, stderr=se)


def _check_synthetic():
    # switching currents drawn exactly from a known rate law under a ramp, then recovered
    rng = np.random.default_rng(101)
    Ic, B, omega, v = 1.0, 40.0, 1e4, 1.0
    rate = lambda I: omega * np.clip(1 - I, 0, None) ** 0.5 * np.exp(-B * np.clip(1 - I, 0, None) ** 1.5)  # noqa: E731
    grid = np.linspace(0.5, Ic, 200_001)
    hazard = np.concatenate([[0.0], np.cumsum(0.5 * (rate(grid[1:]) + rate(grid[:-1])) * np.diff(grid))]) / v
    u = -np.log(rng.random(200_000))
    I_sw = np.interp(u, hazard, grid)
    # every switch must fall inside the histogram (the estimator counts survivors)
    edges = np.linspace(I_sw.min(), I_sw.max() + 1e-12, 41)
    counts, _ = np.histogram(I_sw, bins=edges)
    est = histogram_to_rate(counts, edges, v)
    rows = []
    for c, r, s in zip(est.centres, est.rate, est.stderr, strict=True):
        if np.isfinite(r) and s / r < 0.05:
            rows.append((f"I = {c:.4f}", float(rate(c)), float(r), float(s)))
    return statistical(rows, n_sigma=4.0, what="true rate", ref="recovered")


register(Entry(
    id="K0025",
    slug="fulton-dunkleberger",
    title="Escape rates from a switching histogram (Fulton-Dunkleberger transformation)",
    area="driven",
    kind="estimator",
    statement=(
        "Counts of switching events in bins of the ramped control (current, force, field) give "
        "the escape rate at each bin from the logarithm of the ratio of survivors before and "
        "after the bin. Assigned to bin centres it is accurate to second order in the bin "
        "width; the standard error from counting statistics is added here."
    ),
    model=r"\dot S = -\Gamma(I)\,S,\ I \text{ ramped at constant } |\dot I|",
    formula=r"\Gamma(I_k)=\frac{|\dot I|}{\Delta I}\ln\!\left(\frac{\sum_{j\ge k}P_j}{\sum_{j\ge k+1}P_j}\right)",
    function=histogram_to_rate,
    parameters={"counts": "switching events per bin", "edges": "bin edges (increasing)", "ramp_rate": "dI/dt"},
    sources=(
        Source(
            citation="J. N. Kaemmerer, S. Masis, K. Hambardzumyan, P. Lenhard, U. Strobel, J. Lisenfeld, H. Rotzinger, A. V. Ustinov et al., arXiv:2411.15048 (2024)",
            doi=None,
            locator="Eqs. (11)-(13), p. 4",
            version_read="arXiv:2411.15048v1 PDF, opened 14 Oct 2026 (preprint)",
            printed=r"\Gamma(I_k)=\frac{|\dot I|}{\Delta I}\ln\left(\frac{\sum_{j\ge k}P_j}{\sum_{j\ge k+1}P_j}\right)",
            notation="P_j = n_j / (N Delta I); bins numbered upwards in current",
        ),
        Source(
            citation="T. A. Fulton and L. N. Dunkleberger, Phys. Rev. B 9, 4760 (1974)",
            doi="10.1103/PhysRevB.9.4760",
            locator="the original transformation",
            version_read="not opened (closed access); credited by Kaemmerer et al.",
            role="original",
            evidence="citation-only",
        ),
    ),
    validity=("A single switching event per ramp, constant ramp rate, rate process (quasi-static). "
              "Bin centres: second order in the bin width (relative errors 1.9e-2, 4.9e-3, 1.2e-3 at "
              "widths 2, 1, 0.5 kT/x, from a research agent's numerics); bin edges: first order."),
    checks=(
        Check("recovers a known rate from 200,000 synthetic switches", "exact sampling from the survival law", _check_synthetic),
    ),
    aliases=("switching histogram to rate", "Josephson switching current distribution analysis",
             "histogram inversion", "escape rate from switching statistics"),
    related=("K0024", "K0028"),
    added="2026-10-14",
))
