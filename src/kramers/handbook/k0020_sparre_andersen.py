"""K0020. The Sparre Andersen theorem: universal survival of a symmetric random walk."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaln

from .._checking import combine, relative, statistical
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain


def sparre_andersen_survival(n: ArrayLike) -> np.ndarray | float:
    """Probability that a walk with i.i.d. symmetric continuous jumps, started at 0, stays positive for n steps.

    q_n = binom(2n, n) / 4^n, whatever the jump distribution (Levy flights included).
    """
    arr = np.asarray(n)
    if np.any(arr < 0) or np.any(arr != np.floor(arr)):
        raise OutsideDomain("n must be a non-negative integer")
    k = arr.astype(float)
    q = np.exp(gammaln(2 * k + 1) - 2 * gammaln(k + 1) - 2 * k * np.log(2.0))
    return float(q) if q.ndim == 0 else q


def _check_generating_function():
    n = np.arange(0, 20001)
    q = sparre_andersen_survival(n)
    rows = []
    for s in (0.3, 0.7, 0.9):
        rows.append((f"sum q_n s^n, s={s}", float(np.sum(q * s**n)), 1.0 / np.sqrt(1.0 - s)))
    gen = relative(rows, 1e-12, ref="1/sqrt(1-s) (Bray, Majumdar and Schehr Eq. 13)")
    asym = relative([("q_n sqrt(pi n), n = 1e6", float(sparre_andersen_survival(1_000_000) * np.sqrt(np.pi * 1e6)), 1.0)], 1e-6,
                    ref="1 (Eq. 15)")
    return combine(gen, asym)


def _check_simulation():
    rng = np.random.default_rng(81)
    n_steps, n_walks = 12, 200_000
    rows = []
    jumps = {
        "Gaussian": lambda size: rng.standard_normal(size),
        "uniform": lambda size: rng.uniform(-1, 1, size),
        "Cauchy (Levy flight)": lambda size: rng.standard_cauchy(size),
        "Laplace": lambda size: rng.laplace(size=size),
    }
    q = sparre_andersen_survival(np.array([1, 4, 12]))
    for name, draw in jumps.items():
        x = np.cumsum(draw((n_walks, n_steps)), axis=1)
        alive = np.cumprod(x > 0, axis=1)
        for i, step in enumerate((1, 4, 12)):
            p = float(alive[:, step - 1].mean())
            rows.append((f"{name}, n={step}", float(q[i]), p, float(np.sqrt(p * (1 - p) / n_walks))))
    return statistical(rows, n_sigma=4.5)


register(Entry(
    id="K0020",
    slug="sparre-andersen",
    title="Sparre Andersen theorem: universal survival probability of a symmetric random walk",
    area="active",
    kind="exact",
    statement=(
        "For a random walk on the line with independent jumps drawn from any symmetric, "
        "continuous distribution (Levy flights included), the probability of staying "
        "positive for n steps is binom(2n, n)/4^n, which decays as 1/sqrt(pi n)."
    ),
    model=r"x_n = x_{n-1}+\xi_n,\ x_0=0,\ \xi_n\ \text{i.i.d., symmetric, continuous}",
    formula=r"q(n)={2n\choose n}2^{-2n},\quad \sum_{n\ge0}q(n)s^n=\frac{1}{\sqrt{1-s}},\quad q(n)\sim\frac{1}{\sqrt{\pi n}}",
    function=sparre_andersen_survival,
    parameters={"n": "number of steps"},
    sources=(
        Source(
            citation="A. J. Bray, S. N. Majumdar and G. Schehr, Adv. Phys. 62, 225 (2013)",
            doi="10.1080/00018732.2013.803819",
            locator="Eqs. (13)-(15), Sec. 2.1 (arXiv:1304.1195v1 numbering)",
            version_read="arXiv:1304.1195v1 HTML, opened 14 Oct 2026",
            printed=r"q(n)=Q(0,n)={2n\choose n}2^{-2n}",
            notation="same symbols",
        ),
        Source(
            citation="S. N. Majumdar, Physica A 389, 4299 (2010)",
            doi="10.1016/j.physa.2010.01.021",
            locator="Eqs. (35)-(37) (arXiv:0912.2586v3 numbering)",
            version_read="arXiv:0912.2586v3 HTML, opened 14 Oct 2026",
            role="restatement",
        ),
        Source(
            citation="E. Sparre Andersen, Math. Scand. 1, 263 (1953); 2, 195 (1954)",
            doi=None,
            locator="the original theorem",
            version_read="not opened; citation as given by the two reviews",
            role="original",
            evidence="citation-only",
        ),
    ),
    validity="Symmetric and continuous jump density; start at the origin. Exact for every n. Lattice walks differ.",
    checks=(
        Check("generating function and asymptotics", "Bray, Majumdar and Schehr Eqs. (13), (15)", _check_generating_function),
        Check("four jump laws against simulation", "200,000 walks each: Gaussian, uniform, Cauchy, Laplace", _check_simulation),
    ),
    aliases=("universal first passage", "persistence of random walks", "Levy flight survival"),
    related=(),
    added="2026-10-14",
))
