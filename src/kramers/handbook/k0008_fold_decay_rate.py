"""K0008. Decay rate of the noisy fold (cubic well) at any barrier height: a computed reference."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import airy

from .._checking import relative
from ..registry import Check, Entry, register
from ..result import OutsideDomain
from ..solvers import fokker_planck
from ..solvers.quadrature import FirstPassage1D

# Lowest eigenvalue E0(u) of H = -d^2/dy^2 + (y^2 - u)^2/4 + y, by a Hermite-basis
# computation in 65-digit arithmetic (bases of 104 and 128 functions agree to the digits
# shown). Method and code: GPT-6 Astra, 9 October 2026, as an independent check for the
# geoews package (geoews/tools/reference_values.py); values reproduced by Claude.
HERMITE_REFERENCE = {
    0.0: 0.3541232444531332932242,
    0.25: 0.2589680423874863918907,
    0.5: 0.1833513839092425943653,
    1.0: 0.0820749685574378455751,
    2.0: 0.009534064393129959371442,
    4.0: 1.441249337263231883883e-05,
    4.5: 1.956154596511083624017e-06,
    8.0: 7.04105074706691e-14,
}


@lru_cache(maxsize=4096)
def _rate(u: float) -> float:
    fp = FirstPassage1D(lambda y: u * y - y**3 / 3.0, D=1.0, left="natural", right="exit",
                        dlog_psi=lambda y: y * y - u)
    return float(fp.decay_rate().value)


def fold_decay_rate(u: ArrayLike) -> np.ndarray | float:
    """Long-time decay rate E0(u) of survival in the scaled noisy fold dy = (y^2 - u) ds + sqrt(2) dW.

    ``u >= -2`` is the scaled distance to the fold (u > 0: a well and a barrier of
    4/3 u^(3/2) in units of the noise). In physical units, for dx = (r + x^2) dt +
    sqrt(D) dW, the rate is E0(u) * (D/2)^(1/3) with u = -r / (D/2)^(2/3).

    The value is the smallest eigenvalue of the backward operator, computed by power
    iteration on its Green's operator, which keeps full relative accuracy for rates as
    small as 1e-300. It is not the reciprocal mean passage time (K0004); the two agree
    only when the barrier is high.
    """
    arr = np.asarray(u, dtype=float)
    if np.any(arr < -2.0) or not np.all(np.isfinite(arr)):
        raise OutsideDomain("u must be finite and at least -2 (past the fold the decay is not slow)")
    out = np.vectorize(lambda v: _rate(float(round(v, 15))), otypes=[float])(arr)
    return float(out) if out.ndim == 0 else out


def _check_hermite():
    rows = [(f"u={u}", fold_decay_rate(u), ref) for u, ref in HERMITE_REFERENCE.items()]
    return relative(rows, 1e-12, ref="Hermite basis, 65 digits")


def _check_high_barrier():
    rows = []
    for u in (8.0, 12.0, 20.0):
        ai, _, bi, _ = airy(u)
        rows.append((f"u={u}", fold_decay_rate(u), 1.0 / (np.pi**2 * (ai**2 + bi**2))))
    return relative(rows, 1e-12, ref="1/T, T = pi^2 (Ai^2 + Bi^2) (K0004)")


def _check_fokker_planck():
    rows = []
    for u in (0.0, 1.0):
        sol = fokker_planck.solve(lambda y, u=u: y * y - u, 1.0, (-8.0, 7.0), left="reflecting", right="absorbing",
                                  x0=-np.sqrt(max(u, 0.0)) - 0.5, t_end=80.0 if u == 0 else 160.0, n_cells=1500,
                                  dt=0.02, richardson=False)
        s = sol.survival
        late = (s < 1e-3) & (s > 1e-9)
        slope = -np.polyfit(sol.t[late], np.log(s[late]), 1)[0]
        rows.append((f"u={u}", fold_decay_rate(u), float(slope)))
    return relative(rows, 2e-3, ref="slope of log-survival, time-dependent Fokker-Planck")


register(Entry(
    id="K0008",
    slug="fold-decay-rate",
    title="Decay rate of the noisy fold at any barrier height (computed reference)",
    area="escape",
    kind="computed-reference",
    statement=(
        "The long-time escape rate of the noisy saddle-node normal form, as the smallest "
        "eigenvalue of its backward operator, computed to full double precision at any "
        "distance from the fold, from the bifurcation point (rate 0.354) to barriers where "
        "it is 1e-50 and below."
    ),
    model=r"dy = (y^2 - u)\,ds + \sqrt{2}\,dW,\ y(0)\ \text{in the well},\ \text{escape to } +\infty",
    formula=r"S(s) \sim c\,e^{-E_0(u)\,s},\quad E_0 = \min\operatorname{spec}(-\mathcal{L}_u)",
    function=fold_decay_rate,
    parameters={"u": "scaled distance to the fold"},
    sources=(),
    validity="Any u >= -2. Double precision; agreement with 65-digit values to about 1e-15.",
    checks=(
        Check("against 65-digit Hermite-basis values", "independent high-precision computation", _check_hermite),
        Check("high barriers: equals the reciprocal mean passage time", "Airy closed form (K0004)", _check_high_barrier),
        Check("slope of the log-survival", "time-dependent Fokker-Planck solver", _check_fokker_planck, slow=True),
    ),
    aliases=("escape rate of the cubic potential", "Kramers problem for the saddle-node",
             "lowest eigenvalue of the Fokker-Planck operator", "quasi-stationary decay rate"),
    related=("K0004", "K0006"),
    notes=("The quasi-stationary decay rate and the reciprocal mean passage time are different "
           "observables: at the fold they differ by 43% (0.354 against 0.201)."),
    added="2026-10-14",
))
