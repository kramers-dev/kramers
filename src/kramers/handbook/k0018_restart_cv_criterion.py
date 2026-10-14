"""K0018. When does a small restart rate help? The coefficient-of-variation criterion."""

from __future__ import annotations

import numpy as np

from .._checking import combine, relative
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from .k0017_restart_renewal import _inverse_gaussian_transform, restart_moments


def restart_initial_slope(mean: float, second_moment: float) -> float:
    """d<T_r>/dr at r = 0: <T>^2 - <T^2>/2 = (<T>^2 - var T)/2. Negative (restart helps) iff CV > 1."""
    if not (mean > 0 and second_moment >= mean * mean):
        raise OutsideDomain("need a positive mean and second moment >= mean^2")
    return float(mean * mean - second_moment / 2.0)


def _check_derivative():
    rows = []
    signs = []
    # CV^2 = 2D/(x0 v): 4, 0.5, and just either side of 1
    for x0, v, D in ((1.0, 0.5, 1.0), (1.0, 4.0, 1.0), (1.9, 1.0, 1.0), (2.1, 1.0, 1.0)):
        mean = x0 / v
        var = 2 * D * x0 / v**3  # inverse Gaussian: CV^2 = 2D / (x0 v)
        slope = restart_initial_slope(mean, var + mean * mean)
        Tt = _inverse_gaussian_transform(x0, v, D)
        h = 1e-4 / mean
        # second-order one-sided estimate at 0 from r = h and 2h with <T_0> known
        numerical = (4 * restart_moments(Tt, h).mean - restart_moments(Tt, 2 * h).mean - 3 * mean) / (2 * h)
        rows.append((f"x0={x0}, v={v}, CV={np.sqrt(var) / mean:.2f}", slope, numerical))
        signs.append((f"sign, CV={np.sqrt(var) / mean:.2f}", float(np.sign(slope)), float(np.sign(1.0 - np.sqrt(var) / mean))))
    return combine(relative(rows, 1e-4, ref="derivative of K0017 at r -> 0"),
                   relative(signs, 0.0, what="sign of slope", ref="sign of (1 - CV)"))


register(Entry(
    id="K0018",
    slug="restart-cv-criterion",
    title="Restart helps at small rates exactly when the coefficient of variation exceeds one",
    area="resetting",
    kind="exact",
    statement=(
        "Introducing restart at a small rate r changes the mean completion time by r times "
        "(<T>^2 - var T)/2: it lowers the mean if and only if the standard deviation of the "
        "completion time exceeds its mean."
    ),
    model=r"\text{as K0017}",
    formula=r"\left.\frac{d\langle T_r\rangle}{dr}\right|_{r=0}=\frac{\langle T\rangle^2-\sigma^2(T)}{2},\quad \text{restart helps}\iff \frac{\sigma(T)}{\langle T\rangle}>1",
    function=restart_initial_slope,
    parameters={"mean": "<T> without restart", "second_moment": "<T^2> without restart"},
    sources=(
        Source(
            citation="A. Pal and S. Reuveni, Phys. Rev. Lett. 118, 030603 (2017)",
            doi="10.1103/PhysRevLett.118.030603",
            locator="statement before Eq. (9), proof in the Supplemental Material (arXiv:1607.06048v3)",
            version_read="arXiv:1607.06048v3 HTML, opened 14 Oct 2026 (stated in prose; SI not opened)",
            printed=r"\text{the slope of }\langle T_R\rangle\text{ at } r=0 \text{ is negative whenever } \sigma(T)/\langle T\rangle>1",
            notation="R exponential with rate r",
        ),
        Source(
            citation="H. Chen and F. Huang, Phys. Rev. E 105, 034109 (2022)",
            doi="10.1103/PhysRevE.105.034109",
            locator="Eq. (1) (arXiv:2109.11101v2 numbering)",
            version_read="arXiv:2109.11101v2 HTML, opened 14 Oct 2026",
            printed=r"{\rm CV}=\frac{\sqrt{\langle\tau_0^2\rangle-\langle\tau_0\rangle^2}}{\langle\tau_0\rangle}>1",
            role="restatement",
        ),
        Source(
            citation="S. Reuveni, Phys. Rev. Lett. 116, 170601 (2016)",
            doi="10.1103/PhysRevLett.116.170601",
            locator="the same statement in residual-life form",
            version_read="arXiv:1512.01600v1 HTML, opened 14 Oct 2026",
            role="restatement",
        ),
    ),
    validity="Finite second moment. Local in r: CV < 1 does not exclude a gain at an intermediate rate.",
    checks=(
        Check("slope and sign against the derivative of the renewal formula", "K0017 with the inverse-Gaussian transform", _check_derivative),
    ),
    aliases=("restart criterion", "when does resetting help"),
    related=("K0016", "K0017"),
    added="2026-10-14",
))
