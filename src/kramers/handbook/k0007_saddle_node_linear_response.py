"""K0007. Firing rate near a saddle-node for weak bias: the linear term."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import airy, gamma

from .._checking import relative
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain

_C0 = 3.0 ** (1.0 / 3.0) / gamma(1.0 / 3.0) ** 2
_C1 = 9.0 / 8.0 * 3.0 ** (1.0 / 6.0) / np.pi**3 * gamma(2.0 / 3.0) ** 4


def saddle_node_rate_linear(beta: ArrayLike, D: float) -> np.ndarray | float:
    """r ~ (3D)^(1/3)/Gamma(1/3)^2 + (9/8) 3^(1/6) Gamma(2/3)^4 / pi^3 * D^(-1/3) beta.

    Rate of passage for dx/dt = beta + x^2 + sqrt(2D) xi, to first order in
    beta / D^(2/3) (weak bias or strong noise).
    """
    if not D > 0:
        raise OutsideDomain("D must be positive")
    b = np.asarray(beta, dtype=float)
    r = _C0 * D ** (1.0 / 3.0) + _C1 * D ** (-1.0 / 3.0) * b
    return float(r) if r.ndim == 0 else r


def _exact_rate(beta: float, D: float) -> float:
    ai, _, bi, _ = airy(-beta / D ** (2.0 / 3.0))
    return D ** (1.0 / 3.0) / (np.pi**2 * (ai**2 + bi**2))


def _check_slope():
    rows = []
    for D in (0.1, 1.0, 10.0):
        h = 1e-4 * D ** (2.0 / 3.0)
        slope = (_exact_rate(h, D) - _exact_rate(-h, D)) / (2 * h)
        lin = (saddle_node_rate_linear(h, D) - saddle_node_rate_linear(-h, D)) / (2 * h)
        rows.append((f"slope at beta=0, D={D}", lin, slope))
        rows.append((f"value at beta=0, D={D}", saddle_node_rate_linear(0.0, D), _exact_rate(0.0, D)))
    return relative(rows, 1e-7, ref="exact rate (K0004), central difference")


def _check_printed():
    return relative([("0.201 (printed)", round(_C0, 3), 0.201), ("0.147 (printed)", round(_C1, 3), 0.147)],
                    1e-12, ref="Lindner et al., Eq. (3.11)")


register(Entry(
    id="K0007",
    slug="saddle-node-linear-response",
    title="Rate near a saddle-node to first order in the bias",
    area="neurons",
    kind="asymptotic",
    statement=(
        "For weak bias (or strong noise), the rate of passage for dx/dt = beta + x^2 + "
        "sqrt(2D) xi is 0.201 D^(1/3) + 0.147 D^(-1/3) beta, with both coefficients known in "
        "closed form."
    ),
    model=r"\dot x = \beta + x^2 + \sqrt{2D}\,\xi(t),\ |\beta| \ll D^{2/3}",
    formula=(r"r \approx \frac{(3D)^{1/3}}{[\Gamma(1/3)]^2} + \frac{9}{8}\frac{3^{1/6}}{\pi^3}"
             r"[\Gamma(2/3)]^4 D^{-1/3}\beta \approx 0.201\,D^{1/3} + 0.147\,D^{-1/3}\beta"),
    function=saddle_node_rate_linear,
    parameters={"beta": "bias", "D": "noise intensity"},
    sources=(
        Source(
            citation="B. Lindner, A. Longtin and A. Bulsara, Neural Comput. 15, 1761 (2003)",
            doi="10.1162/08997660360675035",
            locator="Eq. (3.11), p. 1772",
            version_read="author's PDF, page image read 12 Oct 2026",
            printed=(r"r(\beta\ll1,D)\approx\frac{(3D)^{1/3}}{[\Gamma(1/3)]^2}+\frac98\frac{3^{1/6}}{\pi^3}"
                     r"[\Gamma(2/3)]^4D^{-1/3}\beta\approx0.201D^{1/3}+0.147D^{-1/3}\beta"),
            notation="same symbols",
        ),
    ),
    validity="|beta| small compared with D^(2/3). First order in beta; exact value at beta = 0.",
    checks=(
        Check("value and slope at beta = 0 against the exact rate", "exact rate in Airy functions (K0004)", _check_slope),
        Check("the printed decimals", "Lindner et al. (2003), Eq. (3.11)", _check_printed),
    ),
    aliases=("theta neuron gain at rheobase", "quadratic integrate-and-fire neuron, weak input"),
    related=("K0004", "K0005"),
    added="2026-10-14",
))
