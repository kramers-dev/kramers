"""K0011. Variance of the passage time through a noisy saddle-node, at any bias and thresholds."""

from __future__ import annotations

import numpy as np

from .._checking import relative
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from ..solvers import _cheb
from ..solvers.quadrature import FirstPassage1D


def saddle_node_passage_variance(beta: float, D: float, x_minus: float = -np.inf, x_plus: float = np.inf) -> float:
    """Variance of the time from reset x_minus to threshold x_plus for dx/dt = beta + x^2 + sqrt(2D) xi.

    This is the quantity of Lindner, Longtin and Bulsara's Eq. (A.4) (and of Eq. (3.2)
    for infinite thresholds, whose printed prefactor must be doubled; see the notes).
    Computed with the quadrature solver's moment recursion.
    """
    if not D > 0:
        raise OutsideDomain("D must be positive")
    if not x_minus < x_plus:
        raise OutsideDomain("need x_minus < x_plus")
    logpsi = lambda x: (x**3 / 3.0 + beta * x) / D  # noqa: E731  (phi = -V/D, V = -x^3/3 - beta x)
    dlogpsi = lambda x: (x * x + beta) / D  # noqa: E731
    if np.isfinite(x_plus):
        fp = FirstPassage1D(log_psi=logpsi, D=D, domain=(-np.inf, x_plus), left="natural", right="absorbing")
    else:
        fp = FirstPassage1D(log_psi=logpsi, D=D, left="natural", right="exit", dlog_psi=dlogpsi)
    m = fp.moments(x_minus, n=2).value
    return float(m[1] - m[0] ** 2)


def _a4_direct(beta: float, D: float, xm: float, xp: float, lo: float = -9.0) -> float:
    """Lindner et al. Eq. (A.4), evaluated directly as a triple integral in log space."""
    V = lambda x: -(x**3) / 3.0 - beta * x  # noqa: E731
    P = _cheb.build_panels(lambda x: -V(x) / D, lo, xp, breakpoints=[xm], n=25)
    x = P.x
    inner = _cheb.log_cumulative(P, -V(x) / D)  # log int_{-inf}^z e^{-V/D}
    # int_z^{x+} Theta(x - x-) e^{V/D}: from max(z, x-) to x+. The step sits on a panel edge,
    # so the integrand is set to zero on every panel left of x- (all of its nodes included)
    right_of_reset = (P.edges[:-1] >= xm)[:, None] & np.ones_like(x, dtype=bool)
    tail = _cheb.log_cumulative(P, np.where(right_of_reset, V(x) / D, -np.inf), reverse=True)
    total = _cheb.log_integral(P, V(x) / D + 2.0 * inner + tail)
    return float(2.0 / D**2 * np.exp(total))


def _check_appendix_formula():
    rows = []
    for beta, D, xm, xp in ((-0.4, 0.5, -1.5, 2.0), (0.3, 1.0, -2.0, 1.5), (-1.0, 0.3, -1.2, 3.0)):
        rows.append((f"beta={beta}, D={D}, x-={xm}, x+={xp}", saddle_node_passage_variance(beta, D, xm, xp),
                     _a4_direct(beta, D, xm, xp)))
    return relative(rows, 1e-9, ref="Eq. (A.4) evaluated directly")


def _check_threshold_value():
    rows = []
    for D in (0.2, 1.0):
        T = 2.6789385347077476 ** 2 * (1.0 / (3.0 * D)) ** (1.0 / 3.0)  # Gamma(1/3)^2 (3D)^(-1/3)
        rows.append((f"beta=0, D={D}", saddle_node_passage_variance(0.0, D), T**2 / 3.0))
    return relative(rows, 1e-11, ref="Lindner et al. Eq. (3.5): <T>^2 / 3")


register(Entry(
    id="K0011",
    slug="saddle-node-passage-variance",
    title="Variance of the passage time through a noisy saddle-node, any bias and thresholds",
    area="neurons",
    kind="exact",
    statement=(
        "The variance of the interspike interval of the quadratic integrate-and-fire neuron "
        "with white noise, for any bias and any reset and threshold, is a triple integral "
        "of exp(+-V/D) (Lindner, Longtin and Bulsara, Eq. A.4). Together with the mean it "
        "gives the coefficient of variation across the whole firing range."
    ),
    model=r"\dot x = \beta + x^2 + \sqrt{2D}\,\xi(t),\ \text{reset } x_-,\ \text{threshold } x_+,\ V=-x^3/3-\beta x",
    formula=(r"\langle\Delta T^2\rangle=\frac{2}{D^2}\int_{-\infty}^{x_+}dz\,e^{V(z)/D}\left[\int_{-\infty}^{z}dy\,e^{-V(y)/D}\right]^2"
             r"\int_{z}^{x_+}dx\,\Theta(x-x_-)\,e^{V(x)/D}"),
    function=saddle_node_passage_variance,
    parameters={"beta": "bias", "D": "noise intensity", "x_minus": "reset (default -inf)", "x_plus": "threshold (default +inf)"},
    sources=(
        Source(
            citation="B. Lindner, A. Longtin and A. Bulsara, Neural Comput. 15, 1761 (2003)",
            doi="10.1162/08997660360675035",
            locator="Eq. (A.4), p. 1785; Eq. (3.2), p. 1770 (infinite thresholds)",
            version_read="author's PDF (journal typesetting), pages read 12 and 14 Oct 2026",
            printed=(r"\langle\Delta T^2\rangle=\frac{2}{D^2}\int_{-\infty}^{x_+}dz\,e^{V(z)/D}\left[\int_{-\infty}^{z}dy\,"
                     r"e^{-V(y)/D}\right]^2\times\int_{z}^{x_+}dx\,\Theta(x-x_-)\,e^{V(x)/D}"),
            notation="same symbols; V(x) = -x^3/3 - beta x (their Eq. 2.6)",
        ),
    ),
    validity="White noise, any bias, any thresholds (infinite ones included). Exact.",
    checks=(
        Check("against Eq. (A.4) evaluated directly", "independent triple integral in log space", _check_appendix_formula),
        Check("at the bifurcation point", "Lindner et al. Eq. (3.5)", _check_threshold_value),
    ),
    aliases=("CV of the theta neuron", "interspike-interval variance", "quadratic integrate-and-fire"),
    related=("K0004", "K0005", "K0009"),
    notes=(
        "Questions for the authors (Lindner, Longtin and Bulsara), both from computations. "
        "(1) Eq. (3.2) as printed has the prefactor (9/D)^(2/3); with it, Eq. (3.2) gives exactly "
        "half of the paper's own Eq. (3.5) at beta = 0 (computed ratio 2.0000 at beta = 0, -0.5 "
        "and 0.8), and with 2 (9/D)^(2/3) the two agree. (2) In Eq. (A.3) as printed the middle "
        "integrand is exp(-V(z)/D); for a linear potential V = -mu x that gives 8.3e3 instead "
        "of the exact 2 D L / mu^3 = 0.728 (mu = 1.3, D = 0.4, L = 2), while exp(+V(z)/D) "
        "reproduces it. Eqs. (A.1), (A.2) and (A.4) are consistent with each other and with this "
        "entry. We read both as misprints but may have misread the notation. The bare prefactor "
        "of Eq. (3.2) reappears in Vilela and Lindner, J. Theor. Biol. 257, 90 (2009), Eq. (38) of "
        "the arXiv version (journal version not checked). No erratum or comment was found in "
        "Crossref, PubMed or an OpenAlex search of citing works (15 Oct 2026)."
    ),
    added="2026-10-14",
))
