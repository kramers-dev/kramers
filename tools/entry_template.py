"""K0NNN. <One line: the result, in words a physicist would search for.>

Copy this file to src/kramers/handbook/k0NNN_short_name.py with the next free number,
fill in every field, and delete these instructions. See CONTRIBUTING.md.
"""

from __future__ import annotations

from .._checking import relative  # also: absolute, statistical, combine
from ..registry import Check, Entry, Source, register
from ..result import OutsideDomain
from ..solvers.quadrature import FirstPassage1D  # or fokker_planck, simulate, laplace


def my_result(x0: float, D: float) -> float:
    """What it returns, for which model, in one or two sentences, with the formula.

    Say which convention for D is used (here <x^2> = 2 D t) and what raises.
    """
    if not D > 0:
        raise OutsideDomain("D must be positive")
    return float(x0 * x0 / (2.0 * D))  # replace with the printed formula


def _check_against_solver():
    # Compare with something that does not share the derivation. Explain the tolerance.
    rows = []
    for x0, D in ((0.5, 1.0), (1.0, 0.3)):
        fp = FirstPassage1D(lambda x: 0 * x, D=D, domain=(0.0, 2.0), left="reflecting", right="absorbing")
        rows.append((f"x0={x0}, D={D}", my_result(x0, D), float(fp.mean_time(x0).value)))
    return relative(rows, 1e-10, ref="quadrature solver")


register(Entry(
    id="K0NNN",
    slug="short-name",
    title="The result, as a title",
    area="targets",  # escape | driven | targets | resetting | active | transport | neurons
    kind="exact",  # exact | asymptotic | estimator | computed-reference
    statement="Two or three sentences a newcomer can read.",
    model=r"LaTeX of the model",
    formula=r"LaTeX of the result in this entry's notation",
    function=my_result,
    parameters={"x0": "meaning", "D": "meaning"},
    sources=(
        Source(
            citation="A. Author and B. Author, Journal 1, 1 (2000)",
            doi="10.xxxx/xxxxx",  # checked at https://api.crossref.org/works/<doi>
            locator="Eq. (12), p. 3",
            version_read="arXiv:xxxx.xxxxxv2 HTML, opened on <date>",
            printed=r"the equation exactly as printed, in the paper's symbols",
            notation="how their symbols map onto the function's arguments",
        ),
    ),
    validity="Assumptions, limits, and for approximations the error where it was checked.",
    checks=(Check("against the quadrature solver", "Green's-function quadrature", _check_against_solver),),
    aliases=("other names people search for",),
    related=(),
    written_by=("Your Name",),
    added="YYYY-MM-DD",
))
