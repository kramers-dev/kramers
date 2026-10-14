"""Small helpers that turn comparisons into CheckResults with a readable table."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .registry import CheckResult

Row = tuple[str, float, float]


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    cells = [[str(h) for h in header]] + [[f"{c:.12g}" if isinstance(c, float) else str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    return "\n".join("  ".join(c.rjust(w) for c, w in zip(r, widths, strict=True)) for r in cells)


def relative(rows: Sequence[Row], tol: float, what: str = "entry", ref: str = "reference") -> CheckResult:
    """Pass when |entry/reference - 1| <= tol for every row."""
    errs = [abs(v / r - 1.0) if r != 0 else abs(v) for _, v, r in rows]
    worst = float(max(errs))
    detail = _table(["case", what, ref, "rel. diff"], [(c, float(v), float(r), f"{e:.1e}") for (c, v, r), e in zip(rows, errs, strict=True)])
    return CheckResult(passed=bool(worst <= tol and np.all(np.isfinite(errs))), worst=worst, tolerance=tol, detail=detail)


def absolute(rows: Sequence[Row], tol: float, what: str = "entry", ref: str = "reference") -> CheckResult:
    """Pass when |entry - reference| <= tol for every row."""
    errs = [abs(v - r) for _, v, r in rows]
    worst = float(max(errs))
    detail = _table(["case", what, ref, "abs. diff"], [(c, float(v), float(r), f"{e:.1e}") for (c, v, r), e in zip(rows, errs, strict=True)])
    return CheckResult(passed=bool(worst <= tol and np.all(np.isfinite(errs))), worst=worst, tolerance=tol, detail=detail)


def statistical(rows: Sequence[tuple[str, float, float, float]], n_sigma: float = 4.0, what: str = "entry",
                ref: str = "simulation") -> CheckResult:
    """Pass when every entry value lies within n_sigma standard errors of the simulation."""
    z = [abs(v - r) / s if s > 0 else (0.0 if v == r else np.inf) for _, v, r, s in rows]
    worst = float(max(z))
    detail = _table(["case", what, ref, "std. error", "deviation / s.e."],
                    [(c, float(v), float(r), float(s), f"{zz:.2f}") for (c, v, r, s), zz in zip(rows, z, strict=True)])
    return CheckResult(passed=bool(worst <= n_sigma), worst=worst, tolerance=n_sigma, detail=detail)


def combine(*results: CheckResult) -> CheckResult:
    """All must pass; the worst is reported relative to its own tolerance."""
    ratio = max(r.worst / r.tolerance for r in results)
    return CheckResult(passed=all(r.passed for r in results), worst=ratio, tolerance=1.0,
                       detail="\n\n".join(r.detail for r in results) + "\n(worst shown as a fraction of each part's tolerance)")
