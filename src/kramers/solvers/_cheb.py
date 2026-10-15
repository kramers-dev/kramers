"""Chebyshev panels: the integration machinery behind the quadrature solver.

A function on [a, b] is represented by its values at the Chebyshev-Lobatto points of
each of a set of panels. Integrals of positive functions that span many orders of
magnitude (exp(U/kT) with barriers of hundreds of kT) are accumulated in logarithms,
panel by panel, so that nothing overflows and small contributions keep their relative
accuracy.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.polynomial import chebyshev as cheb
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Reference:
    """Nodes on [-1, 1] and the linear maps that act on values at those nodes."""

    s: Array  # nodes, ascending
    cum: Array  # (cum @ v)[j] = integral from -1 to s[j] of the interpolant of v
    cum_rev: Array  # (cum_rev @ v)[j] = integral from s[j] to 1
    weights: Array  # integral from -1 to 1
    bary: Array  # barycentric interpolation weights
    to_coeffs: Array  # values -> Chebyshev coefficients


def _build(s: Array, bary: Array) -> Reference:
    n = s.size
    vander = cheb.chebvander(s, n - 1)
    to_coeffs = np.linalg.inv(vander)
    integrated = np.empty((n, n))
    for m in range(n):
        c = np.zeros(n)
        c[m] = 1.0
        integrated[:, m] = cheb.chebval(s, cheb.chebint(c, lbnd=-1))
    cum = integrated @ to_coeffs
    total = np.empty(n)
    for m in range(n):
        c = np.zeros(n)
        c[m] = 1.0
        total[m] = cheb.chebval(1.0, cheb.chebint(c, lbnd=-1))
    weights = total @ to_coeffs
    # integral from s_j to 1 = total - cumulative; built from the mirror image so that
    # no subtraction happens when it is applied
    cum_rev = cum[::-1, ::-1].copy() if np.allclose(s, -s[::-1]) else weights[None, :] - cum
    return Reference(s=s, cum=cum, cum_rev=cum_rev, weights=weights, bary=bary, to_coeffs=to_coeffs)


@lru_cache(maxsize=16)
def lobatto(n: int) -> Reference:
    """Chebyshev-Lobatto points, endpoints included."""
    k = np.arange(n)
    s = -np.cos(np.pi * k / (n - 1))
    s[n // 2] = 0.0 if n % 2 == 1 else s[n // 2]
    bary = (-1.0) ** k
    bary[0] *= 0.5
    bary[-1] *= 0.5
    return _build(s, bary)


@lru_cache(maxsize=16)
def gauss(n: int) -> Reference:
    """Chebyshev points of the first kind, endpoints excluded."""
    k = np.arange(n)
    theta = (2 * k + 1) * np.pi / (2 * n)
    s = -np.cos(theta)
    bary = (-1.0) ** k * np.sin(theta)
    return _build(s, bary)


@dataclass(frozen=True)
class Panels:
    """Breakpoints and node coordinates of a panel decomposition of [a, b]."""

    edges: Array  # (P + 1,)
    n: int

    @property
    def ref(self) -> Reference:
        return lobatto(self.n)

    @property
    def half(self) -> Array:
        return (np.diff(self.edges) / 2.0)[:, None]

    @property
    def x(self) -> Array:
        mid = (self.edges[:-1] + self.edges[1:]) / 2.0
        return mid[:, None] + self.half * self.ref.s[None, :]

    def with_order(self, n: int) -> Panels:
        return Panels(self.edges, n)

    def bisected(self) -> Panels:
        """Every panel split in two (for an independent error estimate)."""
        mids = 0.5 * (self.edges[:-1] + self.edges[1:])
        return Panels(np.sort(np.concatenate([self.edges, mids])), self.n)

    def locate(self, points: Array) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Panel and node index of points that are panel edges (exactly)."""
        points = np.atleast_1d(np.asarray(points, dtype=float))
        idx = np.searchsorted(self.edges, points)
        p_idx = np.empty(points.size, dtype=np.intp)
        j_idx = np.empty(points.size, dtype=np.intp)
        for i, (pt, k) in enumerate(zip(points, idx, strict=True)):
            if k < self.edges.size and self.edges[k] == pt:
                if k < self.edges.size - 1:
                    p_idx[i], j_idx[i] = k, 0
                else:
                    p_idx[i], j_idx[i] = k - 1, self.n - 1
            else:
                raise ValueError(f"{pt} is not a panel edge")
        return p_idx, j_idx


def build_panels(
    phi: Callable[[Array], Array] | Sequence[Callable[[Array], Array]],
    a: float,
    b: float,
    breakpoints: Sequence[float] = (),
    n: int = 17,
    max_range: float = 3.0,
    coeff_tol: float = 1e-12,
    max_panels: int = 200_000,
    min_panels: int = 32,
    probe: int = 8192,
) -> Panels:
    """Split [a, b] until each log-weight varies by at most ``max_range`` and is resolved on each panel.

    ``phi`` is the logarithm of a weight to be integrated (or a sequence of them, all
    resolved on the same panels), so ``max_range`` bounds the dynamic range of each
    weight within one panel. The search starts from ``min_panels`` equal panels, with
    an extra edge at every local extremum found on ``probe`` equally spaced points. A
    feature narrower than that spacing, (b - a) / probe, can still be missed: put a
    breakpoint at it.
    """
    funcs = list(phi) if isinstance(phi, Sequence) else [phi]
    ref = lobatto(n)
    a, b = float(a), float(b)
    uniform = np.linspace(a, b, max(1, int(min_panels)) + 1)[1:-1]
    extrema: list[float] = []
    if probe > 2:
        xs = np.linspace(a, b, int(probe) + 1)
        for func in funcs:
            with np.errstate(all="ignore"):
                fs = np.asarray(func(xs), dtype=float)
            if np.all(np.isfinite(fs)):
                # each panel spans at most max_range of variation: a lower bound on the count
                if np.sum(np.abs(np.diff(fs))) / max_range > max_panels:
                    raise RuntimeError("panel refinement did not converge; the log-weight varies too fast")
                d = np.sign(np.diff(fs))
                turn = np.flatnonzero(d[1:] * d[:-1] < 0) + 1  # sign change of the slope
                if turn.size <= int(probe) // 8:  # a smooth function, not rounding noise
                    extrema += xs[turn].tolist()
    cuts = sorted({a, b, *(float(x) for x in breakpoints if a < x < b), *uniform.tolist(), *extrema})
    stack = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
    done: list[tuple[float, float]] = []
    while stack:
        lo, hi = stack.pop()
        x = (lo + hi) / 2.0 + (hi - lo) / 2.0 * ref.s
        split = False
        for func in funcs:
            f = np.asarray(func(x), dtype=float)
            if not np.all(np.isfinite(f)):
                raise ValueError(f"log-weight is not finite on [{lo}, {hi}]")
            coeffs = ref.to_coeffs @ f
            # rounding floor of the samples: the function itself, and the rounding of x times the slope
            slope = np.ptp(f) / (hi - lo)
            noise = 8.0 * np.finfo(float).eps * (np.max(np.abs(f)) + slope * max(abs(lo), abs(hi)))
            unresolved = np.max(np.abs(coeffs[-3:])) > max(coeff_tol * max(1.0, np.max(np.abs(f)) * 1e-3), noise)
            if unresolved or np.ptp(f) > max_range:
                split = True
                break
        tiny = (hi - lo) <= 1e-13 * max(1.0, abs(lo), abs(hi))
        if split and not tiny:
            mid = (lo + hi) / 2.0
            stack.append((mid, hi))
            stack.append((lo, mid))
        else:
            done.append((lo, hi))
        if len(done) + len(stack) > max_panels:
            raise RuntimeError("panel refinement did not converge; the log-weight varies too fast")
    done.sort()
    edges = np.array([done[0][0]] + [hi for _, hi in done])
    return Panels(edges=edges, n=n)


def log_cumulative(panels: Panels, logg: Array, reverse: bool = False) -> Array:
    """log of the integral of exp(logg) from a to each node (or from each node to b).

    ``logg`` has shape (P, n) and may contain -inf (a zero integrand).
    """
    ref = panels.ref
    shift = np.max(logg, axis=1, keepdims=True)
    empty = ~np.isfinite(shift[:, 0])
    shift = np.where(np.isfinite(shift), shift, 0.0)
    with np.errstate(under="ignore"):
        v = np.exp(logg - shift)
    mat = ref.cum_rev if reverse else ref.cum
    within = panels.half * (v @ mat.T)
    np.maximum(within, 0.0, out=within)
    with np.errstate(divide="ignore"):
        logwithin = np.log(within) + shift
    logwithin[empty] = -np.inf
    with np.errstate(invalid="ignore"):
        if not reverse:
            totals = logwithin[:, -1]
            acc = np.logaddexp.accumulate(totals)
            before = np.concatenate([[-np.inf], acc[:-1]])
            out = np.logaddexp(before[:, None], logwithin)
        else:
            totals = logwithin[:, 0]
            acc = np.logaddexp.accumulate(totals[::-1])[::-1]
            after = np.concatenate([acc[1:], [-np.inf]])
            out = np.logaddexp(after[:, None], logwithin)
    return out


def log_integral(panels: Panels, logg: Array) -> float:
    """log of the integral of exp(logg) over [a, b]."""
    ref = panels.ref
    shift = np.max(logg, axis=1)
    finite = np.isfinite(shift)
    if not np.any(finite):
        return -np.inf
    with np.errstate(under="ignore"):
        v = np.exp(logg[finite] - shift[finite, None])
    parts = panels.half[finite, 0] * (v @ ref.weights)
    with np.errstate(divide="ignore"):
        logs = np.log(np.maximum(parts, 0.0)) + shift[finite]
    return float(np.logaddexp.reduce(logs))


def interpolate(panels: Panels, values: Array, x: Array) -> Array:
    """Evaluate the panel-wise Chebyshev interpolant of ``values`` (shape (P, n)) at x."""
    ref = panels.ref
    x = np.atleast_1d(np.asarray(x, dtype=float))
    p = np.clip(np.searchsorted(panels.edges, x, side="right") - 1, 0, panels.edges.size - 2)
    lo, hi = panels.edges[p], panels.edges[p + 1]
    s = (2.0 * x - lo - hi) / (hi - lo)
    out = np.empty_like(x)
    for i in range(x.size):
        diff = s[i] - ref.s
        hit = np.flatnonzero(diff == 0.0)
        if hit.size:
            out[i] = values[p[i], hit[0]]
            continue
        terms = ref.bary / diff
        out[i] = np.dot(terms, values[p[i]]) / np.sum(terms)
    return out
