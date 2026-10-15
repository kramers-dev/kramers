"""Exact first-passage statistics of one-dimensional diffusions, by quadrature.

Model (Ito)::

    dX = A(X) dt + sqrt(2 D(X)) dW,        a < X < b.

With the scale density ``1/psi`` and the speed density ``w = psi / D``, where
``psi = exp(phi)`` and ``phi' = A / D``, the generator is ``L u = (D / psi) (psi u')'``.
For a potential ``U`` in units of kT and constant ``D``, ``A = -D U'`` and ``phi = -U``.

Every quantity here is the solution ``u`` of ``L u = -f`` with ``u = 0`` at absorbing
ends and ``u' = 0`` at reflecting ends, written with the Green's function of that
problem (published: the Pontryagin-Andronov-Vitt quadratures; see Haenggi, Talkner and
Borkovec, Rev. Mod. Phys. 62, 251 (1990), and Gardiner, Handbook of Stochastic
Methods). With ``f = 1`` it is the mean first-passage time; with ``f = n T_{n-1}`` the
n-th moment; with ``f = pi_b`` the time restricted to exits at ``b``.

Numerics (computed checks in ``tests/test_quadrature.py``): Chebyshev panels on which
exp(phi) varies by at most a factor e^3, integrals of positive functions accumulated in
logarithms, so barriers of hundreds of kT neither overflow nor lose relative accuracy.
Every result is computed twice, the second time on bisected panels at a higher order,
and the difference is returned as its error.

Known limits (also in the README). The panels are found by sampling phi (and log D);
a feature narrower than the probe spacing, 1/8192 of the domain, can be missed, and both
computations of the error estimate would then miss it too: give ``breakpoints`` at
narrow features. Infinite ends are cut by marching outwards; the region beyond each cut
is scanned geometrically out to 1e8 times the cut's distance, and a natural cut is moved
past any rebound found, while an exit end with structure beyond its cut is refused. A
feature narrower than about 0.1% of its distance from the cut can escape the scan.

Boundaries
----------
``"absorbing"``, ``"reflecting"``: finite ends.
``"natural"``: an infinite end the process cannot reach (the speed density exp(phi) / D
    vanishes there); the domain is cut where it has fallen by e^-50 below its largest
    value, beyond every rebound above e^-35 found by the scan, and the cut is made
    reflecting.
``"exit"``: an infinite end reached in finite time (an explosive drift, as for the
    noisy saddle-node). The domain is cut at a point ``b0`` far past every barrier, the
    time from ``b0`` to infinity is added by the strong Markov property, and that time
    is computed from local Laplace-type integrals on a compactified tail. The cut is
    placed where phi has risen 45 e-folds above its smallest value; the omitted part of
    the Green's operator is then of relative order exp(-45) (first order in that factor).
    Exit and entrance ends need a constant D.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import roots_legendre

from ..result import OutsideDomain, Result
from . import _cheb

Array = NDArray[np.float64]
Func = Callable[[Array], Array]

__all__ = ["FirstPassage1D"]

_FINITE = {"absorbing", "reflecting"}
_INFINITE = {"natural", "exit"}
_DROP = 50.0  # e-folds by which a truncated weight has fallen
_RETURN = 45.0  # e-folds that make a return from beyond b0 negligible
_REGAIN = 15.0  # e-folds a natural tail may climb back beyond the cut (to e^-35 of the peak)


def _numerical_derivative(f: Func) -> Func:
    """Derivative of a vectorised function: complex step if it accepts complex input."""
    probe = np.array([0.37, 1.3])
    try:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = np.asarray(f(probe + 1e-30j))
        complex_ok = np.iscomplexobj(val) and np.all(np.isfinite(val))
    except Exception:  # noqa: BLE001 - any failure means: no complex step
        complex_ok = False
    if complex_ok:
        def d(x: Array) -> Array:
            x = np.asarray(x, dtype=float)
            return np.imag(f(x + 1e-30j)) / 1e-30
        return d

    def d_fd(x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        h = 1e-3 * np.maximum(1.0, np.abs(x))
        return (8 * (f(x + h) - f(x - h)) - (f(x + 2 * h) - f(x - 2 * h))) / (12 * h)

    return d_fd


@dataclass(frozen=True)
class _Grid:
    panels: _cheb.Panels
    phi: Array
    logD: Array
    logS_fwd: Array  # log of integral of 1/psi from a to x
    logS_rev: Array  # log of integral of 1/psi from x to b

    @property
    def logw(self) -> Array:
        return self.phi - self.logD

    @property
    def logS_tot(self) -> float:
        return float(self.logS_fwd[-1, -1])


def _log_floor(value: Array) -> Array:
    """A value obtained as exp(log v) is known only to a few units in the last place of log v."""
    v = np.abs(np.asarray(value, dtype=float))
    with np.errstate(divide="ignore"):
        lg = np.abs(np.log(v))
    return 16.0 * np.finfo(float).eps * np.maximum(1.0, np.where(np.isfinite(lg), lg, 0.0)) * v


class FirstPassage1D:
    """First-passage problem for a one-dimensional diffusion.

    Parameters
    ----------
    potential
        U(x) in units of kT, with constant ``D``: drift ``-D U'(x)``.
    D
        Diffusion coefficient: a number, or a function of x (then give ``log_psi``).
    log_psi
        phi(x) with ``phi' = A / D``, instead of ``potential``.
    drift
        A(x), instead of ``potential`` or ``log_psi``; finite domains only.
    dlog_psi
        phi'(x). Only used for an ``"exit"`` boundary; computed if not given.
    domain
        (a, b); ends may be infinite.
    left, right
        Boundary types (see the module docstring). At least one must absorb.
    breakpoints
        Points where the panels must have an edge (wells, barrier tops); optional.
    """

    def __init__(
        self,
        potential: Func | None = None,
        *,
        D: float | Func = 1.0,
        log_psi: Func | None = None,
        drift: Func | None = None,
        dlog_psi: Func | None = None,
        domain: tuple[float, float] = (-np.inf, np.inf),
        left: str,
        right: str,
        breakpoints: Sequence[float] = (),
        order: int = 17,
    ) -> None:
        a, b = float(domain[0]), float(domain[1])
        if not a < b:
            raise OutsideDomain("the domain must have a < b")
        for name, side, end in (("left", left, a), ("right", right, b)):
            allowed = _FINITE if np.isfinite(end) else _INFINITE
            if side not in allowed:
                raise OutsideDomain(f"{name} boundary at {end}: choose one of {sorted(allowed)}")
        if left not in {"absorbing", "exit"} and right not in {"absorbing", "exit"}:
            raise OutsideDomain("at least one boundary must absorb (absorbing or exit)")
        if left == "exit" and right == "exit":
            raise OutsideDomain("exit at both ends is not supported")
        given = sum(x is not None for x in (potential, log_psi, drift))
        if given != 1:
            raise OutsideDomain("give exactly one of potential, log_psi, drift")
        if potential is not None and callable(D):
            raise OutsideDomain("with a potential, D must be constant; use log_psi for D(x)")
        if drift is not None and not (np.isfinite(a) and np.isfinite(b)):
            raise OutsideDomain("a drift alone defines the problem on a finite domain only")

        self._mirror = left == "exit"
        sign = -1.0 if self._mirror else 1.0
        self._order = order

        def _D(x: Array) -> Array:
            x = np.asarray(x, dtype=float)
            return np.broadcast_to(np.asarray(D(sign * x) if callable(D) else D, dtype=float), x.shape)

        self._D = _D
        self._variable_D = callable(D)
        if potential is not None:
            pot = potential
            self._phi_raw: Func = lambda x: -np.asarray(pot(sign * np.asarray(x)))
        elif log_psi is not None:
            lp = log_psi
            self._phi_raw = lambda x: np.asarray(lp(sign * np.asarray(x)))
        else:
            self._phi_raw = self._phi_from_drift(drift, D, a, b, sign)  # type: ignore[arg-type]
        if dlog_psi is not None:
            dlp = dlog_psi
            self._dphi: Func = lambda x: sign * np.asarray(dlp(sign * np.asarray(x, dtype=float)), dtype=float)
        else:
            self._dphi = _numerical_derivative(self._phi_raw)
            self._dphi_numerical = True

        if self._mirror:
            a, b = -b, -a
            left, right = right, "exit"
        self.a, self.b = a, b
        self.left, self.right = left, right
        self._sign = sign
        self._user_breaks = [sign * float(x) for x in breakpoints]

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _phi_from_drift(drift: Func, D: float | Func, a: float, b: float, sign: float) -> Func:
        """phi = integral of A / D from the midpoint, as a Chebyshev interpolant."""
        def ratio(x: Array) -> Array:
            xs = sign * np.asarray(x, dtype=float)
            Dx = D(xs) if callable(D) else D
            return sign * np.asarray(drift(xs), dtype=float) / Dx

        lo, hi = (a, b) if sign > 0 else (-b, -a)
        panels = _cheb.build_panels(lambda x: np.zeros_like(x), lo, hi, breakpoints=np.linspace(lo, hi, 9)[1:-1])
        # refine on the resolution of A / D itself
        ref = panels.ref
        edges = list(panels.edges)
        for _ in range(30):
            new = [edges[0]]
            changed = False
            for lo_p, hi_p in zip(edges[:-1], edges[1:], strict=True):
                xs = (lo_p + hi_p) / 2 + (hi_p - lo_p) / 2 * ref.s
                coeffs = ref.to_coeffs @ ratio(xs)
                if np.max(np.abs(coeffs[-3:])) > 1e-13 * max(1.0, np.max(np.abs(coeffs))):
                    new.append((lo_p + hi_p) / 2)
                    changed = True
                new.append(hi_p)
            edges = new
            if not changed:
                break
        panels = _cheb.Panels(np.asarray(edges), panels.n)
        vals = ratio(panels.x)
        within = panels.half * (vals @ ref.cum.T)
        totals = within[:, -1]
        before = np.concatenate([[0.0], np.cumsum(totals)[:-1]])
        cum = before[:, None] + within
        mid = 0.5 * (lo + hi)
        offset = _cheb.interpolate(panels, cum, np.array([mid]))[0]
        cum = cum - offset

        def phi(x: Array) -> Array:
            x = np.asarray(x, dtype=float)
            return _cheb.interpolate(panels, cum, x.ravel()).reshape(x.shape)

        return phi

    def _phi(self, x: Array) -> Array:
        return np.asarray(self._phi_raw(np.asarray(x, dtype=float)), dtype=float)

    def _tail_safe(self, x: float) -> bool:
        """phi' large and slowly varying at x, so local integrals there are accurate."""
        d1 = float(self._dphi(np.array([x]))[0])
        h = 1e-3 * max(1.0, abs(x))
        d2 = float((self._dphi(np.array([x + h])) - self._dphi(np.array([x - h])))[0] / (2 * h))
        return d1 > 0 and abs(d2) <= 0.02 * d1 * d1 and abs(x) * d1 >= 200.0

    def _logw(self, x: Array) -> Array:
        """log of the speed density psi / D (+inf where D underflows to 0)."""
        with np.errstate(divide="ignore"):
            return self._phi(x) - np.log(self._D(x))

    def _scan_beyond(self, cut: float, direction: float, slope: bool, ceiling: float = np.inf) -> float | None:
        """The farthest point beyond a cut where the tail breaks the cut's assumption, or None.

        ``slope``: phi' must stay positive (exit and entrance tails, whose time is
        computed from local integrals). Otherwise the log speed density, and the log of
        its mass per decade of distance, must stay below ``ceiling`` (natural ends: the
        weight dropped beyond the cut must stay negligible and integrable).
        The scan is geometric, from 1e-4 to 1e8 times the cut's distance from the origin,
        2,000 points per decade, and stops where the functions overflow: a feature
        narrower than about 0.1% of its distance from the cut can still escape it.
        """
        scale = max(1.0, abs(cut))
        t = np.concatenate([[0.0], scale * np.logspace(-4.0, 8.0, 24001)])
        xs = cut + direction * t
        with np.errstate(all="ignore"):
            vals = self._dphi(xs) if slope else self._logw(xs)
        finite = np.isfinite(vals)
        stop = int(np.argmin(finite)) if not np.all(finite) else vals.size
        xs, vals = xs[:stop], vals[:stop]
        if slope:
            bad = ~(vals > 0)
        else:
            # the weight must stay low pointwise, and its mass per decade of distance,
            # about w(t) (t + scale), must stay low too, or the tail is not integrable
            mass = vals + np.log(t[:stop] + scale) - np.log(scale)
            bad = (vals > ceiling) | (mass > ceiling)
        if not np.any(bad):
            return None
        return float(xs[np.flatnonzero(bad)[-1]])

    def _march(self, start: float, direction: float, *, exit: bool, entrance: bool = False,
               extreme: float = -np.inf) -> tuple[float, float]:
        """Walk outwards until the weight is negligible (natural) or the tail is safe (exit).

        Returns the cut and the extreme of the log-weight met on the way (the largest log
        speed density for a natural end, the smallest phi for an exit).
        """
        if self._variable_D and (exit or entrance):
            raise OutsideDomain("exit and entrance ends need a constant D; cut the domain at a finite point")
        weight = self._phi if exit else self._logw
        step = max(0.5, 0.25 * abs(start))
        x = start
        xs = np.linspace(start, start + direction * 1e-9, 2)
        if exit:
            extreme = float(np.min(weight(xs)))
        else:
            extreme = max(extreme, float(np.max(weight(xs))))
        for _ in range(400):
            x_new = x + direction * step
            seg = np.linspace(x, x_new, 257)
            vals = weight(seg)
            if not np.all(np.isfinite(vals)):
                raise OutsideDomain(f"log-weight is not finite near {x_new}")
            if not exit:
                extreme = max(extreme, float(np.max(vals)))
                outward_decreasing = vals[-1] < vals[-2]
                if vals[-1] < extreme - _DROP and outward_decreasing and (not entrance or self._tail_safe(x_new)):
                    return float(x_new), extreme
            else:
                extreme = min(extreme, float(np.min(vals)))
                if vals[-1] > extreme + _RETURN and self._tail_safe(x_new):
                    return float(x_new), extreme
            x = x_new
            step *= 1.5
        kind = "an exit (explosive)" if exit else "a natural"
        raise OutsideDomain(f"the boundary does not behave as {kind} boundary")

    def _natural_cut(self, start: float, direction: float, entrance: bool = False) -> float:
        """Cut for a natural end, moved outwards past every rebound of the weight the scan finds."""
        cut, extreme = self._march(start, direction, exit=False, entrance=entrance)
        for _ in range(100):
            if entrance:
                far = self._scan_beyond(cut, direction, slope=True)
                if far is not None:
                    raise OutsideDomain(f"beyond the entrance cut at {cut:.4g}, phi' turns negative (near {far:.4g}); "
                                        "a start at infinity needs a drift-dominated tail")
            far = self._scan_beyond(cut, direction, slope=False, ceiling=extreme - (_DROP - _REGAIN))
            if far is None:
                return cut
            # the weight comes back beyond the cut: continue the march from the farthest such point
            with np.errstate(divide="ignore"):
                extreme = max(extreme, float(self._logw(np.array([far]))[0]))
            cut, extreme = self._march(far, direction, exit=False, entrance=entrance, extreme=extreme)
        raise OutsideDomain("the weight keeps coming back beyond every cut; cut the domain at a finite point")

    def _exit_cut(self, start: float) -> float:
        cut, _ = self._march(start, +1.0, exit=True)
        far = self._scan_beyond(cut, +1.0, slope=True)
        if far is not None:
            raise OutsideDomain(f"beyond the cut at {cut:.4g}, phi' turns negative (near {far:.4g}): a well or barrier "
                                "lies there; cut the domain at a finite point beyond it, or give a breakpoint at it")
        return cut

    def _effective_domain(self, points: Array, entrance: bool = False) -> tuple[float, float, float]:
        """Finite cuts (a, b) of the domain and, for a start at -infinity, the entrance cut.

        A natural end is cut where the speed density has fallen by e^-50 below its
        largest value, and moved further out while a scan beyond the cut finds it
        climbing back above e^-35 of that value; a reflecting wall there then changes no
        result by more than that. For a start at -infinity the entrance tail begins at a
        cut ``a0`` and the domain itself is extended 50 more e-folds beyond it.
        """
        inner = np.concatenate([points, np.asarray(self._user_breaks, dtype=float)])
        lo = float(np.min(inner)) if inner.size else 0.0
        hi = float(np.max(inner)) if inner.size else 0.0
        a, b = self.a, self.b
        a0 = np.nan
        if self.left == "natural":
            if entrance:
                a0 = self._natural_cut(lo, -1.0, entrance=True)
                a = self._natural_cut(a0, -1.0)
            else:
                a = self._natural_cut(lo, -1.0)
        if self.right == "natural":
            b = self._natural_cut(hi, +1.0)
        if self.right == "exit":
            b = self._exit_cut(hi)
        return a, b, a0

    def _grid(self, a: float, b: float, points: Array, order: int) -> _Grid:
        breaks = [*points.tolist(), *self._user_breaks]
        # the integrands carry exp(-phi) and the speed density exp(phi) / D: resolve both
        weights = [self._phi, self._logw] if self._variable_D else [self._phi]
        try:
            panels = _cheb.build_panels(weights, a, b, breakpoints=breaks, n=self._order)
        except ValueError as err:
            raise OutsideDomain(f"{err}: D must be positive and finite, and phi finite, on the domain") from None
        except RuntimeError as err:
            raise OutsideDomain(f"{err} (more than 200,000 panels on [{a:.4g}, {b:.4g}])") from None
        if order > self._order:  # the check computation: finer panels and a higher order
            panels = panels.bisected()
        panels = panels.with_order(order)
        x = panels.x
        phi = self._phi(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            logD = np.log(self._D(x))
        if not (np.all(np.isfinite(logD)) and np.all(np.isfinite(phi))):
            raise OutsideDomain("D must be positive and finite, and phi finite, on the domain")
        logS_fwd = _cheb.log_cumulative(panels, -phi)
        logS_rev = _cheb.log_cumulative(panels, -phi, reverse=True)
        return _Grid(panels, phi, logD, logS_fwd, logS_rev)

    # ------------------------------------------------------------ green's function
    def _kind(self) -> str:
        left_abs = self.left == "absorbing"
        right_abs = self.right in {"absorbing", "exit"}
        return ("A" if left_abs else "R") + ("A" if right_abs else "R")

    def _green_log(self, g: _Grid, logf: Array) -> Array:
        """log u at every node, where L u = -f with the boundary conditions."""
        P = g.panels
        cum = _cheb.log_cumulative
        logw = g.logw
        kind = self._kind()
        with np.errstate(invalid="ignore"):
            if kind == "RA":
                t1 = g.logS_rev + cum(P, logw + logf)
                t2 = cum(P, g.logS_rev + logw + logf, reverse=True)
                return np.logaddexp(t1, t2)
            if kind == "AR":
                t1 = cum(P, g.logS_fwd + logw + logf)
                t2 = g.logS_fwd + cum(P, logw + logf, reverse=True)
                return np.logaddexp(t1, t2)
            t1 = g.logS_rev + cum(P, g.logS_fwd + logw + logf)
            t2 = g.logS_fwd + cum(P, g.logS_rev + logw + logf, reverse=True)
            return np.logaddexp(t1, t2) - g.logS_tot

    # ------------------------------------------------------------------ the tails
    @cached_property
    def _legendre_theta(self) -> tuple[Array, Array]:
        t, w = roots_legendre(10)
        return (t + 1.0) / 2.0, w / 2.0

    def _delta_phi(self, z: Array, t: Array, forward: bool) -> Array:
        """phi(z + t) - phi(z) (forward) or phi(z) - phi(z - t), with no cancellation."""
        theta, wt = self._legendre_theta
        sgn = 1.0 if forward else -1.0
        pts = z[..., None] + sgn * theta * t[..., None]
        dphi = self._dphi(pts.reshape(-1)).reshape(pts.shape)
        return t * np.sum(dphi * wt, axis=-1)

    def _local(self, z: Array, forward: bool, limit: float | None, weight: bool) -> Array:
        """Local Laplace-type integral at each z.

        forward:  integral over y in (z, limit) of exp(-(phi(y) - phi(z))) [g(y)] dy
        backward: integral over y in (limit, z) of exp(-(phi(z) - phi(y))) [g(y)] dy
        ``limit`` None means infinity; g = 1 / D when ``weight``. phi must increase
        through the range, which the choice of cut points guarantees.
        """
        d1 = self._dphi(z)
        if np.any(d1 <= 0):
            raise OutsideDomain("the tail is not drift-dominated: phi' must be positive there")
        span = np.inf if limit is None else np.abs(limit - z)
        tau_max = np.minimum(d1 * span, 60.0)
        gx, gw = roots_legendre(16)
        out = np.zeros_like(z)
        pieces = 8
        sgn = 1.0 if forward else -1.0
        for k in range(pieces):
            lo = tau_max * k / pieces
            hi = tau_max * (k + 1) / pieces
            tau = (lo + hi)[:, None] / 2 + (hi - lo)[:, None] / 2 * gx[None, :]
            t = tau / d1[:, None]
            zz = np.broadcast_to(z[:, None], t.shape)
            with np.errstate(under="ignore"):
                kernel = np.exp(-self._delta_phi(zz, t, forward=forward))
            if weight:
                y = (zz + sgn * t).reshape(-1)
                kernel = kernel / self._D(y).reshape(t.shape)
            out += (hi - lo) / 2 * np.sum(gw * kernel, axis=1)
        return out / d1

    def _growth(self, cut: float, direction: float) -> float:
        """Exponent p of the asymptotic growth phi'(z) ~ |z|^p, snapped to quarters.

        The tail is compactified with z - cut ~ s^(-1/(p-1)); with the right p the
        integrands are smooth in s and the quadrature converges spectrally.
        """
        scale = max(1.0, abs(cut))
        far = cut + direction * scale * np.array([1e3, 1e6])
        with np.errstate(over="ignore", invalid="ignore"):
            d = self._dphi(far)
        if not np.all(np.isfinite(d)) or np.any(d <= 0):
            return 5.0  # faster than any power: the largest compactification is used
        p = float(np.log(d[1] / d[0]) / np.log(abs(far[1] - cut) / abs(far[0] - cut)))
        snapped = round(p * 4.0) / 4.0
        return snapped if abs(p - snapped) < 0.02 else p

    def _tail_grid(self, cut: float, direction: float, ns: int) -> tuple[_cheb.Reference, Array, Array]:
        """Nodes z(s) on the compactified tail beyond ``cut`` and |dz/ds|; s=0 is infinity."""
        scale = max(1.0, abs(cut))
        p = self._growth(cut, direction)
        if p <= 1.0:
            raise OutsideDomain("the drift does not grow fast enough for the end to be reached in finite time")
        q = float(min(p - 1.0, 4.0))
        ref = _cheb.gauss(ns)
        s = (ref.s + 1.0) / 2.0
        z = cut + direction * scale * (s ** (-1.0 / q) - 1.0)
        jac = scale / q * s ** (-1.0 / q - 1.0)
        return ref, z, jac

    def _exit_tail(self, b0: float, ns: int, logR1: float, second: Callable[[float], float] | None) -> tuple[float, float]:
        """Mean and second moment of the time from b0 to +infinity.

        Exact (up to e^-45) Green's-function values at b0 for the whole problem:
        ``logR1`` is log of exp(-phi(b0)) times the integral of w (times the probability
        of reaching b0 first, when the left end absorbs) over the main domain; it carries
        the short excursions below b0. ``second(t1)`` returns the same quantity for the
        second moment once the mean beyond the cut, ``t1``, is known.
        """
        ref, z, jac = self._tail_grid(b0, +1.0, ns)
        Dz = self._D(z)
        K = self._local(z, forward=True, limit=None, weight=False)
        K0 = float(self._local(np.array([b0]), forward=True, limit=None, weight=False)[0])
        J = self._local(z, forward=False, limit=b0, weight=True)
        integrand = K / Dz * jac
        beyond = 0.5 * (ref.cum @ integrand)  # integral from z_i to infinity of K / D
        R1 = float(np.exp(logR1))
        t1 = K0 * R1 + 0.5 * float(ref.weights @ integrand)
        if second is None:
            return t1, np.nan
        with np.errstate(under="ignore", over="ignore"):
            decay = np.exp(-(self._phi(z) - self._phi(np.array([b0]))[0]))
        t1_nodes = R1 * decay * K + K * J + beyond
        R2 = second(t1)
        t2 = K0 * R2 + 0.5 * float(ref.weights @ (2.0 * K * t1_nodes / Dz * jac))
        return t1, t2

    def _entrance_tail(self, a0: float, ns: int) -> tuple[float, float]:
        """Mean and second moment of the time from -infinity to a0."""
        ref, z, jac = self._tail_grid(a0, -1.0, ns)
        Dz = self._D(z)
        Kt = self._local(z, forward=True, limit=a0, weight=False)
        Kb = self._local(z, forward=False, limit=None, weight=True)
        integrand = Kt / Dz * jac
        within = 0.5 * (ref.cum_rev @ integrand)  # integral from z_i to a0
        e1 = 0.5 * float(ref.weights @ integrand)
        te_nodes = Kt * Kb + within
        e2 = 0.5 * float(ref.weights @ (2.0 * Kt * te_nodes / Dz * jac))
        return e1, e2

    # -------------------------------------------------------------- public API
    def _points(self, x0: ArrayLike, allow_entrance: bool = False) -> Array:
        pts = np.atleast_1d(np.asarray(x0, dtype=float)) * self._sign
        entering = np.isneginf(pts)
        if np.any(entering) and not (allow_entrance and self.left == "natural"):
            raise OutsideDomain("a start at infinity needs an entrance boundary on that side")
        finite = pts[~entering]
        if np.any(finite < self.a) or np.any(finite > self.b) or np.any(np.isposinf(pts)):
            raise OutsideDomain("starting point outside the domain")
        return pts

    def _at(self, g: _Grid, values: Array, pts: Array) -> Array:
        p, j = g.panels.locate(pts)
        return values[p, j]

    def _moments_once(self, pts: Array, order: int, ns: int, nmom: int) -> tuple[Array, dict]:
        entering = np.isneginf(pts)
        finite = pts[~entering]
        a, b, a0 = self._effective_domain(finite, entrance=bool(np.any(entering)))
        evaluate = np.concatenate([finite, [a0]]) if np.any(entering) else finite
        g = self._grid(a, b, evaluate, order)
        kind = self._kind()
        logT = [np.zeros_like(g.phi)]
        for m in range(1, nmom + 1):
            logT.append(self._green_log(g, np.log(m) + logT[m - 1]))
        T = np.array([np.exp(self._at(g, logT[m], evaluate)) for m in range(1, nmom + 1)])
        info: dict = {"domain": (a, b), "panels": g.panels.edges.size - 1, "order": order}
        if self.right == "exit":
            if nmom > 2:
                raise OutsideDomain("with an exit boundary, moments up to the second only")
            P = g.panels
            cum = _cheb.log_cumulative
            end = (P.edges.size - 2, P.n - 1)  # the node at b0
            phi_b0 = g.phi[end]
            if kind == "RA":
                logpi = np.zeros_like(g.phi)
            else:
                logpi = g.logS_fwd - g.logS_tot  # probability of reaching b0 before a
            logR1 = float(cum(P, g.logw + logpi)[end] - phi_b0)

            def second(t1: float) -> float:
                T1_true = np.exp(logT[1]) + np.exp(logpi) * t1
                return float(np.exp(cum(P, g.logw + np.log(2.0 * T1_true) + logpi)[end] - phi_b0))

            t1, t2 = self._exit_tail(b, ns, logR1, second if nmom == 2 else None)
            info["exit_cut"] = b
            info["time_beyond_exit_cut"] = t1
            if kind == "RA":
                out = [T[0] + t1]
                if nmom == 2:
                    out.append(T[1] + 2.0 * T[0] * t1 + t2)
            else:
                p_b = np.exp(self._at(g, logpi, evaluate))
                out = [T[0] + p_b * t1]
                if nmom == 2:
                    restricted = np.exp(self._at(g, self._green_log(g, logpi), evaluate))
                    out.append(T[1] + 2.0 * restricted * t1 + p_b * t2)
            T = np.array(out)
        if np.any(entering):
            if nmom > 2:
                raise OutsideDomain("from an entrance at infinity, moments up to the second only")
            e1, e2 = self._entrance_tail(a0, ns)
            info["entrance_cut"] = a0
            info["time_before_entrance_cut"] = e1
            at_cut = T[:, -1]
            from_inf = [e1 + at_cut[0]]
            if nmom == 2:
                from_inf.append(e2 + 2.0 * e1 * at_cut[0] + at_cut[1])
            full = np.empty((nmom, pts.size))
            full[:, ~entering] = T[:, :-1]
            full[:, entering] = np.array(from_inf)[:, None]
            T = full
        return T, info

    def moments(self, x0: ArrayLike, n: int = 2) -> Result:
        """First ``n`` moments of the first-passage time from ``x0`` (array of shape (n, ...)).

        ``x0`` may be -inf (or +inf for a mirrored problem) when that end is an entrance
        boundary, as for the quadratic integrate-and-fire neuron reset to -infinity.
        """
        pts = self._points(x0, allow_entrance=True)
        lo, info_lo = self._moments_once(pts, self._order, 64, n)
        hi, info_hi = self._moments_once(pts, self._order + 8, 96, n)
        shape = (n,) + np.shape(x0)
        return Result(
            value=hi.reshape(shape),
            error=(np.abs(hi - lo) + _log_floor(hi)).reshape(shape),
            method="quadrature: Green's function on Chebyshev panels, log-space accumulation",
            details=info_hi,
        )

    def mean_time(self, x0: ArrayLike) -> Result:
        """Mean first-passage time from ``x0``."""
        r = self.moments(x0, n=1)
        return Result(np.asarray(r.value)[0], np.asarray(r.error)[0], r.method, r.details)

    def variance(self, x0: ArrayLike) -> Result:
        """Variance of the first-passage time from ``x0``."""
        r = self.moments(x0, n=2)
        v = np.asarray(r.value)
        e = np.asarray(r.error)
        var = v[1] - v[0] ** 2
        err = e[1] + 2 * np.abs(v[0]) * e[0]
        return Result(var, err, r.method, r.details)

    def _splitting_once(self, pts: Array, order: int) -> Array:
        a, b, _ = self._effective_domain(pts)
        g = self._grid(a, b, pts, order)
        return np.exp(self._at(g, g.logS_fwd, pts) - g.logS_tot)

    def splitting(self, x0: ArrayLike) -> Result:
        """Probability of leaving through the right end (both ends absorbing or exit)."""
        if self._kind() != "AA":
            raise OutsideDomain("splitting probabilities need two absorbing ends")
        pts = self._points(x0)
        lo = self._splitting_once(pts, self._order)
        hi = self._splitting_once(pts, self._order + 8)
        if self._mirror:  # the "right" end of the user is the left end internally
            lo, hi = 1.0 - lo, 1.0 - hi
        shape = np.shape(x0)
        return Result(hi.reshape(shape), (np.abs(hi - lo) + _log_floor(hi)).reshape(shape),
                      "quadrature: scale function on Chebyshev panels")

    def _conditional_once(self, pts: Array, order: int, end: str) -> Array:
        a, b, _ = self._effective_domain(pts)
        g = self._grid(a, b, pts, order)
        log_pb = g.logS_fwd - g.logS_tot
        log_pa = g.logS_rev - g.logS_tot
        right = end == "right"
        if self._mirror:
            right = not right
        logp = log_pb if right else log_pa
        logu = self._green_log(g, logp)
        return np.exp(self._at(g, logu, pts) - self._at(g, logp, pts))

    def conditional_mean_time(self, x0: ArrayLike, end: str) -> Result:
        """Mean first-passage time given that the exit is at ``end`` ("left" or "right")."""
        if self._kind() != "AA" or self.right == "exit":
            raise OutsideDomain("conditional times need two finite absorbing ends")
        if end not in {"left", "right"}:
            raise OutsideDomain("end is 'left' or 'right'")
        pts = self._points(x0)
        lo = self._conditional_once(pts, self._order, end)
        hi = self._conditional_once(pts, self._order + 8, end)
        shape = np.shape(x0)
        return Result(hi.reshape(shape), (np.abs(hi - lo) + _log_floor(hi)).reshape(shape),
                      "quadrature: Green's function with the splitting probability as source")

    # ------------------------------------------------------------ the decay rate
    def _decay_once(self, order: int, max_terms: int) -> tuple[float, dict, Array]:
        """Smallest root of the characteristic function, from its Taylor coefficients.

        The eigenfunction q of L q = -lambda q that meets the boundary condition at one
        end solves the Volterra equation q = q_0 - lambda K q, with K f(x) the double
        integral of w f and then 1/psi from that end. At the other end q is the entire
        function F(lambda) = sum_n (-lambda)^n c_n, with c_n = (K^n q_0) there, and
        F(lambda) = F(0) prod_i (1 - lambda / lambda_i) over the eigenvalues. Every c_n
        is an integral of positive functions, so it is computed in logarithms to full
        relative accuracy; on [0, lambda_1] F is decreasing and convex, so Newton's
        method from 0 rises monotonically to lambda_1. When lambda_2 is close, F is
        nearly flat at the root and the root inherits the coefficients' errors amplified
        (to about their square root for a double root); ``decay_rate`` accounts for it.
        Returns the root, a summary, and the log-coefficients.

        Prior work: the series is the spectral-parameter power series of Kravchenko and
        Porter (Math. Methods Appl. Sci. 33, 459 (2010), Theorem 1), here with zero
        potential, a case they credit to Weyl (1910); that its zeros are the decay rates
        of the passage time is Kent's (Z. Wahrsch. 52, 309 (1980)); the first Newton step
        from zero is the reciprocal mean passage time. What is done here is the
        evaluation of the coefficients in logarithms, which keeps double precision for
        rates near 1e-300 where the published method uses multiprecision arithmetic, and
        the bracket of ``decay_rate``. The bracket rests on estimated coefficient errors,
        not on proven bounds: it is not a validated enclosure.
        """
        pts = np.array([], dtype=float)
        a, b, _ = self._effective_domain(pts)
        # With an exit end, the cut b0 is made absorbing. A path at b0 returns to the
        # well with probability of order exp(-45), so the rate changes by a relative
        # amount of that order; the time spent beyond b0 does not enter a decay rate.
        g = self._grid(a, b, pts, order)
        P = g.panels
        cum = _cheb.log_cumulative
        kind = self._kind()
        reverse = kind == "AR"  # integrate from the reflecting end
        end = (0, 0) if reverse else (P.edges.size - 2, P.n - 1)
        logq = g.logS_fwd if kind == "AA" else np.zeros_like(g.phi)
        logc = [float(logq[end])]

        def more() -> None:
            nonlocal logq
            inner = cum(P, g.logw + logq, reverse=reverse)
            logq = cum(P, -g.phi + inner, reverse=reverse)
            logc.append(float(logq[end]))

        def evaluate(lam: float) -> tuple[float, float, float]:
            """F / s, lambda F' / s and sum |terms| / s for a common scale s."""
            while True:
                n = np.arange(len(logc))
                lt = n * np.log(lam) + np.asarray(logc)
                top = float(np.max(lt))
                last_small = lt[-1] - top < -40.0 and (len(logc) < 3 or lt[-1] < lt[-2])
                if len(logc) >= 4 and last_small:
                    break
                if len(logc) > max_terms:
                    raise RuntimeError("the characteristic series did not converge; raise max_terms")
                more()
            terms = np.exp(lt - top)
            sign = (-1.0) ** n
            return float(np.sum(sign * terms)), float(np.sum(sign * n * terms)), float(np.sum(terms))

        more()
        lam = float(np.exp(logc[0] - logc[1]))  # the Newton step from lambda = 0
        steps = 1
        rounding = np.inf
        converged = False
        previous = np.inf
        while steps < 200 and not converged:
            steps += 1
            F, lamFp, size = evaluate(lam)
            if lamFp >= 0.0:  # cannot happen below lambda_1 except by rounding at the root
                rounding = np.finfo(float).eps * size / max(abs(lamFp), np.finfo(float).tiny)
                break
            step = -F / lamFp  # relative Newton step, (lambda_new - lambda) / lambda
            rounding = 4.0 * np.finfo(float).eps * size / abs(lamFp)
            lam *= 1.0 + step
            # done when the step is at the rounding level, or has stopped shrinking there
            stalled = abs(step) < 1e-10 and abs(step) > 0.9 * previous
            converged = abs(step) <= max(16.0 * np.finfo(float).eps, 4.0 * rounding) or stalled
            if stalled:
                rounding = max(rounding, abs(step))
            previous = abs(step)
        if not converged and lamFp < 0.0:
            rounding = max(rounding, abs(step))
            warnings.warn("Newton's method for the decay rate did not converge in 200 steps; the error "
                          "includes the last step", RuntimeWarning, stacklevel=3)
        info = {"newton_steps": steps, "series_terms": len(logc), "domain": (a, b),
                "panels": P.edges.size - 1, "rounding_error": float(rounding)}
        return lam, info, np.asarray(logc)

    @staticmethod
    def _series(lam: float, logc: Array, eta: Array) -> tuple[float, float]:
        """Smallest and largest value of F(lam) when each c_n may be off by a relative eta_n.

        Both are scaled by the same positive factor, so only their signs are meaningful.
        """
        n = np.arange(logc.size)
        lt = n * np.log(lam) + logc
        t = np.exp(lt - np.max(lt))
        F = float(np.sum((-1.0) ** n * t))
        # the coefficients' errors, and the rounding of this alternating sum itself
        spread = float(np.sum(t * (eta + 4.0 * np.finfo(float).eps * (n + 2))))
        return F - spread, F + spread

    def _rayleigh_upper(self, iterations: int = 40) -> float:
        """An upper bound on lambda_1: the reciprocal of a Rayleigh quotient of the Green's operator.

        Power iteration from a positive vector; every Rayleigh quotient of the positive
        self-adjoint Green's operator is at most its largest eigenvalue 1/lambda_1.
        """
        pts = np.array([], dtype=float)
        a, b, _ = self._effective_domain(pts)
        g = self._grid(a, b, pts, self._order + 8)
        P = g.panels
        logf = np.zeros_like(g.phi)
        best = -np.inf
        for _ in range(iterations):
            logg = self._green_log(g, logf)
            num = _cheb.log_integral(P, g.logw + logf + logg)
            den = _cheb.log_integral(P, g.logw + 2 * logf)
            best = max(best, num - den)
            logf = logg - 0.5 * _cheb.log_integral(P, g.logw + 2 * logg)
        return float(np.exp(-best))

    def _bracket(self, lam: float, logc: Array, eta: Array) -> tuple[float, float]:
        """An interval for lambda_1 from the coefficient intervals c_n (1 +- eta_n).

        Every F with coefficients in those intervals lies between the two bounds of
        ``_series``. Below the first zero of the lower bound, every such F is positive,
        so lambda_1 is at least that zero. At a point where the upper bound is not
        positive, every such F has had a root, so lambda_1 is at most that point; when
        the upper bound stays positive (two roots too close to separate), the
        Rayleigh bound is used instead.
        """
        def sign_change(which: int, lo: float, hi: float) -> tuple[float, float]:
            """Bisection keeping the bound positive at lo and not positive at hi."""
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if self._series(mid, logc, eta)[which] > 0.0:
                    lo = mid
                else:
                    hi = mid
                if hi - lo <= 2.0 * np.finfo(float).eps * hi:
                    break
            return lo, hi

        top = lam
        while self._series(top, logc, eta)[0] > 0.0:  # the lower bound must change sign by here
            top *= 1.0 + 1e-6
            if top > 1.01 * lam:
                raise RuntimeError("the decay-rate bracket failed; the series is too inaccurate")
        # the lower bound is decreasing up to lambda_1 and negative from there to lambda_2,
        # so its sign change below ``top`` is its first zero
        lower = sign_change(0, 0.0, top)[0]
        upper = np.inf
        grid = lam * (1.0 + np.concatenate([[0.0], np.geomspace(1e-15, 1e-2, 131)]))
        previous = 0.0
        for x in grid:
            if self._series(float(x), logc, eta)[1] <= 0.0:
                upper = sign_change(1, previous, float(x))[1] if previous > 0.0 else float(x)
                break
            previous = float(x)
        if not np.isfinite(upper):
            upper = self._rayleigh_upper()
        return lower, upper

    def decay_rate(self, max_terms: int = 4000) -> Result:
        """Smallest eigenvalue of -L: the long-time decay rate of the survival probability.

        Computed as the smallest root of the characteristic function (see
        ``_decay_once``), which keeps full relative accuracy however small the rate is.
        Each coefficient is given an error interval: the difference between two
        discretisations, with a floor of a few units in the last place of its logarithm
        (a coefficient near e^500 is known to about 1e-13). The root is then bracketed:
        below by the first zero of the smallest series those intervals allow, above by
        the first zero of the largest one or, when two roots are too close for that, by a
        Rayleigh quotient. The error is the half-width of that bracket (plus the
        difference between the discretisations). When the two smallest eigenvalues nearly
        coincide (two equal wells far apart) the bracket is about 1e-6 wide.
        """
        lo, info_lo, logc_lo = self._decay_once(self._order, max_terms)
        hi, info, logc_hi = self._decay_once(self._order + 8, max_terms)
        m = min(logc_lo.size, logc_hi.size)
        eps = np.finfo(float).eps
        n = np.arange(logc_hi.size)
        eta = np.empty(logc_hi.size)
        eta[:m] = np.abs(logc_hi[:m] - logc_lo[:m])
        eta[m:] = np.max(eta[:m]) if m else 1.0
        # a coefficient held as its logarithm carries an absolute error of a few units in
        # the last place of that logarithm, which no change of discretisation reveals
        eta = np.maximum(eta, 32.0 * eps * (n + 1) * np.maximum(1.0, np.abs(logc_hi)))
        lower, upper = self._bracket(hi, logc_hi, eta)
        value = min(max(hi, lower), upper)
        info["bracket"] = (lower, upper)
        error = max(value - lower, upper - value) + abs(hi - lo)
        return Result(value, error, "smallest root of the characteristic function (quadrature)", info)
