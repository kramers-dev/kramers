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
Every result is computed at two orders and the difference is returned as its error.

Boundaries
----------
``"absorbing"``, ``"reflecting"``: finite ends.
``"natural"``: an infinite end the process cannot reach (exp(phi) vanishes there); the
    domain is cut where exp(phi) has fallen by e^-50 and the cut is made reflecting.
``"exit"``: an infinite end reached in finite time (an explosive drift, as for the
    noisy saddle-node). The domain is cut at a point ``b0`` far past every barrier, the
    time from ``b0`` to infinity is added by the strong Markov property, and that time
    is computed from local Laplace-type integrals on a compactified tail. Returns from
    beyond ``b0`` are neglected; their probability is below e^-45 by construction.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import roots_laguerre, roots_legendre

from ..result import OutsideDomain, Result
from . import _cheb

Array = NDArray[np.float64]
Func = Callable[[Array], Array]

__all__ = ["FirstPassage1D"]

_FINITE = {"absorbing", "reflecting"}
_INFINITE = {"natural", "exit"}
_DROP = 50.0  # e-folds by which a truncated weight has fallen
_RETURN = 45.0  # e-folds that make a return from beyond b0 negligible


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

    def _march(self, start: float, direction: float, *, exit: bool, entrance: bool = False) -> float:
        """Walk outwards until the weight is negligible (natural) or the tail is safe (exit)."""
        step = max(0.5, 0.25 * abs(start))
        x = start
        xs = np.linspace(start, start + direction * 1e-9, 2)
        extreme = float(np.max(self._phi(xs))) if not exit else float(np.min(self._phi(xs)))
        for _ in range(400):
            x_new = x + direction * step
            seg = np.linspace(x, x_new, 33)
            vals = self._phi(seg)
            if not np.all(np.isfinite(vals)):
                raise OutsideDomain(f"log-weight is not finite near {x_new}")
            if not exit:
                extreme = max(extreme, float(np.max(vals)))
                outward_decreasing = vals[-1] < vals[-2]
                if vals[-1] < extreme - _DROP and outward_decreasing and (not entrance or self._tail_safe(x_new)):
                    return float(x_new)
            else:
                extreme = min(extreme, float(np.min(vals)))
                if vals[-1] > extreme + _RETURN and self._tail_safe(x_new):
                    return float(x_new)
            x = x_new
            step *= 1.5
        kind = "an exit (explosive)" if exit else "a natural"
        raise OutsideDomain(f"the boundary does not behave as {kind} boundary")

    def _effective_domain(self, points: Array, entrance: bool = False) -> tuple[float, float, float]:
        """Finite cuts (a, b) of the domain and, for a start at -infinity, the entrance cut.

        A natural end is cut where exp(phi) has fallen by e^-50, so a reflecting wall
        there changes no result at the starting points by more than that. For a start
        at -infinity the entrance tail begins at a cut ``a0`` and the domain itself is
        extended 50 more e-folds beyond it, so that the wall is negligible at ``a0`` too.
        """
        inner = np.concatenate([points, np.asarray(self._user_breaks, dtype=float)])
        lo = float(np.min(inner)) if inner.size else 0.0
        hi = float(np.max(inner)) if inner.size else 0.0
        a, b = self.a, self.b
        a0 = np.nan
        if self.left == "natural":
            if entrance:
                a0 = self._march(lo, -1.0, exit=False, entrance=True)
                a = self._march(a0, -1.0, exit=False)
            else:
                a = self._march(lo, -1.0, exit=False)
        if self.right == "natural":
            b = self._march(hi, +1.0, exit=False)
        if self.right == "exit":
            b = self._march(hi, +1.0, exit=True)
        return a, b, a0

    def _grid(self, a: float, b: float, points: Array, order: int) -> _Grid:
        breaks = [*points.tolist(), *self._user_breaks]
        panels = _cheb.build_panels(self._phi, a, b, breakpoints=breaks, n=self._order)
        panels = panels.with_order(order)
        x = panels.x
        phi = self._phi(x)
        logD = np.log(self._D(x))
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
            error=np.abs(hi - lo).reshape(shape),
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
        return Result(hi.reshape(shape), np.abs(hi - lo).reshape(shape),
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
        return Result(hi.reshape(shape), np.abs(hi - lo).reshape(shape),
                      "quadrature: Green's function with the splitting probability as source")

    def _decay_once(self, order: int, tol: float, max_iter: int) -> tuple[float, dict]:
        pts = np.array([], dtype=float)
        a, b, _ = self._effective_domain(pts)
        if self.right == "exit":
            # the decay rate does not see the tail: an absorbing cut at b0 changes it by
            # less than exp(-45) relative (see the module docstring)
            pass
        g = self._grid(a, b, pts, order)
        P = g.panels
        logf = np.zeros_like(g.phi)
        mu_old = np.inf
        mu = np.nan
        it = 0
        for it in range(1, max_iter + 1):
            logg = self._green_log(g, logf)
            num = _cheb.log_integral(P, g.logw + logf + logg)
            den = _cheb.log_integral(P, g.logw + 2 * logf)
            mu = float(np.exp(num - den))
            norm = 0.5 * _cheb.log_integral(P, g.logw + 2 * logg)
            logf = logg - norm
            if abs(mu - mu_old) <= tol * abs(mu):
                break
            mu_old = mu
        return 1.0 / mu, {"iterations": it, "domain": (a, b), "panels": P.edges.size - 1}

    def decay_rate(self, tol: float = 1e-15, max_iter: int = 2000) -> Result:
        """Smallest eigenvalue of -L: the long-time decay rate of the survival probability.

        Power iteration on the Green's operator, so it keeps full relative accuracy
        however small the rate is.
        """
        lo, _ = self._decay_once(self._order, tol, max_iter)
        hi, info = self._decay_once(self._order + 8, tol, max_iter)
        return Result(hi, abs(hi - lo), "power iteration on the Green's operator (quadrature)", info)
