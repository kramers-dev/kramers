"""First-passage times by direct simulation of the stochastic differential equation.

Model (Ito)::

    dX = A(X, t) dt + sqrt(2 D) dW

Euler-Maruyama steps. At an absorbing end the probability that the path crossed and
came back within a step is added (Brownian-bridge correction, so that the bias in the
first-passage time is of order dt rather than sqrt(dt)). Reflecting ends fold the path
back. Optional Poissonian resetting to a fixed point, as in stochastic-resetting
problems. Every random draw comes from a ``numpy.random.Generator`` built from
``seed``.

This is the slow, assumption-light reference: it is used to check formulas whose
derivation one does not want to trust, and its error bars are statistical.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..result import OutsideDomain, Result

Array = NDArray[np.float64]

__all__ = ["SimulationResult", "first_passage_times"]


@dataclass(frozen=True)
class SimulationResult:
    """First-passage times of ``n`` paths; ``inf`` for paths still alive at ``t_max``."""

    times: Array
    exit: NDArray[np.int8]  # -1 left end, +1 right end, 0 not absorbed
    info: dict = field(default_factory=dict)

    @property
    def absorbed(self) -> NDArray[np.bool_]:
        return self.exit != 0

    def mean(self) -> Result:
        """Mean first-passage time with one standard error. Refuses if paths are censored."""
        if not np.all(self.absorbed):
            raise OutsideDomain(f"{np.sum(~self.absorbed)} paths not absorbed by t_max; the mean is biased")
        t = self.times
        return Result(float(t.mean()), float(t.std(ddof=1) / np.sqrt(t.size)),
                      "Euler-Maruyama simulation, bridge-corrected", dict(self.info))

    def second_moment(self) -> Result:
        if not np.all(self.absorbed):
            raise OutsideDomain("paths not absorbed by t_max")
        t2 = self.times**2
        return Result(float(t2.mean()), float(t2.std(ddof=1) / np.sqrt(t2.size)),
                      "Euler-Maruyama simulation, bridge-corrected", dict(self.info))

    def probability(self, end: str) -> Result:
        """Fraction absorbed at ``end`` ("left" or "right"), with its binomial standard error."""
        side = -1 if end == "left" else 1
        p = float(np.mean(self.exit == side))
        n = self.exit.size
        return Result(p, float(np.sqrt(max(p * (1 - p), 1e-300) / n)), "simulation", dict(self.info))

    def survival(self, t: Array) -> Array:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return np.mean(self.times[None, :] > t[:, None], axis=1)


def first_passage_times(
    drift: Callable[..., Array],
    D: float,
    x0: float,
    *,
    n_paths: int,
    dt: float,
    t_max: float,
    absorbing: tuple[float | None, float | None] = (None, None),
    reflecting: tuple[float | None, float | None] = (None, None),
    time_dependent: bool = False,
    reset_rate: float = 0.0,
    reset_to: float | None = None,
    reset_period: float | None = None,
    seed: int | np.random.Generator = 0,
) -> SimulationResult:
    """Simulate ``n_paths`` paths from ``x0`` until they are absorbed or ``t_max`` passes.

    Parameters
    ----------
    absorbing, reflecting
        (left, right) positions; None for no boundary on that side.
    reset_rate
        Rate of Poissonian resetting to ``reset_to`` (default ``x0``).
    reset_period
        Instead, reset deterministically every ``reset_period`` (sharp resetting).
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    a_abs, b_abs = absorbing
    a_ref, b_ref = reflecting
    if a_abs is None and b_abs is None:
        raise OutsideDomain("give at least one absorbing end")
    lo = a_abs if a_abs is not None else a_ref
    hi = b_abs if b_abs is not None else b_ref
    if (lo is not None and x0 < lo) or (hi is not None and x0 > hi):
        raise OutsideDomain("x0 outside the domain")
    if x0 == a_abs or x0 == b_abs:
        raise OutsideDomain("a start on an absorbing end is absorbed at once; start inside the domain")
    if reset_rate > 0 and reset_period is not None:
        raise OutsideDomain("choose Poissonian or periodic resetting, not both")
    reset_point = x0 if reset_to is None else float(reset_to)
    x = np.full(n_paths, float(x0))
    times = np.full(n_paths, np.inf)
    exit_side = np.zeros(n_paths, dtype=np.int8)
    alive = np.arange(n_paths)
    sigma = np.sqrt(2.0 * D * dt)
    n_steps = int(np.ceil(t_max / dt))
    p_reset = -np.expm1(-reset_rate * dt) if reset_rate > 0 else 0.0
    steps_per_period = int(round(reset_period / dt)) if reset_period is not None else 0
    since_reset = 0
    t = 0.0
    for _ in range(n_steps):
        if alive.size == 0:
            break
        xa = x[alive]
        drift_val = drift(xa, t) if time_dependent else drift(xa)
        x_new = xa + drift_val * dt + sigma * rng.standard_normal(alive.size)
        t_new = t + dt
        hit = np.zeros(alive.size, dtype=np.int8)
        if a_abs is not None:
            crossed = x_new <= a_abs
            inside = ~crossed
            gap = np.maximum((xa - a_abs) * (x_new - a_abs), 0.0)
            bridge = rng.random(alive.size) < np.exp(-gap / (D * dt))
            hit[crossed | (inside & bridge)] = -1
        if b_abs is not None:
            crossed = x_new >= b_abs
            inside = ~crossed
            gap = np.maximum((b_abs - xa) * (b_abs - x_new), 0.0)
            bridge = rng.random(alive.size) < np.exp(-gap / (D * dt))
            right_hit = (crossed | (inside & bridge)) & (hit == 0)
            hit[right_hit] = 1
        if a_ref is not None:
            x_new = np.where(x_new < a_ref, 2.0 * a_ref - x_new, x_new)
        if b_ref is not None:
            x_new = np.where(x_new > b_ref, 2.0 * b_ref - x_new, x_new)
        done = hit != 0
        if np.any(done):
            idx = alive[done]
            times[idx] = t_new
            exit_side[idx] = hit[done]
        survivors = ~done
        x_new = x_new[survivors]
        alive = alive[survivors]
        if p_reset > 0 and alive.size:
            reset = rng.random(alive.size) < p_reset
            x_new[reset] = reset_point
        since_reset += 1
        if steps_per_period and since_reset >= steps_per_period:
            x_new[:] = reset_point
            since_reset = 0
        x[alive] = x_new
        t = t_new
    info = {"n_paths": n_paths, "dt": dt, "t_max": t_max, "bias": "first order in dt"}
    return SimulationResult(times=times, exit=exit_side, info=info)
