"""The validity atlas: where approximate entries of the handbook fail, computed.

A map is the signed relative error of one or more approximate entries against an exact
computation, over a grid of their parameters. For each map this module holds

- ``compute()``: the full grid (run by ``tools/build_atlas.py``, which stores the result
  as ``kramers/data/atlas_<id>.json`` and draws the figure on the website);
- ``facts(data)``: the numbers quoted in the map's claim, derived from the stored data,
  so the sentence cannot drift away from the computation;
- ``recompute(data)``: a few grid points computed afresh, for the nightly check that
  the stored map is current.

Use::

    from kramers import atlas
    atlas.maps()                 # the maps, in order
    m = atlas.get("M03")
    data = m.load()              # the stored grid: axes, series, facts
    print(m.claim())             # the computed statement, numbers filled in from the data

The exact side of every comparison is named in ``reference``. A map says nothing about
the parameters outside its grid.
"""

from __future__ import annotations

import datetime as dt
import json
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import airy

from ._checking import absolute
from .registry import Check, CheckResult
from .result import OutsideDomain, ValidityWarning

_DATA = Path(__file__).resolve().parent / "data"
FORMAT = "kramers-atlas/1"


@dataclass(frozen=True)
class Prior:
    """Earlier work on the same question, with whether we have read it."""

    citation: str
    doi: str
    read: str  # what was opened, or "not opened"


@dataclass(frozen=True)
class AtlasMap:
    id: str
    title: str
    entries: tuple[str, ...]  # handbook entries whose approximations are mapped
    approximation: str  # what is compared, in words
    reference: str  # what "exact" means here, and how it is computed
    quantity: str  # what the stored numbers are
    claim_template: str  # str.format over facts()
    compute: Callable[[], dict[str, Any]]
    facts: Callable[[Mapping[str, Any]], dict[str, Any]]
    recompute: Callable[[Mapping[str, Any]], list[tuple[str, float, float]]]
    notes: str = ""
    prior: tuple[Prior, ...] = ()
    status: str = "draft"  # "draft" until the maintainer has approved the claim; then "approved"

    @property
    def path(self) -> Path:
        return _DATA / f"atlas_{self.id}.json"

    def load(self) -> dict[str, Any]:
        """The stored map (raises FileNotFoundError before tools/build_atlas.py has run)."""
        return json.loads(self.path.read_text())

    def claim(self, data: Mapping[str, Any] | None = None) -> str:
        """The map's statement, with its numbers taken from the (stored) data."""
        facts = self.facts(data if data is not None else self.load())
        missing = [k for k, v in facts.items() if isinstance(v, float) and not np.isfinite(v)]
        if missing:
            raise ValueError(f"{self.id}: the claim quotes values the grid does not define: {', '.join(missing)}")
        return self.claim_template.format(**facts)

    def checks(self) -> tuple[Check, Check]:
        """The map's checks: its claim follows from its stored data (fast); its stored data are current (slow)."""
        return (Check("the claim follows from the stored data", "the stored grid", self._check_claim),
                Check("the stored map is current", "the same computation, at sample points", self._check_current,
                      slow=True))

    def _check_claim(self) -> CheckResult:
        data = self.load()
        now = self.facts(data)
        rows = []
        for k, v in data["facts"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                rows.append((k, float(now[k]), float(v)))
            elif now[k] != v:
                rows.append((k, 1.0, 0.0))
        self.claim(data)  # raises if a quoted value is undefined
        return absolute(rows, 0.0, what="from the data now", ref="stored with the map")

    def _check_current(self) -> CheckResult:
        return absolute(self.recompute(self.load()), 1e-6, what="computed now", ref="stored")

    def build(self) -> dict[str, Any]:
        """Compute the grid and wrap it with the map's metadata (not written to disk)."""
        from ._version import __version__

        data = self.compute()
        data = {"format": FORMAT, "id": self.id, "title": self.title, "entries": list(self.entries),
                "approximation": self.approximation, "reference": self.reference, "quantity": self.quantity,
                "computed": dt.date.today().isoformat(), "kramers": __version__, **data}
        data["facts"] = self.facts(data)
        return data


_MAPS: dict[str, AtlasMap] = {}


def _register(m: AtlasMap) -> AtlasMap:
    if m.id in _MAPS:
        raise ValueError(f"map {m.id} is already registered")
    _MAPS[m.id] = m
    return m


def maps() -> list[AtlasMap]:
    """All maps, in order of their numbers."""
    return [_MAPS[k] for k in sorted(_MAPS)]


def get(key: str) -> AtlasMap:
    """A map by id (``"M02"``)."""
    return _MAPS[key]


def for_entry(entry_id: str) -> list[AtlasMap]:
    """The maps that cover a handbook entry."""
    return [m for m in maps() if entry_id in m.entries]


def _r(v: float, nd: int = 4) -> float:
    return float(round(float(v), nd))


def _clean(a: np.ndarray) -> list:
    """Floats for JSON, with None where a value is undefined."""
    return [None if not np.isfinite(v) else float(v) for v in np.ravel(a)]


def _arr(v: Any) -> np.ndarray:
    return np.array([np.nan if x is None else x for x in np.ravel(np.asarray(v, dtype=object))], dtype=float)


# --------------------------------------------------------------------------------------- M01
# Kramers' overdamped rate (K0001) on three wells, against the exact decay rate.

_M01_SHAPES = ("cubic well", "quartic double well", "cosine well")
_M01_BARRIERS = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 17.0, 20.0, 25.0, 30.0, 35.0,
                  40.0, 50.0, 60.0, 70.0, 80.0)


def _m01_point(shape: str, B: float) -> float:
    from .handbook.k0001_kramers_overdamped import _exact_and_formula

    k, lam = _exact_and_formula(shape, B)
    return k / lam - 1.0


def _m01_compute() -> dict[str, Any]:
    series = {s: _clean(np.array([_m01_point(s, B) for B in _M01_BARRIERS])) for s in _M01_SHAPES}
    return {"axes": {"barrier_kT": list(_M01_BARRIERS)}, "series": series}


def _first_below(B: np.ndarray, err: np.ndarray, tol: float) -> float:
    """The barrier above which |error| <= tol everywhere on the grid: the last crossing of tol,
    placed by linear interpolation of the error in log(barrier) between the two grid points."""
    a = np.abs(err)
    bad = np.where(~(a <= tol))[0]
    if bad.size == 0:
        return float(B[0])
    i = int(bad[-1])
    if i == B.size - 1:
        return float("nan")
    t = (a[i] - tol) / (a[i] - a[i + 1])
    return float(np.exp(np.log(B[i]) + t * (np.log(B[i + 1]) - np.log(B[i]))))


def _m01_facts(d: Mapping[str, Any]) -> dict[str, Any]:
    B = _arr(d["axes"]["barrier_kT"])
    f: dict[str, Any] = {"bmin": _r(B[0], 2), "bmax": _r(B[-1], 2)}
    keys = {"cubic well": "c", "quartic double well": "q", "cosine well": "k"}
    allpos = True
    for s, k in keys.items():
        e = _arr(d["series"][s])
        allpos &= bool(np.all(e > 0))
        f[f"b5{k}"] = _r(_first_below(B, e, 0.05), 1)
        f[f"b1{k}"] = _r(_first_below(B, e, 0.01), 1)
        f[f"e10{k}"] = _r(100 * float(e[int(np.argmin(np.abs(B - 10.0)))]), 1)
    f["sign"] = "too high" if allpos else "not always too high"
    return f


def _m01_recompute(d: Mapping[str, Any]) -> list[tuple[str, float, float]]:
    rows = []
    B = list(d["axes"]["barrier_kT"])
    for s in _M01_SHAPES:
        for i in (0, len(B) // 2, len(B) - 1):
            rows.append((f"{s}, {B[i]:g} kT", _m01_point(s, B[i]), float(d["series"][s][i])))
    return rows


_register(AtlasMap(
    id="M01",
    title="Kramers' overdamped rate on three wells",
    entries=("K0001",),
    approximation="Kramers' overdamped escape rate (K0001)",
    reference=("the exact decay rate of the survival probability (the smallest eigenvalue of the backward "
               "operator), from the quadrature solver; set-ups as in K0001's own checks"),
    quantity="Kramers' rate / exact decay rate - 1, at each barrier (kT)",
    claim_template=("Kramers' overdamped rate is {sign} at every barrier from {bmin:g} to {bmax:g} kT on all three "
                    "wells. At 10 kT it is {e10c}%, {e10q}% and {e10k}% high (cubic well, quartic double well, "
                    "cosine well). It is within 5% above {b5c:g}, {b5q:g} and {b5k:g} kT, and within 1% above {b1c:g}, "
                    "{b1q:g} and {b1k:g} kT (each threshold placed between two grid barriers by interpolation)."),
    compute=_m01_compute,
    facts=_m01_facts,
    recompute=_m01_recompute,
    notes=("Barriers below 2 kT, and the comparison with the reciprocal mean passage time (which differs from "
           "the decay rate at low barriers), are left out of this map for now. Cosine well: escape over either "
           "of two barriers, so twice the one-sided Kramers rate."),
    prior=(
        Prior("A. Edholm and O. Leimar, The accuracy of Kramers' theory of chemical kinetics, Physica A 98, 313 (1979)",
              "10.1016/0378-4371(79)90182-1", "not opened (title and bibliographic record checked on Crossref)"),
        Prior("J. J. Mazo, O. Y. Fajardo and D. Zueco, Thermal activation at moderate-to-high and high damping: "
              "finite barrier effects and force spectroscopy, J. Chem. Phys. 138, 104105 (2013)",
              "10.1063/1.4793983", "not opened (bibliographic record checked on Crossref)"),
    ),
))


# --------------------------------------------------------------------------------------- M02
# Rate of the noisy saddle-node: the weak-noise (K0006) and weak-bias (K0007) formulas
# against the exact rate 1/T in Airy functions (K0004). With D = 1 all three depend on
# b = beta / D^(2/3) only: r = D^(1/3) f(b).

_M02_B = tuple(float(b) for b in np.round(np.arange(-10.0, 6.0 + 1e-9, 0.02), 2))


def _m02_exact(b: np.ndarray) -> np.ndarray:
    ai, _, bi, _ = airy(-b)
    return 1.0 / (np.pi**2 * (ai**2 + bi**2))


def _m02_series(b: np.ndarray) -> dict[str, np.ndarray]:
    from .handbook.k0006_saddle_node_weak_noise import saddle_node_kramers_rate
    from .handbook.k0007_saddle_node_linear_response import saddle_node_rate_linear

    ex = _m02_exact(b)
    weak = np.full_like(b, np.nan)
    neg = b < 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValidityWarning)
        weak[neg] = saddle_node_kramers_rate(b[neg], 1.0) / ex[neg] - 1.0
    lin = saddle_node_rate_linear(b, 1.0) / ex - 1.0
    return {"weak noise (K0006)": weak, "weak bias (K0007)": lin}


def _m02_compute() -> dict[str, Any]:
    b = np.array(_M02_B)
    s = _m02_series(b)
    return {"axes": {"b": list(_M02_B)}, "series": {k: _clean(v) for k, v in s.items()}}


def _m02_facts(d: Mapping[str, Any]) -> dict[str, Any]:
    b = _arr(d["axes"]["b"])
    weak = _arr(d["series"]["weak noise (K0006)"])
    lin = _arr(d["series"]["weak bias (K0007)"])
    i0 = int(np.argmin(np.abs(b)))
    f: dict[str, Any] = {}
    for tol, tag in ((0.01, "1"), (0.05, "5"), (0.10, "10")):
        ok = np.abs(lin) <= tol
        lo = hi = i0
        while lo > 0 and ok[lo - 1]:
            lo -= 1
        while hi < b.size - 1 and ok[hi + 1]:
            hi += 1
        f[f"lin{tag}_lo"], f[f"lin{tag}_hi"] = _r(b[lo], 2), _r(b[hi], 2)
        okw = (np.abs(weak) <= tol) & (b < 0)
        first_bad = int(np.where(~okw)[0][0])  # from the left: all more negative b are within tol
        bw = float(b[first_bad - 1]) if first_bad > 0 else float("nan")
        f[f"weak{tag}"] = _r(bw, 2)
        f[f"weak{tag}_barrier"] = _r(4.0 / 3.0 * abs(bw) ** 1.5, 1)
    win = (b < 0) & (b > -4)
    j = int(np.nanargmax(np.where(win, weak, -np.inf)))
    f["weak_peak"], f["weak_peak_b"] = _r(100 * weak[j], 1), _r(b[j], 2)
    f["gap10_lo"], f["gap10_hi"] = f["weak10"], f["lin10_lo"]
    f["gap5_lo"], f["gap5_hi"] = f["weak5"], f["lin5_lo"]
    from .handbook.k0007_saddle_node_linear_response import _C0, _C1

    f["lin_zero"] = _r(-_C0 / _C1, 2)
    return f


def _m02_recompute(d: Mapping[str, Any]) -> list[tuple[str, float, float]]:
    b = _arr(d["axes"]["b"])
    now = _m02_series(b)
    rows = []
    for name, vals in now.items():
        stored = _arr(d["series"][name])
        for i in range(0, b.size, 40):
            if np.isfinite(stored[i]):
                rows.append((f"{name}, b = {b[i]:g}", float(vals[i]), float(stored[i])))
    return rows


_register(AtlasMap(
    id="M02",
    title="Firing rate of the noisy saddle-node: weak-noise and weak-bias formulas",
    entries=("K0006", "K0007"),
    approximation="the weak-noise rate (K0006) and the rate to first order in the bias (K0007)",
    reference=("the exact rate 1/T, T the mean passage time from minus to plus infinity, in Airy functions (K0004); "
               "the sources define the rate this way, as a firing rate"),
    quantity="formula / exact rate - 1, against the scaled bias b = beta / D^(2/3)",
    claim_template=("For dx/dt = beta + x^2 + sqrt(2D) xi both formulas depend only on b = beta/D^(2/3). The "
                    "weak-noise rate is within 10% of the exact rate only for b <= {weak10:g} (a barrier of "
                    "{weak10_barrier:g} noise units, 4|b|^(3/2)/3), within 5% for b <= {weak5:g} and within 1% for "
                    "b <= {weak1:g} (barrier {weak1_barrier:g}); closer to the bifurcation it is up to "
                    "{weak_peak}% too high (at b = {weak_peak_b:g}). The weak-bias line is within 10% for "
                    "{lin10_lo:g} <= b <= {lin10_hi:g}, within 1% for {lin1_lo:g} <= b <= {lin1_hi:g}, and gives a "
                    "negative rate below b = {lin_zero:g}. So for {gap10_lo:g} < b < {gap10_hi:g} neither formula "
                    "is within 10% (within 5%: {gap5_lo:g} < b < {gap5_hi:g} is uncovered)."),
    compute=_m02_compute,
    facts=_m02_facts,
    recompute=_m02_recompute,
    notes=("All three rates are in closed form, so the map is exact up to rounding. b > 0 is above the bifurcation "
           "(no barrier), where the weak-noise formula does not apply."),
    prior=(
        Prior("B. Lindner, A. Longtin and A. Bulsara, Analytic expressions for rate and CV of a type I neuron "
              "driven by white Gaussian noise, Neural Comput. 15, 1761 (2003)", "10.1162/08997660360675035",
              "author's PDF, read 12 Oct 2026 for Eqs. (3.10)-(3.11); whether it plots this comparison was not "
              "checked"),
    ),
))


# --------------------------------------------------------------------------------------- M03
# The force-dependent rate of Dudko, Hummer and Szabo (K0026) against exact decay rates.

_M03_DG = (5.0, 7.0, 10.0, 14.0, 20.0, 28.0, 40.0, 56.0, 80.0)
_M03_F = tuple(float(v) for v in np.round(np.arange(0.0, 0.981, 0.02), 2))
_M03_NU = {"linear-cubic (nu = 2/3)": 2.0 / 3.0, "cusp (nu = 1/2)": 0.5}


def _m03_exact(nu: float, dG: float, F: float) -> float:
    """Exact decay rate (kT = D = 1, x = 1) for the landscapes behind nu = 2/3 and nu = 1/2."""
    from .solvers.quadrature import FirstPassage1D

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValidityWarning)
        if nu == 0.5:  # harmonic well with a cusp at y = 1 (absorbing there)
            fp = FirstPassage1D(lambda y: dG * y * y - F * y, D=1.0, domain=(-np.inf, 1.0), left="natural",
                                right="absorbing")
        else:  # linear-cubic surface of K0026, escape to +infinity
            fp = FirstPassage1D(lambda y: 1.5 * dG * y - 2.0 * dG * y**3 - F * y, D=1.0, left="natural",
                                right="exit")
        return float(fp.decay_rate().value)


def _m03_point(nu: float, dG: float, f: float, lam0: float | None = None) -> float:
    from .handbook.k0026_dhs_rate import dhs_rate

    F = f * dG / nu
    lam0 = _m03_exact(nu, dG, 0.0) if lam0 is None else lam0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValidityWarning)
        ratio = dhs_rate(F, 1.0, 1.0, dG, nu)
    return ratio / (_m03_exact(nu, dG, F) / lam0) - 1.0


def _m03_compute() -> dict[str, Any]:
    series = {}
    for name, nu in _M03_NU.items():
        grid = np.empty((len(_M03_DG), len(_M03_F)))
        for i, dG in enumerate(_M03_DG):
            lam0 = _m03_exact(nu, dG, 0.0)
            for j, f in enumerate(_M03_F):
                grid[i, j] = _m03_point(nu, dG, f, lam0)
        series[name] = [_clean(row) for row in grid]
    return {"axes": {"barrier_kT": list(_M03_DG), "force_over_critical": list(_M03_F)}, "series": series}


def _m03_facts(d: Mapping[str, Any]) -> dict[str, Any]:
    dG = _arr(d["axes"]["barrier_kT"])
    f = _arr(d["axes"]["force_over_critical"])
    out: dict[str, Any] = {}
    for name, nu, tag in (("linear-cubic (nu = 2/3)", 2.0 / 3.0, "c"), ("cusp (nu = 1/2)", 0.5, "p")):
        g = np.array([_arr(row) for row in d["series"][name]])
        left = dG[:, None] * (1.0 - f[None, :]) ** (1.0 / nu)  # barrier left at force F
        region = left >= 3.0
        k = np.unravel_index(int(np.nanargmax(np.where(region, np.abs(g), -np.inf))), g.shape)
        out[f"worst3{tag}"] = _r(100 * g[k], 1)
        out[f"worst3{tag}_left"] = _r(left[k], 1)
        out[f"worst3{tag}_dG"] = _r(dG[k[0]], 0)
        for tol, t in ((0.05, "5"), (0.10, "10")):
            # the smallest barrier-left threshold above which every grid point is within tol
            bad = left[~(np.abs(g) <= tol)]
            out[f"left{t}{tag}"] = _r(float(np.max(bad)) if bad.size else 0.0, 1)
    return out


def _m03_recompute(d: Mapping[str, Any]) -> list[tuple[str, float, float]]:
    rows = []
    dG = list(d["axes"]["barrier_kT"])
    f = list(d["axes"]["force_over_critical"])
    for name, nu in _M03_NU.items():
        for i, j in ((2, 20), (6, 35), (8, 45)):
            rows.append((f"{name}, dG = {dG[i]:g}, F/Fc = {f[j]:g}", _m03_point(nu, dG[i], f[j]),
                         float(d["series"][name][i][j])))
    return rows


_register(AtlasMap(
    id="M03",
    title="Force-dependent rate of Dudko, Hummer and Szabo",
    entries=("K0026",),
    approximation=("the force dependence k(F)/k(0) of the Dudko-Hummer-Szabo rate (K0026), with k(0) the exact "
                   "zero-force rate, as when k0 is fitted"),
    reference=("the exact decay rate at each force, from the quadrature solver, for the landscapes the formula is "
               "derived for: the tilted linear-cubic surface (nu = 2/3) and a harmonic well cut off by a cusp, "
               "absorbing there (nu = 1/2)"),
    quantity=("[k_DHS(F)/k_DHS(0)] / [lambda(F)/lambda(0)] - 1, over the zero-force barrier dG (kT) and F as a "
              "fraction of the critical force dG/(nu x)"),
    claim_template=("Where at least 3 kT of barrier is left at the applied force (the formula warns below that), "
                    "the largest error on the grid in the force dependence of the rate is {worst3c:+g}% for the linear-cubic "
                    "surface (at {worst3c_left:g} kT left, dG = {worst3c_dG:g} kT) and {worst3p:+g}% for the cusp (at "
                    "{worst3p_left:g} kT left, dG = {worst3p_dG:g} kT). Every grid point is within 10% where more "
                    "than {left10c:g} kT (linear-cubic) or {left10p:g} kT (cusp) of barrier is left, and within 5% "
                    "above {left5c:g} kT and {left5p:g} kT."),
    compute=_m03_compute,
    facts=_m03_facts,
    recompute=_m03_recompute,
    notes=("The error is zero at zero force by construction (k0 is the exact zero-force rate). For nu = 2/3 the "
           "formula is Kramers' rate for the tilted surface (K0026's first check), so this map is the cubic curve "
           "of M01 seen through the force: the error depends on the barrier left at F and at zero force only. "
           "kT = D = x = 1 throughout; the map is in scaled units."),
    prior=(
        Prior("O. K. Dudko, G. Hummer and A. Szabo, Intrinsic rates and activation free energies from "
              "single-molecule pulling experiments, Phys. Rev. Lett. 96, 108101 (2006)", "10.1103/PhysRevLett.96.108101",
              "published PDF (Zenodo record 1233953), read 14 Oct 2026 for Eqs. (1)-(6); its comparisons with "
              "simulation were not reviewed for this map"),
        Prior("J. J. Mazo, O. Y. Fajardo and D. Zueco, Thermal activation at moderate-to-high and high damping: "
              "finite barrier effects and force spectroscopy, J. Chem. Phys. 138, 104105 (2013)",
              "10.1063/1.4793983", "not opened; the closest earlier work by its title, to read before any claim of novelty"),
    ),
))


# --------------------------------------------------------------------------------------- M04
# Rupture force under constant loading on the linear-cubic surface: the Dudko-Hummer-Szabo
# mean and spread (K0027) and the Bell-Evans most probable force (K0029), against the exact
# quasi-static distribution (K0024 with the exact decay rate at each force).

_M04_DG = (7.0, 10.0, 14.0, 20.0, 28.0, 40.0, 56.0, 80.0)
_M04_S = tuple(float(v) for v in np.round(np.arange(0.05, 1.101, 0.05), 2))
_M04_SERIES = ("mean force, DHS (K0027)", "spread of the force, DHS (K0027)", "most probable force, Bell-Evans (K0029)")
_M04_TAIL = 1e-3  # largest allowed probability of surviving to the critical force


def _m04_rates(dG: float) -> tuple[CubicSpline, float, float]:
    """log(exact decay rate) against force, as a spline from 0 to 1.3 Fc; returns (spline, k0, Fc)."""
    Fc = dG / (2.0 / 3.0)
    F = np.linspace(0.0, 1.3 * Fc, 131)
    ll = np.log([_m03_exact(2.0 / 3.0, dG, v) for v in F])
    return CubicSpline(F, ll), float(np.exp(ll[0])), Fc


def _m04_point(dG: float, s: float, spl: CubicSpline, k0: float, Fc: float) -> tuple[float, float, float]:
    from .handbook.k0024_switching_distribution import switching_distribution
    from .handbook.k0027_dhs_rupture_force_moments import dhs_rupture_force_moments
    from .handbook.k0029_bell_evans_force import bell_evans_force

    r = np.exp(s * dG) * k0  # loading rate: ln(r x / (k0 kT)) = s dG, x = kT = 1
    dist = switching_distribution(lambda v: np.exp(spl(v)), r, 1.3 * Fc, partial=True)
    if float(dist.survival(np.array([Fc]))[0]) > _M04_TAIL:
        return (np.nan, np.nan, np.nan)
    try:
        m = dhs_rupture_force_moments(r, k0, 1.0, dG, 2.0 / 3.0)
        e_mean = m.mean / dist.mean - 1.0
        e_sd = np.sqrt(m.variance / dist.variance) - 1.0
    except OutsideDomain:
        e_mean = e_sd = np.nan
    try:
        e_mode = bell_evans_force(r, k0, 1.0) / dist.mode - 1.0
    except OutsideDomain:
        e_mode = np.nan
    return (e_mean, e_sd, e_mode)


def _m04_compute() -> dict[str, Any]:
    grids = {k: np.full((len(_M04_DG), len(_M04_S)), np.nan) for k in _M04_SERIES}
    for i, dG in enumerate(_M04_DG):
        spl, k0, Fc = _m04_rates(dG)
        for j, s in enumerate(_M04_S):
            for k, v in zip(_M04_SERIES, _m04_point(dG, s, spl, k0, Fc), strict=True):
                grids[k][i, j] = v
    return {"axes": {"barrier_kT": list(_M04_DG), "log_loading": list(_M04_S)},
            "series": {k: [_clean(row) for row in g] for k, g in grids.items()}}


def _m04_facts(d: Mapping[str, Any]) -> dict[str, Any]:
    dG = _arr(d["axes"]["barrier_kT"])
    s = _arr(d["axes"]["log_loading"])
    g = {k: np.array([_arr(row) for row in d["series"][k]]) for k in _M04_SERIES}
    out: dict[str, Any] = {}

    def from_s(row: np.ndarray, ok: Callable[[np.ndarray], np.ndarray]) -> float:
        """The grid loading s from which every faster computed point satisfies ok."""
        fin = np.isfinite(row)
        bad = np.where(fin & ~ok(row))[0]
        if bad.size == 0:
            return float(s[np.where(fin)[0][0]])
        nxt = np.where(fin & (np.arange(s.size) > bad[-1]))[0]
        return float(s[nxt[0]]) if nxt.size else float("nan")

    be = g["most probable force, Bell-Evans (K0029)"]
    out["be_neg_from"] = _r(max(from_s(row, lambda v: v < 0) for row in be), 2)
    j05 = int(np.argmin(np.abs(s - 0.5)))
    out["be05_lo"], out["be05_hi"] = _r(-100 * np.nanmax(be[:, j05]), 0), _r(-100 * np.nanmin(be[:, j05]), 0)
    out["be_most"] = _r(-100 * float(np.nanmin(be)), 0)
    out["be_pos"] = _r(100 * float(np.nanmax(be)), 0)
    mean, sd = g["mean force, DHS (K0027)"], g["spread of the force, DHS (K0027)"]
    for b in (10.0, 20.0, 40.0, 80.0):
        i = int(np.argmin(np.abs(dG - b)))
        out[f"m5_{b:.0f}"] = _r(from_s(mean[i], lambda v: np.abs(v) <= 0.05), 2)
        out[f"sd5_{b:.0f}"] = _r(from_s(sd[i], lambda v: np.abs(v) <= 0.05), 2)
    i10, j01 = int(np.argmin(np.abs(dG - 10.0))), int(np.argmin(np.abs(s - 0.1)))
    out["mean_slow"], out["sd_slow"] = _r(100 * mean[i10, j01], 0), _r(100 * sd[i10, j01], 0)
    last = [float(np.max(s[np.isfinite(row)])) for row in be if np.any(np.isfinite(row))]
    out["smax_lo"], out["smax_hi"] = _r(min(last), 2), _r(max(last), 2)
    return out


def _m04_recompute(d: Mapping[str, Any]) -> list[tuple[str, float, float]]:
    dG = list(d["axes"]["barrier_kT"])
    s = list(d["axes"]["log_loading"])
    i = dG.index(20.0)
    spl, k0, Fc = _m04_rates(20.0)
    rows = []
    for j in (3, 9, 15):
        now = _m04_point(20.0, s[j], spl, k0, Fc)
        for k, v in zip(_M04_SERIES, now, strict=True):
            stored = d["series"][k][i][j]
            if stored is not None and np.isfinite(v):
                rows.append((f"{k}, dG = 20, s = {s[j]:g}", float(v), float(stored)))
    return rows


_register(AtlasMap(
    id="M04",
    title="Rupture force under constant loading: Dudko-Hummer-Szabo and Bell-Evans",
    entries=("K0027", "K0029"),
    approximation=("the mean and spread of the rupture force of Dudko, Hummer and Szabo (K0027, nu = 2/3) and the "
                   "Bell-Evans most probable force (K0029), with k0 the exact zero-force rate and x the zero-force "
                   "distance to the barrier"),
    reference=("the exact quasi-static rupture-force distribution (K0024) built from the exact decay rate at each "
               "force on the tilted linear-cubic surface (quadrature solver, interpolated in force by a spline of "
               "its logarithm, 131 forces from 0 to 1.3 times the critical force)"),
    quantity=("formula / exact - 1 for the mean, the standard deviation and the most probable force, over the "
              "zero-force barrier dG (kT) and the scaled loading rate s = ln(r x / (k0 kT)) / (dG / kT)"),
    claim_template=("On the linear-cubic surface the Bell-Evans most probable force is too low wherever "
                    "s >= {be_neg_from:g}, by an amount that grows with the loading rate and hardly depends on the "
                    "barrier: {be05_lo:g}% to {be05_hi:g}% at s = 0.5, and up to {be_most:g}% at the fastest "
                    "loading mapped. (Bell's model takes the distance to the barrier as fixed; on this surface it "
                    "shrinks under force.) At the slowest loading on low barriers it is too high instead, by up to "
                    "{be_pos:g}%. The Dudko-Hummer-Szabo mean force is within 5% of the exact mean for s >= "
                    "{m5_10:g}, {m5_20:g}, {m5_40:g} and {m5_80:g} at zero-force barriers of 10, 20, 40 and 80 kT, "
                    "and its spread within 5% for s >= {sd5_10:g}, {sd5_20:g}, {sd5_40:g} and {sd5_80:g} (grid "
                    "values). At slower loading, where the force rises by only a few kT/x during the zero-force "
                    "lifetime 1/k0, both fail: at "
                    "10 kT and s = 0.1 the mean is off by {mean_slow:+g}% and the spread by {sd_slow:+g}%. Loading "
                    "faster than s = {smax_lo:g} to {smax_hi:g} (depending on the barrier), where more than 0.1% "
                    "of bonds would survive to the critical force, is left out: there the frozen-rate picture "
                    "itself fails."),
    compute=_m04_compute,
    facts=_m04_facts,
    recompute=_m04_recompute,
    notes=("Both sides use the quasi-static (frozen-rate) law of K0024, so the map isolates the error of the "
           "closed forms; the frozen-rate law itself fails at fast loading (K0030 for the noisy fold). Points where "
           "a formula refuses its input (OutsideDomain: loading so slow that the bond breaks before the force "
           "matters) are blank. kT = D = x = 1."),
    prior=(
        Prior("O. K. Dudko, G. Hummer and A. Szabo, Intrinsic rates and activation free energies from "
              "single-molecule pulling experiments, Phys. Rev. Lett. 96, 108101 (2006)", "10.1103/PhysRevLett.96.108101",
              "published PDF (Zenodo record 1233953), read 14 Oct 2026 for Eqs. (1)-(6)"),
        Prior("J. T. Bullerjahn, S. Sturm and K. Kroy, Theory of rapid force spectroscopy, Nat. Commun. 5, 4463 (2014)",
              "10.1038/ncomms5463", "not opened (bibliographic record checked on Crossref)"),
        Prior("A. Maitra and G. Arya, Model accounting for the effects of pulling-device stiffness in the analyses "
              "of single-molecule force measurements, Phys. Rev. Lett. 104, 108301 (2010)",
              "10.1103/PhysRevLett.104.108301", "not opened (bibliographic record checked on Crossref)"),
    ),
))


# --------------------------------------------------------------------------------------- M05
# The histogram-to-rate estimator (K0025) and the standard error kramers adds to it. Given
# N_k switches still to come when the ramp enters bin k, the survivors past the bin are
# binomial, X ~ Bin(N_k, exp(-mu)), where mu = Gamma dI / |dI/dt| is the expected hazard in
# the bin. The estimate is ln(N_k / X) times |dI/dt| / dI, with standard error
# sqrt(n_k / (N_k X)) in the same units, n_k = N_k - X. Its relative bias, spread and the
# coverage of that standard error therefore depend on (N_k, mu) only, and are exact sums.

_M05_N = tuple(int(v) for v in np.unique(np.round(np.geomspace(10, 1e5, 17)).astype(int)))
_M05_MU = tuple(float(v) for v in np.round(np.geomspace(0.003, 5.0, 17), 5))
_M05_SERIES = ("relative bias of the rate", "relative spread of the rate", "coverage of the reported standard error")
_M05_UNDEFINED = 1e-3  # cells where the estimate is undefined (no survivors) this often are left out


def _m05_cell(N: int, mu: float) -> tuple[float, float, float, float]:
    """(bias, spread, coverage, P(no survivors)) of K0025's estimate for one bin, exactly."""
    from scipy.stats import binom

    X = np.arange(1, N + 1, dtype=float)
    w = binom.pmf(X, N, np.exp(-mu))
    p0 = float(binom.pmf(0, N, np.exp(-mu)))
    w = w / w.sum()
    est = np.log(N / X)
    se = np.sqrt((N - X) / (N * X))
    mean = float(np.sum(w * est))
    spread = float(np.sqrt(np.sum(w * (est - mean) ** 2)))
    cover = float(np.sum(w[np.abs(est - mu) <= se]))
    return mean / mu - 1.0, spread / mu, cover, p0


def _m05_compute() -> dict[str, Any]:
    grids = {k: np.full((len(_M05_N), len(_M05_MU)), np.nan) for k in _M05_SERIES}
    for i, N in enumerate(_M05_N):
        for j, mu in enumerate(_M05_MU):
            b, s, c, p0 = _m05_cell(N, mu)
            if p0 <= _M05_UNDEFINED:
                for k, v in zip(_M05_SERIES, (b, s, c), strict=True):
                    grids[k][i, j] = v
    return {"axes": {"entering": list(_M05_N), "hazard_per_bin": list(_M05_MU)},
            "series": {k: [_clean(row) for row in g] for k, g in grids.items()}}


def _m05_facts(d: Mapping[str, Any]) -> dict[str, Any]:
    N = _arr(d["axes"]["entering"])
    mu = _arr(d["axes"]["hazard_per_bin"])
    g = {k: np.array([_arr(row) for row in d["series"][k]]) for k in _M05_SERIES}
    switches = N[:, None] * (1.0 - np.exp(-mu[None, :]))  # expected switches in the bin
    survivors = N[:, None] * np.exp(-mu[None, :])  # expected survivors past it
    least = np.minimum(switches, survivors)
    cov, bias = g["coverage of the reported standard error"], g["relative bias of the rate"]
    out: dict[str, Any] = {}

    def threshold(ok: np.ndarray) -> float:
        """The smallest count c in the list such that every mapped cell with least >= c satisfies ok."""
        for c in (3, 5, 10, 20, 30, 50, 100, 200, 500, 1000):
            sel = (least >= c) & np.isfinite(cov)
            if sel.any() and bool(np.all(ok[sel])):
                return float(c)
        return float("nan")

    out["cov_c"] = threshold((cov >= 0.60) & (cov <= 0.76))
    out["bias1_c"] = threshold(np.abs(bias) <= 0.01)
    out["bias5_c"] = threshold(np.abs(bias) <= 0.05)
    few = (switches < 1.0) & np.isfinite(cov)
    out["cov_few"] = _r(100 * float(np.nanmin(cov[few])), 0)
    out["cov_ideal"] = 68.3
    return out


def _m05_recompute(d: Mapping[str, Any]) -> list[tuple[str, float, float]]:
    N = list(d["axes"]["entering"])
    mu = list(d["axes"]["hazard_per_bin"])
    rows = []
    for i, j in ((0, 8), (4, 12), (8, 4), (12, 14), (16, 0)):
        now = _m05_cell(N[i], mu[j])[:3]
        for k, v in zip(_M05_SERIES, now, strict=True):
            stored = d["series"][k][i][j]
            if stored is not None:
                rows.append((f"{k}, N = {N[i]}, mu = {mu[j]:g}", float(v), float(stored)))
    return rows


_register(AtlasMap(
    id="M05",
    title="Rates from a switching histogram: bias, spread and the error bar",
    entries=("K0025",),
    approximation=("the Fulton-Dunkleberger estimate of the rate in one bin (K0025) and the delta-method standard "
                   "error that kramers reports with it (an addition, not in the source)"),
    reference=("the exact distribution of the estimate: given N switches still to come when the ramp enters the "
               "bin, the survivors past it are binomial with probability exp(-mu), mu the expected hazard in the "
               "bin; sums over that distribution, conditioned on at least one survivor"),
    quantity=("relative bias and relative spread of the estimated rate, and the probability that the reported "
              "standard error covers the true rate (68.3% if it were a Gaussian one-sigma interval), over N and mu"),
    claim_template=("A one-sigma error bar should cover the true rate {cov_ideal:g}% of the time. The one kramers "
                    "reports with each bin's rate does so roughly (60% to 76% on this grid) wherever at least "
                    "{cov_c:g} switches are expected both inside the bin and after it, and as rarely as {cov_few:g}% "
                    "for bins where fewer than one switch is expected: such a bin often reports a rate of zero with "
                    "an error bar of zero. The estimate itself is biased by less than 1% where at least {bias1_c:g} "
                    "switches are expected both inside the bin and after it, and by less than 5% where at least "
                    "{bias5_c:g} are."),
    compute=_m05_compute,
    facts=_m05_facts,
    recompute=_m05_recompute,
    notes=("Counting statistics of a single bin only. Not included: the error from assigning the rate to the bin "
           "centre (second order in the bin width, K0025), correlations between bins (they share survivors), and "
           "anything beyond a rate process under a ramp. Cells where no survivor is left with probability above "
           "0.1% are not mapped; there the estimate is often undefined."),
    prior=(
        Prior("T. A. Fulton and L. N. Dunkleberger, Lifetime of the zero-voltage state in Josephson tunnel junctions, "
              "Phys. Rev. B 9, 4760 (1974)", "10.1103/PhysRevB.9.4760", "not opened (closed access)"),
    ),
))
