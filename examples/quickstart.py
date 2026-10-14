"""Sixty seconds with kramers: published numbers, recomputed, with their sources.

    python examples/quickstart.py
"""

import warnings

import numpy as np

import kramers
from kramers.formulas import (
    kramers_rate_overdamped,
    passage_time_saddle_node,
    resetting_optimal_rate,
    saddle_node_threshold_statistics,
    sparre_andersen_survival,
)
from kramers.solvers import FirstPassage1D

print(f"kramers {kramers.__version__}: {len(kramers.entries())} entries\n")

s = saddle_node_threshold_statistics(D=1.0)
print("1. A model neuron at threshold (K0005; Lindner, Longtin and Bulsara 2003, Eq. 3.6)")
print(f"   firing rate {s.rate:.6f} D^(1/3)   (printed: 0.201)    CV {s.cv:.6f}   (printed: 1/sqrt 3)\n")

print("2. Passage through a noisy saddle-node, closed form in Airy functions (K0004; Hathcock and Sethna 2021)")
for eps0 in (-2.0, 0.0, 2.0):
    print(f"   epsilon0 = {eps0:+.0f}, g0 = 1:  mean passage time {passage_time_saddle_node(eps0, 1.0):.6f}")
print()

print("3. How good is Kramers' formula? Against the exact escape rate of a cubic well (K0001)")
for B in (5.0, 10.0, 20.0):
    u = (0.75 * B) ** (2 / 3)
    exact = float(FirstPassage1D(lambda y, u=u: u * y - y**3 / 3, D=1.0, left="natural", right="exit",
                                 dlog_psi=lambda y, u=u: y * y - u).decay_rate().value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", kramers.ValidityWarning)
        k = kramers_rate_overdamped(2 * np.sqrt(u), 2 * np.sqrt(u), B, 1.0)
    print(f"   barrier {B:4.0f} kT: Kramers {k:.4e}, exact {exact:.4e}, too high by {100 * (k / exact - 1):.1f}%")
print()

print("4. Searching with resetting: the optimal rate (K0016; Evans and Majumdar 2011, Eq. 8)")
print(f"   r* x0^2 / D = {resetting_optimal_rate(1.0, 1.0):.6f}   (printed: 2.53964...)\n")

print("5. Sparre Andersen: survival of any symmetric random walk (K0020)")
print("   " + ", ".join(f"q_{n} = {sparre_andersen_survival(n):.6f}" for n in (1, 2, 3, 10)) + "\n")

print("Where each of these is printed, and how it is checked:")
for key in ("K0005", "K0004", "K0001", "K0016", "K0020"):
    e = kramers.get(key)
    prim = next(s for s in e.sources if s.role == "primary")
    print(f"   {e.id}: {prim.citation}, {prim.locator}")
