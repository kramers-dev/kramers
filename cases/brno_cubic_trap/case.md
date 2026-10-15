# Case: a bead in a cubic optical potential (Brno, Olomouc, Prague)

Two related experiments by the groups of P. Zemánek (Institute of Scientific
Instruments, Brno), R. Filip (Palacký University, Olomouc) and A. Ryabov (Charles
University, Prague) approximate the cubic normal form near an inflection,
V(x) = μx³/3, around a microsphere in optical tweezers:

- **P1.** M. Šiler, P. Jákl, O. Brzobohatý, A. Ryabov, R. Filip and P. Zemánek,
  *Sci. Rep.* **7**, 1697 (2017), doi:10.1038/s41598-017-01848-4 (open access).
- **P2.** M. Šiler, L. Ornigotti, O. Brzobohatý, P. Jákl, A. Ryabov, V. Holubec,
  P. Zemánek and R. Filip, *Phys. Rev. Lett.* **121**, 230601 (2018),
  doi:10.1103/PhysRevLett.121.230601; arXiv:1803.07833v2 and its supplement (SM).
- Theory: A. Ryabov, P. Zemánek and R. Filip, *Phys. Rev. E* **94**, 042108 (2016)
  (P3), whose Eq. (6) is the exact mean first-passage time used below.

Everything below is recomputed from the printed parameters by `reproduce.py` (a few
minutes); its output is in `output.txt`. Numbers read from figures carry the precision
of the reading. Computed values are for the ideal model, an overdamped bead with
constant D in an exact cubic potential, and are numerically converged within that
model; nothing is fitted to the data unless the text says so. **This page is a draft
for the authors to correct. Where our numbers differ from theirs, the likeliest reason
is a detail of the set-up that we have not reconstructed; we say what we assumed.**

## 1. P1: the mean first-passage times at the experimental points

P1 releases the bead at x0 = 1.2 µm (Fig. 4; Fig. 1 marks about 1.47 µm) and records
the first passages to 0 and to x1 = −2.1 µm, with μ3 = 6.65 ± 0.05 kT µm⁻³. From the 26
points in Fig. 4: ⟨τ(x0→0)⟩ = 1.16 ± 0.14 s and ⟨τ(x0→x1)⟩ = 3.61 ± 0.29 s; printed:
R = ⟨τ(x0→0)⟩/⟨τ(x0→x1)⟩ = 0.32 ± 0.05 and mean/SD of τ(x0→x1) = 2.3 ± 1.6. Runs lasted
8 s. P1 compares R with 1/3 and mean/SD with √3, the values for distant points (P3
Eqs. 27, 32), and with simulations, which give mean/SD = 1.62 without a time limit and
2.32 with the 8 s limit.

Two readings of the printed 6.65:

- **Reading B, the one P1's simulations use.** P1's Methods state that the measured
  drift coefficient κ is kept and the wall factor γc enters only the random term.
  Written as D μ_eff = κ, with 6.65 = κ/D_Stokes, that is D = D_Stokes/γc and an
  effective μ of 6.65 γc. We take γc = 4.6, the value P1 fitted.
- **Reading A.** Fig. 1 calls 6.65 kT µm⁻³ the reconstructed potential coefficient,
  which taken literally is the potential itself, with D left free. The naming is
  ambiguous; the simulation prescription in the Methods is clearer.

P3's Eq. (6), and the same model with P1's 8 s limit. For the limit we assume that both
times come from the same runs, so that a run that has not reached x1 by 8 s drops out
of both means:

| x0 = 1.2 µm unless stated | R | mean/SD of τ(x0→x1) |
|---|---|---|
| Measured (P1) | 0.32 ± 0.05 | 2.3 ± 1.6 |
| Distant points (P3) | 1/3 | 1.732 |
| P1's simulation, no time limit / 8 s limit | not given | 1.62 / 2.32 |
| Ideal model, reading B, no time limit | 0.306 | 1.587 |
| Ideal model, reading B, 8 s limit | 0.361 | 2.300 |
| Ideal model, reading A, no time limit | 0.287 | 1.495 |
| Ideal model, reading A, x0 = 1.47 µm, no time limit | 0.299 | 1.520 |
| Ideal model, reading A, 8 s limit (D = 0.336 µm²/s, fitted below without the limit) | 0.319 | 1.906 |

Without a time limit, R at the experimental points is 0.29 to 0.31 in the ideal model,
below the asymptotic 1/3 and within the measured 0.32 ± 0.05, and it does not depend on
D; with the 8 s limit it does. Under reading B the model gives mean/SD = 1.587 without
and 2.300 with the 8 s limit, close to P1's own simulations (1.62 and 2.32): numerical
support for reading B, which rests first on the Methods. The time limit, not the
distance of the points, moves mean/SD from about 1.6 to about 2.3.

**The absolute times.** They need D. Under reading B, without a time limit, the model's
means are 1.43 s and 4.69 s, 24% and 30% above the measured 1.16 s and 3.61 s; with the
8 s limit they are 1.39 s and 3.84 s, 19.5% and 6% above (the second within the measured
error, the first about 1.6 standard errors out). Under reading A, the measured means
require D = 0.30 to 0.34 µm²/s without a time limit, a wall factor of about 1.4
against the Stokes value for a 994 nm bead in water at 22 °C (0.458 µm²/s; temperature
and viscosity are our assumptions). Every estimate of D from these means depends on how
unfinished runs, the release position and the time origin were handled, none of which
is printed.

## 2. P2: the simulated survival curves

P2 gives μ = 4.07 ± 0.03 kT µm⁻³ and a quasi-stationary decay rate λ0 = 0.44 ± 0.01 s⁻¹,
but not D. For V = μx³/3 the decay rate is λ0 = E0(0) D μ^(2/3), with
E0(0) = 0.354123... (entry K0008), so P2's λ0 corresponds to D = 0.487 ± 0.011 µm²/s.
P2 computed λ0 from its Eq. (3) with a friction coefficient it does not print, so this
D reconstructs P2's assumed friction; it is not an independent measurement.

The SM states that the dashed curves of SM Fig. S2 simulate the ideal cubic, with a
trajectory counted as diverged when it crosses −10 µm. We solved the same model with
the D above, for all four curves (20 points read from the plot to ±0.005):

| Divergence counted at | D (µm²/s) | RMS error | Largest error |
|---|---|---|---|
| −10 µm, as stated | 0.487 | 0.257 | 0.830 |
| −10 µm | 0.962 (unweighted best fit to the 20 points) | 0.173 | 0.387 |
| −2.75 µm | 0.487 | 0.029 | 0.044 |

No D between 0.2 and 2 µm²/s gives a satisfactory simultaneous fit with −10 µm under
this criterion: at the best D the curve from x0 = −2 µm still has S = 0.56 at
t = 0.1 s, against 0.17. (Without the two points of that curve the best D is about
0.67, which is why the fit is sensitive to it.) With divergence counted at
about −2.75 µm (the outermost absorbing boundary used in SM Fig. S3) and P2's own D,
they agree to within the reading precision plus a few hundredths. **Our solution is
therefore inconsistent with the stated −10 µm criterion but agrees numerically with an
effective boundary near −2.75 µm. This points to a convention or implementation detail
that we have not reconstructed; we do not claim to know which boundary was used.** See
the left panel of `figure.png`.

## 3. An illustrative extension not studied in P1–P3: a tilt swept through the fold

Add a linear tilt to P2's potential, V(x) = μx³/3 − f(t)x, and sweep f linearly through
zero, where the well disappears (the fold). In the ideal model the tilt at which the
bead escapes depends on one number, the sweep rate in units of kT/(ℓτ), with
ℓ = μ^(−1/3) = 0.626 µm and τ = ℓ²/D = 0.805 s (P2's μ and D). Computed with the
time-dependent Fokker-Planck equation (entry K0030), starting 26 fN from the fold from
the quasi-stationary density (well at 1.25 µm, saddle at −1.25 µm, barrier 10.7 kT) and
counting escape at −2.75 µm:

| Sweep rate (fN/s) | Tilt at which 10% / 50% / 90% have escaped (fN) | Quasi-static median (fN) | Shift of the median | Time to the 90% point (s) |
|---|---|---|---|---|
| 0.16 | 14.0 / 9.4 / 5.9 | 9.5 | −1.4% | 125 |
| 0.40 | 11.7 / 6.6 / 2.5 | 6.9 | −4.6% | 58 |
| 0.81 | 9.6 / 4.0 / −0.5 | 4.7 | −13% | 33 |
| 2.42 | 5.3 / −1.2 / −6.8 | 0.5 | changes sign | 14 |

Negative tilts are past the fold. The quasi-static median uses the frozen escape rate
(entry K0024 with K0008). At the fastest sweep shown, the median escape comes after the
fold, where the quasi-static picture puts it before. This is a calculation in the ideal
model: P2 did not calibrate tilted potentials, and whether the trap can realise such a
sweep is an open question.

## 4. Questions for the authors

1. In P1, is the printed μ3 = 6.65 kT µm⁻³ the measured drift coefficient converted
   with the bulk drag, as the simulation prescription in the Methods suggests, or the
   potential itself? At what temperature and viscosity?
2. In P1, is x0 = 1.2 µm (Fig. 4 caption) or about 1.47 µm (Fig. 1); is τ timed from
   the switch to the cubic potential or from the first crossing of x0; and were runs that
   had not reached x1 by 8 s left out of both measured means?
3. What exact drift or potential, D or γ, time origin and divergence criterion were used
   to generate the dashed curves in SM Fig. S2? The stated ideal cubic with divergence
   at −10 µm does not reproduce them in our solution.
4. Would data you already hold test a swept tilt, or is it worth a measurement?

*Prepared with the kramers package. Code and text written largely with AI tools
(Claude, Anthropic) under the direction of Alexander Sokol, and reviewed by a second
AI model (GPT-6 Astra, OpenAI); the computations are reproducible with `reproduce.py`.*
