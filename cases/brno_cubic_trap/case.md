# Case: a bead in a cubic optical potential (Brno, Olomouc, Prague)

Two experiments by the groups of P. Zemánek (Institute of Scientific Instruments, Brno),
R. Filip (Palacký University, Olomouc) and A. Ryabov (Charles University, Prague) are
the only ones we found that build the potential of a noisy saddle-node exactly at its
bifurcation, V(x) = μx³/3, around a microsphere in optical tweezers:

- **P1.** M. Šiler, P. Jákl, O. Brzobohatý, A. Ryabov, R. Filip and P. Zemánek,
  *Sci. Rep.* **7**, 1697 (2017), doi:10.1038/s41598-017-01848-4 (open access).
- **P2.** M. Šiler, L. Ornigotti, O. Brzobohatý, P. Jákl, A. Ryabov, V. Holubec,
  P. Zemánek and R. Filip, *Phys. Rev. Lett.* **121**, 230601 (2018),
  doi:10.1103/PhysRevLett.121.230601; arXiv:1803.07833v2 and its supplement.
- Theory: A. Ryabov, P. Zemánek and R. Filip, *Phys. Rev. E* **94**, 042108 (2016)
  (P3), whose Eq. (6) is the exact mean first-passage time used below.

Everything below is recomputed from the printed parameters by `reproduce.py` (about
two minutes); its output is in `output.txt`. Numbers read from figures carry the
precision of the reading. **This page is a draft for the authors to correct. Nothing
here says their analysis is wrong; where we disagree, the most likely reason is that
we misread the set-up, and we say what we assumed.**

## 1. What P1 measured, and the exact quadrature at the experimental points

P1 releases the bead at x0 = 1.2 µm (Fig. 4; Fig. 1 marks about 1.47 µm) and records
the first passages to 0 and to x1 = −2.1 µm, with μ3 = 6.65 ± 0.05 kT µm⁻³. From the
26 points in Fig. 4: ⟨τ(x0→0)⟩ = 1.16 ± 0.14 s, ⟨τ(x0→x1)⟩ = 3.61 ± 0.29 s, ratio
R = 0.32 ± 0.05 (printed), mean/SD of τ(x0→x1) = 2.3 ± 1.6 (printed). P1 compares R
with 1/3 and mean/SD with √3, the values for distant points (P3 Eqs. 27, 32), and with
simulations.

P3's Eq. (6), evaluated exactly at the experimental points (quadrature solver of this
package, accurate to about 1e-12):

| | R = ⟨τ(x0→0)⟩/⟨τ(x0→x1)⟩ | mean/SD of τ(x0→x1) |
|---|---|---|
| Measured (P1) | 0.32 ± 0.05 | 2.3 ± 1.6 |
| Distant points (P3) | 1/3 | 1.732 |
| P1's own simulation at the experimental points | not given | 1.62 |
| Exact, x0 = 1.2 µm, μ3 = 6.65 taken as the potential (reading A) | 0.287 | 1.495 |
| Exact, x0 = 1.47 µm, reading A | 0.299 | 1.520 |
| Exact, x0 = 1.2 µm, drift measured and γc = 4.6 in the noise only (reading B) | 0.306 | 1.587 |

The exact ratio at the experimental points is 0.29 to 0.31, below the asymptotic 1/3
and well within the measured 0.32 ± 0.05. Reading B reproduces P1's simulated mean/SD
(1.62) to 2%.

**What we could not settle.** The absolute times need the bead's diffusion coefficient,
which P1 does not print. Under reading A (μ3 is the potential in kT, D free), the
measured means require D = 0.30 to 0.34 µm²/s, a wall factor of about 1.4 against the
Stokes value for a 994 nm bead in water at 22 °C (0.458 µm²/s; our assumption). P1
reports γc = 4.2 ± 0.3 and 4.6 ± 0.2. Under reading B (P1 Methods: the drift was
measured near the surface, so the wall factor enters only the noise), γc = 4.6 gives
1.43 s and 4.69 s, about 25% longer than measured; both scale with the assumed
viscosity and temperature, which P1 does not state.

## 2. What P2 simulated, reproduced

P2 gives μ = 4.07 ± 0.03 kT µm⁻³ and a quasi-stationary decay rate λ0 = 0.44 ± 0.01 s⁻¹,
but not D. For V = μx³/3 the decay rate is λ0 = E0(0) D μ^(2/3), with
E0(0) = 0.354123... (entry K0008), so P2's λ0 implies **D = 0.487 ± 0.011 µm²/s**, 7%
above the Stokes value for a 1 µm bead in water at 22 °C (0.455).

With that D, the survival curves of P2's simulations (SM Fig. S2, read to ±0.005) are
reproduced to within 0.05 (most points within 0.03) **if divergence is counted at
about −2.75 µm**, the absorbing boundary used in SM Fig. S3; with −10 µm, which is how
we read the text, they are not (deviations up to 0.45, at x0 = −1 µm and t = 0.3 s). See
the left panel of `figure.png`.

## 3. A measurement not yet made: escape under a ramped tilt

If a linear tilt f is added to P2's potential and lowered at a constant rate to zero
(the fold), the bead escapes at a tilt that depends on one number, the ramp rate in
units of kT/(ℓτ) with ℓ = μ^(−1/3) = 0.626 µm and τ = ℓ²/D = 0.805 s. Computed with the
time-dependent Fokker-Planck equation (entry K0030), starting 26 fN from the fold (well
at 1.25 µm, inside P2's calibrated range) and counting escape at −2.75 µm:

| Ramp of the tilt (fN/s) | Tilt at which 10% / 50% / 90% have escaped (fN) | Quasi-static median (fN) | One run (s) |
|---|---|---|---|
| 0.16 | 14.0 / 9.4 / 5.9 | 9.5 | 161 |
| 0.40 | 11.7 / 6.6 / 2.5 | 6.9 | 64 |
| 0.81 | 9.6 / 4.0 / −0.5 | 4.7 | 32 |
| 2.42 | 5.3 / −1.2 / −6.8 | 0.5 | 11 |

Negative values: the bead escapes after the fold. At the faster ramps the frozen-rate
(quasi-static) picture, the standard one for switching under a ramp, is visibly wrong,
and most beads leave only after the potential has lost its well. One thousand runs
take 3 to 45 hours.

## 4. Questions for the authors

1. In P1, is μ3 = 6.65 kT µm⁻³ the potential itself (the measured drift converted with
   γ0γc), or the drift coefficient converted with the bulk drag γ0? At what
   temperature and viscosity?
2. In P1, is x0 = 1.2 µm (Fig. 4 caption) or about 1.47 µm (Fig. 1), and is τ timed
   from the switch to the cubic potential or from the first crossing of x0?
3. In P2's SM Fig. S2, at which position was a trajectory counted as diverged, in the
   simulation and in the measurement?
4. Would data you already hold test the ramp of section 3, or is it worth a
   measurement?

*Prepared with the kramers package. Code and text written largely with AI tools
(Claude, Anthropic) under the direction of Alexander Sokol; the computations are
reproducible with `reproduce.py`.*
