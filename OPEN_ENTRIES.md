# Open entries

An **open entry** is a published first-passage or escape result that the handbook does
not hold yet. Each one is sized for one contributor and one sitting. Every item below
names the paper and DOI, the equation as printed and the copy where it was read, the
conventions, and the check that defines "done".

**How to claim one.** Open an issue titled `Claim: <item id>` (for example
`Claim: O002`). Copy `tools/entry_template.py` to `src/kramers/handbook/k0NNN_name.py`
and follow CONTRIBUTING.md. The entry is done when the check described in its item
passes at the stated tolerance. Claims lapse after two weeks without a pull request.

**Status of this list (14 October 2026).** This is a partial list: the first research
pass stopped after three items. Only items whose source was opened, and whose equation
was seen, are listed as open entries; the other candidates are under "Could not open"
and move up as their sources are opened. The target is fifty items across the seven
areas.

**Evidence labels.**
- OPENED: the equation was seen in the copy named (URL and location).
- Crossref ✓: authors, title, journal, volume, first page or article number, and year
  match `https://api.crossref.org/works/<DOI>`.
- Sanity test: a quick numerical test of the transcription, run on 14 October 2026 and
  not kept. It is not the contributor's check.

Not opened in this pass, and checked first by the claimant: journal equation
numbering, for items read only on arXiv.

Excluded from this list, because the handbook already holds them (K0001-K0030) or they
are reserved for the founding author: the first-passage density to a point, with and
without drift (Lévy-Smirnov, inverse Gaussian). That exclusion also covers the
white-noise perfect integrate-and-fire neuron, whose interspike interval density is the
same inverse Gaussian.

---

## Escape over barriers

No item reached OPENED in this pass. Candidates are under "Could not open".

## Driven escape and switching

No item reached OPENED in this pass. Candidates are under "Could not open".

## Diffusion to targets

No item reached OPENED in this pass. Candidates are under "Could not open".

## Stochastic resetting

### O001. Mean first-passage time to a sphere under resetting in d dimensions
- **Paper.** Martin R Evans; Satya N Majumdar, "Diffusion with resetting in arbitrary
  spatial dimension", *Journal of Physics A: Mathematical and Theoretical* 47, 285001
  (2014). doi:10.1088/1751-8113/47/28/285001. Crossref ✓.
- **Equation.** OPENED: arXiv:1404.4574v1 (the only version), two copies.
  - The LaTeX alttext of Eqs. (64)-(78) at https://ar5iv.labs.arxiv.org/html/1404.4574
    (anchors `#S4.E67`, `#S4.E68`, `#S4.E69`, `#S4.E71`, `#S4.E76`).
  - The page image of PDF p. 14 (Eqs. 68-71).
  - The journal numbering was not checked.
  - Main result, Eq. (65):
    `T(R_r)=\frac{1}{r}\left[\left(\frac{a}{R_r}\right)^{\nu}\frac{K_\nu(\alpha_0 a)}{K_\nu(\alpha_0 R_r)}-1\right]`
  - The same in scaled form, Eq. (67):
    `T=\frac{1}{\gamma^{2}}\,\left[\epsilon^{\nu}\,\frac{K_{|\nu|}(\epsilon\,\gamma)}{K_{|\nu|}(\gamma)}-1\right]\,\frac{R_{r}^{2}}{D}`
  - Limits, Eqs. (70) and (75):
    `T_{d=1}=\frac{1}{\gamma^{2}}\left[{\rm e}^{\gamma}-1\right]\,\frac{R_{r}^{2}}{D}`,
    `T_{d=2}=\frac{1}{\gamma^{2}}\,\left[\frac{K_{0}(\epsilon\,\gamma)}{K_{0}(\gamma)}-1\right]\,\frac{R_{r}^{2}}{D}`
  - Symbols (p. 14): ν = 1 − d/2, ε = a/R_r, γ = √(r/D) R_r.
- **Conventions.** ⟨ξ_i(t)ξ_j(t′)⟩ = 2Dδ_ijδ(t−t′), so ⟨x_i²⟩ = 2Dt per coordinate.
  - The sphere |x| = a is absorbing; Q(|x₀| = a, t) = 0.
  - The particle starts at, and resets to, a point X_r at distance R_r > a.
  - Resetting is Poissonian at rate r. No units are fixed.
- **Check.** `simulate.first_passage_times` on the radial process, with drift
  D(d−1)/x, `absorbing=(a, None)` and `reset_rate=r` (the reset point defaults to x0 = R_r).
  - Parameters: d = 3, D = 1, a = 0.5, R_r = 1, r ∈ {0.5, 2, 8}; 10⁵ paths; dt = 10⁻⁴
    and 2.5×10⁻⁵.
  - Pass: within 4 standard errors, and the bias falls with dt.
  - Exact cross-check: d = 1 must equal K0015 with distance R_r − a, to 1e-12.
- **Difficulty.** One hour.
- **Watch.**
  - Eq. (65) needs K_{|ν|}: K is even in its order. Eq. (67) writes the |ν| out, and
    (65) does not.
  - The reset point is the start point.
  - In d ≥ 2 the mean diverges as a → 0 (Eqs. 71, 76). Raise `OutsideDomain` for a = 0
    when d ≥ 2.
  - Sanity test: Eq. (65) at d = 1 equals (e^{α₀(R_r−a)} − 1)/r to 6e-16 relative, and
    (65) equals (67) at d = 3.

### O002. Mean first-passage time under resetting in an interval with two absorbing ends
- **Paper.** Arnab Pal; V. V. Prasad, "First passage under stochastic resetting in an
  interval", *Physical Review E* 99, 032123 (2019). doi:10.1103/PhysRevE.99.032123.
  Crossref ✓.
- **Equation.** OPENED: arXiv:1812.03009 (v2 current), LaTeX alttext at
  https://ar5iv.labs.arxiv.org/html/1812.03009, anchor `#S3.E15`, Eq. (15). The journal
  numbering was not checked.
  - `\langle T_{r}(x_{0})\rangle=\frac{1}{r}\left[\frac{\sinh(b-a)\alpha_{0}}{\sinh(b-x_{0})\alpha_{0}+\sinh(x_{0}-a)\alpha_{0}}-1\right]`
  - Eq. (16): α₀ = √(r/D), the s → 0 value of α = √((r+s)/D).
- **Conventions.** Eq. (1) is the backward equation
  `\frac{\partial Q_{r}(x,t)}{\partial t}=D\frac{\partial^{2}Q_{r}(x,t)}{\partial x^{2}}-rQ_{r}(x,t)+rQ_{r}(x_{0},t)`,
  so ⟨x²⟩ = 2Dt between resets.
  - Both ends a < b are absorbing.
  - The particle starts at, and resets to, x₀ ∈ (a, b), at constant rate r. The paper's
    figures use D = 1/2.
- **Check.**
  - (1) Exact: the renewal formula (K0017), with the interval's Laplace transform
    [sinh((b−x₀)α) + sinh((x₀−a)α)]/sinh((b−a)α), α = √(s/D). Pass at 1e-12.
  - (2) Independent: `simulate.first_passage_times(drift=0, D=0.5, x0, absorbing=(a, b), reset_rate=r)`
    with a = −0.5, b = 1.2, x₀ = 0.3, r ∈ {0.5, 3}; 10⁵ paths. Pass within 4 standard errors.
  - (3) As r → 0 the value must tend to (x₀−a)(b−x₀)/(2D) (K0012).
- **Difficulty.** One hour.
- **Watch.**
  - The print writes `\sinh(b-a)\alpha_{0}` for sinh((b−a)α₀). Read the argument as the
    product.
  - Resetting can raise or lower the mean here. Do not assume a gain.
  - Sanity test: at r = 10⁻⁶ the formula gives 0.5142858 against
    (x₀−a)(b−x₀)/2D = 0.5142857 (D = 0.7).

## Active and anomalous motion

### O003. Universal survival probability of a run-and-tumble particle in d dimensions
- **Paper.** Francesco Mori; Pierre Le Doussal; Satya N. Majumdar; Grégory Schehr,
  "Universal Survival Probability for a d-Dimensional Run-and-Tumble Particle",
  *Physical Review Letters* 124, 090603 (2020). doi:10.1103/PhysRevLett.124.090603.
  Crossref ✓.
- **Equation.** OPENED: arXiv:2001.01492v1, LaTeX at
  https://ar5iv.labs.arxiv.org/html/2001.01492, anchor `#S0.E1`, Eq. (1). The journal
  numbering was not checked.
  - `S(t)=\frac{1}{2}e^{-\gamma t/2}\left(I_{0}\left(\gamma t/2\right)+I_{1}\left(\gamma t/2\right)\right)`
- **Conventions.**
  - Runs are straight, at speed v₀ (or at a speed drawn afresh after each tumble from
    any distribution).
  - Tumbles happen at constant rate γ, and each draws a new isotropic direction.
  - The particle starts at the origin with a random velocity.
  - S(t) is the probability that the x-component has stayed positive up to t.
  - S does not depend on d or on v₀. It needs exponential run times.
- **Check.** A plain event-driven Monte Carlo in d = 1, 2, 3, with γ = 1.3, v₀ = 1,
  t ∈ {0.5, 1, 5, 20} and 2×10⁵ paths.
  - Within a straight run, x crosses 0 if and only if the end of the run is negative, so
    the simulation has no time step.
  - Pass within 4 standard errors at every t and in every d.
  - Also check S(0) = 1/2 and S ~ 1/√(πγt) at large t.
- **Difficulty.** One hour.
- **Watch.**
  - γ is the tumble rate with a fresh random direction. In d = 1 this is not the
    reversal rate (Malakar et al. 2018 use the reversal rate).
  - S(0) = 1/2 because half the particles start moving into x < 0.
  - Sanity test: Monte Carlo in d = 3 at t = 1 (2×10⁵ paths) gave 0.37797 ± 0.00108;
    the formula gives 0.37872. The asymptote 1/√(πγt) agrees to 0.1% at t = 200.

## Transport and neurons

No item reached OPENED in this pass. Candidates are under "Could not open".

---

## Could not open

Nothing below is an open entry. Each needs its source opened first.

**Opened, but the transcription is incomplete**
- Evans & Majumdar, "Effects of refractory period on stochastic resetting", *J. Phys. A*
  52, 01LT01 (2019), doi:10.1088/1751-8121/aaf080 (Crossref ✓), arXiv:1809.01551.
  - The ar5iv page (Eqs. 35-38, anchors `#S3.E36`, `#S3.E38`) came back as garbled
    rendered text, not LaTeX.
  - The definitions of g(t), h(t) and ⟨τ|t⟩ were not captured.
  - A possible dτ/dt inconsistency in Eq. (38) may be a rendering artefact. Do not
    report it.
  - Reopen it as a page image.

**Papers located but not opened.** These records come from the arXiv API (`journal_ref`
and DOI fields); Crossref was checked only where marked ✓.

| Paper | arXiv | Journal |
|---|---|---|
| Ray & Reuveni, diffusion with resetting in a logarithmic potential | 2004.01898 | J. Chem. Phys. 152, 234110 (2020), Crossref ✓ |
| Nagar & Gupta, resetting at power-law times | 1512.02092 | PRE 93, 060102 |
| Pal, Kuśmierz & Reuveni, space-time coupled returns | 1907.13453 | NJP 21, 113024 |
| Mercado-Vásquez et al., intermittent resetting potentials | 2007.15696 | J. Stat. Mech. (2020) 113203 |
| Dhar et al., run-and-tumble particle in confining potentials | 1811.03808 | PRE 99, 032132 |
| Dybiec, Gudowska-Nowak & Hänggi, Lévy flights on intervals | cond-mat/0512492 | PRE 73, 046104 |
| Chupeau, Bénichou & Majumdar, Brownian motion in a wedge | 1501.02274 | PRE 91, 032106 |
| Grebenkov, mean first-passage time in planar domains | 1612.05889 | PRL 117, 260201 |
| Godec & Metzler, proximity effect | 1611.07788 | PRX 6, 041037 |
| Grebenkov, paradigm shift (partially reactive surfaces) | 2007.11224 | PRL 125, 078102 |
| Lanoiselée, Moutal & Grebenkov, diffusing diffusivity | 1911.01656 | Nat. Commun. 9, 4398 |
| Singh & Kundu, generalised arcsine laws for run-and-tumble | 1906.09442 | J. Stat. Mech., doi 10.1088/1742-5468/ab3283 |
| Krug et al., persistence exponents | cond-mat/9704238 | PRE 56, 2702 |
| Reguera & Rubí, entropic barriers | cond-mat/0110442 | PRE 64, 061106 |
| Vilela & Lindner, comparative study of integrate-and-fire neurons | 0912.2336 | PRE 80, 031909 |
| Bouchet & Reygner, Eyring-Kramers for irreversible diffusions | 1507.02104 | doi 10.1007/s00023-016-0507-4 |
| Pollak & Ankerhold, improvements to Kramers turnover theory | 1304.1933 | JCP 138, 164116 |

**No usable DOI.** Berglund, Kramers' law review, arXiv:1106.5799 (MPRF 19, 459, 2013),
and Berglund & Gentz, nonquadratic saddles, arXiv:0807.1681 (MPRF 16, 549, 2010). They
are useful as second sources only.

**Not on arXiv by exact title (arXiv API).**
- Bullerjahn, Sturm & Kroy, *Nat. Commun.* 5, 4463 (2014), which is open access.
- Ray, Mondal & Reuveni, drift-diffusion restart.
- Koren et al., Lévy leapover.
- Richardson & Swarbrick 2010.
- Lindner 2004, colored-noise interspike intervals.

**MEMORY only (open copies to be sought).**

| Area | Papers |
|---|---|
| Escape over barriers | Hänggi-Talkner-Borkovec, RMP 62, 251 (1990): the finite-barrier correction (4.57), the spatial-diffusion and energy-diffusion rates, Mel'nikov-Meshkov, the cusp barrier. The scan is in the Augsburg repository, opus4 file 41911, which did not answer this session's probe; the earlier note says it was opened. |
| Driven escape and switching | Friddle-Noy-De Yoreo, PNAS 109, 13573 (2012); Cossio-Hummer-Szabo, Biophys. J. 111, 832 (2016); Hummer-Szabo, Biophys. J. 85, 5 (2003); Gammaitoni et al., RMP 70, 223 (1998) |
| Diffusion to targets | Berg-Purcell, Biophys. J. 20, 193 (1977); Zwanzig, PNAS 85, 2029 (1988) |
| Neurons | Fourcaud-Trocmé et al., J. Neurosci. 23, 11628 (2003) |

**Garg, PRB 51, 15592 (1995), doi:10.1103/PhysRevB.51.15592.** NOT OPENED; there is no
arXiv version. The earlier research note (`software_build/research/2026-10-14_sources_A_escape_ramps.md`,
item 6) found that the open restatement (Wernsdorfer, arXiv:cond-mat/0101104, Eq. 3.9)
misses the computed mean by 5-11%. That is a computed check, recorded there. Do not
list Garg's mean and width until the original is read.

## Sources of the list

- **Areas and quotas.** The maintainer's brief of 14 October 2026.
- **Exclusions.** The handbook's entries K0001-K0030 and `RESERVED_FOR_AUTHOR.md`.
- **Leads and known traps.** The research notes `2026-10-14_sources_A_escape_ramps.md`
  and `2026-10-14_sources_B_targets_resetting_transport.md` in
  `~/Desktop/geoews/software_build/research/`.
- **Finding papers.** arXiv API title search (`export.arxiv.org/api/query`).
- **Reading equations.** ar5iv HTML (LaTeX alttext), and arXiv PDF page images read
  through the fetch tool.
- **Metadata.** Crossref REST API, with no mailto parameter.
- **Sanity tests.** Python with numpy 2.0.2 and scipy 1.13.1, run inline. No files
  were kept.
