# Reserved for the founding author

These parts are designed and written by Alexander Sokol himself. An AI agent may
specify them (as below) and review them, and may not write them. When one is done, it
is recorded in AUTHORSHIP.md with the date.

## 1. The next entry: first passage of Brownian motion to a point

**What.** The first-passage-time density of Brownian motion with diffusion D, started
at distance x0 from an absorbing point, with drift v towards it (the inverse Gaussian)
and without drift (the Levy-Smirnov density); its survival probability; its mean x0/v
and variance 2 D x0 / v^3.

**Sources, already opened by a research agent on 14 October 2026** (its full report is
in the author's build notes):

- P. L. Krapivsky and S. Redner, J. Stat. Mech. (2018) 093208, doi
  10.1088/1742-5468/aaddb3, Eq. (1) p. 3 (journal version, NSF public-access copy):
  F(t) = l / sqrt(4 pi D t^3) exp(-(l - |v| t)^2 / 4 D t), for v < 0 (towards the
  target). Mean and cumulants below their Eq. (2b).
- S. Redner, Physica A 631, 128545 (2023), doi 10.1016/j.physa.2023.128545, Eq. (3.2),
  p. 3: the zero-drift density x0 / sqrt(4 pi D t^3) exp(-x0^2 / 4 D t). (A stray
  "t -> infinity" after it is a typo; the formula holds for all t.)
- A. J. Bray, S. N. Majumdar and G. Schehr, Adv. Phys. 62, 225 (2013), Eq. (30)
  (arXiv numbering): survival erf(x / sqrt(4 D t)).

**Function to write.** `first_passage_to_point(t, x0, D, v=0.0)` returning the
density; raise `OutsideDomain` for x0 <= 0 or D <= 0; for v away from the target
(v < 0 in your own convention, choose and document it) the density is defective:
document its total mass exp(-v x0 / D) (Krapivsky and Redner Eq. 3).

**Checks to write.** (1) Against the time-dependent Fokker-Planck solver
(`kramers.solvers.fokker_planck.solve` with an absorbing end at 0 and a far reflecting
wall), relative tolerance 5e-4 for t between 1 and 16 (see
`tests/test_fokker_planck.py` for the set-up). (2) Mean and variance against the
quadrature solver (`FirstPassage1D` with `drift`, natural far boundary). (3) The total
mass exp(-v x0 / D) for drift away from the target, against numerical integration.

**Time.** About an hour; follow CONTRIBUTING.md.

## 2. The optical-trap conversion and its test

A small module that converts a bead in a cubic optical potential (stiffness of the
cubic term in kT per cubic micrometre, bead diameter, viscosity, temperature, a
ramped tilt force) into the scaled units of the noisy fold, and back; with one
known-answer test that you choose and justify. The conversions to derive and check: for
drag gamma, cubic stiffness kappa and tilt force f, the scaled distance is
u = f / ((kT)^(2/3) kappa^(1/3)) and the scaled ramp rate gamma (df/dt) / (kT kappa).

## 3. One known-answer test per solver

For each of `quadrature`, `fokker_planck`, `simulate` and `laplace`: one test of your
own choosing, against a result you derive or look up yourself, in
`tests/test_known_answers.py`.

## 4. Narrative text

The tutorial texts and any software paper.
