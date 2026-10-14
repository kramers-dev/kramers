# Changelog

## Unreleased (0.1.0.dev0)

- Reference solvers: quadrature (first-passage moments, splitting probabilities and
  decay rates at any barrier height, with entrance and exit tails), time-dependent
  Fokker-Planck (Scharfetter-Gummel, TR-BDF2, exact discrete balance), simulation with
  bridge correction and resetting, fixed-Talbot Laplace inversion.
- The registry: one published result per entry, with sources, validity and checks;
  generated pages, a data file and a verification report.
- Entries K0001 to K0030 in seven areas.
- Questions for authors (`docs/errata.md`): three printed equations that disagree with
  our computations, each with the computation, listed as questions until the authors
  answer.
- After the second model's first review (14 October): the panel search starts from 32
  equal panels plus every local extremum found on 8192 points, and stops refining at the
  rounding floor of the samples; the error estimate uses bisected panels at a higher
  order; natural ends are judged on the speed density, so a position-dependent D is
  handled, and exit ends refuse it; the region beyond each cut is scanned for a hidden
  well or barrier; the decay rate is the smallest root of the characteristic function
  (Taylor coefficients from iterated positive integrals, Newton's method from zero),
  which converges in milliseconds where power iteration needed 10,000 steps; the
  Fokker-Planck documentation no longer claims positivity; starts on an absorbing end
  are refused; K0013 is stable at any drift; K0024 refuses an incomplete range unless
  asked; K0027 refuses slow loading; K0030 starts from the quasi-stationary density and
  has a convergence check. Each failing input from the review is a test
  (`tests/test_review_cases.py`).
