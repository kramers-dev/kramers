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
- After the first independent review (14 October): the panel search starts from 32
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
- After the second round of that review (15 October): panels resolve the speed density
  as well as phi when D varies, and D <= 0 or non-finite is refused; the scan beyond a
  cut is geometric, out to 1e8 times the cut's distance; a natural cut is moved past
  any rebound of the weight instead of being refused (periodic tails work); the
  decay-rate error now includes how far the root can move when the coefficients carry
  their own errors, which matters when two wells have nearly equal rates (the accuracy
  is then about 1e-6, and the error says so); moments, splitting probabilities and
  conditional times carry a floor of a few units in the last place of their logarithm;
  Newton's method stops at the rounding floor; K0012/K0013 no longer overflow at
  extreme drift; a domain that would need more than 200,000 panels is refused at once.
- After the third round (15 October): the decay rate is bracketed, not given a
  heuristic error: below by the first zero of the smallest series the coefficient
  intervals allow, above by the first zero of the largest or by a Rayleigh quotient,
  and the reported value lies in the bracket; a natural tail must also have a
  vanishing mass per decade (a tail that levels off is refused); a weight still
  unresolved at the width of rounding is treated as a singularity and refused (a zero
  of D between sample points); the early panel-count refusal accounts for bisection.
