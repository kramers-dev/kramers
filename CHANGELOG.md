# Changelog

## Unreleased

- Three entries from the open list, each with an exact cross-check and a simulation:
  K0031 (diffusion with resetting in d dimensions: mean time to reach an absorbing
  sphere; Evans and Majumdar 2014), K0032 (diffusion with resetting in an interval with
  two absorbing ends; Pal and Prasad 2019), K0033 (universal survival probability of a
  run-and-tumble particle in d dimensions; Mori, Le Doussal, Majumdar and Schehr 2020).
- Two more resetting entries, each read from the TeX source and PDF by two readers: K0034 (drift-diffusion to
  a target under restart: mean, spread and the Peclet transition; Ray, Mondal and Reuveni 2019) and K0035
  (resetting in a logarithmic potential: mean and spread; Ray and Reuveni 2020). K0035 keeps full accuracy
  at small resetting rates by writing the transform as an expectation over the Bessel hitting law. Its
  repulsive cases are checked by exact event-driven sampling: there, Euler-Maruyama's bias falls only like
  sqrt(dt).
- Five more entries, each read by two readers:
  - K0036: resetting between an absorbing end and a reflecting wall, with the critical start point
    1 - 1/sqrt 5 (Ahmad et al. 2019).
  - K0037: an Ornstein-Uhlenbeck process with resetting and the target at the trap centre. Ahmad et al.'s
    Kummer form and Dubey and Pal's parabolic-cylinder form agree to 2e-15. The critical stiffness computed
    here is 0.174898, against the printed 0.174895.
  - K0038: resetting with a refractory period (Evans and Majumdar 2019), checked against the paper's
    time-domain formula and exact event-driven sampling.
  - K0039: the mean exit time of a symmetric Levy flight from an interval (Getoor's formula as printed by
    Dybiec et al. 2017).
  - K0040: search with a return home at constant speed (Pal, Kusmierz and Reuveni 2020).
  - K0041: Brownian survival in a planar wedge of any angle (Chupeau, Benichou and Majumdar 2015). The closed
    form agrees with the Bessel series to 1e-14.
  - K0042: exact first-passage densities of a run-and-tumble particle on a half-line (Malakar et al. 2018).
  - K0043: a run-and-tumble particle in a harmonic trap, absorbed at the centre: survival transform, mean time
    and decay exponent (Dhar et al. 2019). Its hypergeometric function is summed directly, because
    scipy.special.hyp2f1 is wrong by thirty orders of magnitude at some of the parameters needed. For a start
    away from the centre the entry follows Eqs. (82) and (67); Eq. (85) as printed in arXiv v2 does not match
    them or the simulation, and is noted on the entry as a question.
- Documentation: "The false-alarm clock", reading K0010 (Siegert) as the mean time to a false alarm of a
  threshold alarm on an Ornstein-Uhlenbeck indicator, with a table and its assumptions. No new result.
- Questions for authors: the K0002 question now records that a clean copy of p. 291
  (read 28 October) confirms the printed sign.

## 0.1.0 (25 October 2026)

The first public version: on PyPI (`pip install kramers`), with the handbook at
https://kramers-dev.github.io/kramers/.

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
