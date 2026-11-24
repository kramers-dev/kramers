# Verification report

Generated 2026-11-24T09:13:59+00:00 with kramers 0.1.0, Python 3.12.13, NumPy 2.5.5, SciPy 1.18.1.

**118 of 118 checks passed** across 43 entries and 5 validity maps (including the slow simulation checks; 1008.2 s).

| Entry or map | Check | Reference | Worst | Tolerance | Result |
|---|---|---|---|---|---|
| K0001 | converges to the exact decay rate like 1/barrier | smallest eigenvalue, quadrature solver | 8.61e-01 | 1e+00 | pass |
| K0001 | the stored validity map is current | the same computation | 1.70e-03 | 1e-02 | pass |
| K0002 | closed forms: flat potential and constant drift | Redner (2023) Eqs. (4.16); Srivastava et al. (2017) Eq. (7) | 6.66e-16 | 1e-12 | pass |
| K0002 | a non-flat potential against the time-dependent solver | Fokker-Planck solver, integral of the survival | 4.33e-06 | 2e-05 | pass |
| K0002 | double well against simulation | Euler-Maruyama, 20,000 paths | 9.88e-02 | 1e+00 | pass |
| K0003 | constant force against the printed closed form | Redner (2023), Eq. (4.12) | 3.11e-15 | 1e-11 | pass |
| K0003 | position-dependent D against adaptive quadrature | scipy.integrate.quad of the printed integrals | 2.22e-16 | 1e-10 | pass |
| K0003 | double well against simulation | Euler-Maruyama simulation, 40,000 paths | 1.64e+00 | 4e+00 | pass |
| K0004 | against the quadrature solver | Green's-function quadrature with entrance and exit tails | 3.15e-14 | 1e-11 | pass |
| K0004 | at the bifurcation, against another paper | Lindner et al. (2003), Eq. (3.4) | 4.44e-16 | 1e-13 | pass |
| K0004 | the printed asymptotic limits | Hathcock and Sethna, Eqs. (8)-(9) | 7.31e-03 | 1e-02 | pass |
| K0005 | mean and variance against the quadrature solver | quadrature with entrance and exit tails | 3.26e-14 | 1e-11 | pass |
| K0005 | the numbers printed in the paper | Lindner et al. (2003), p. 1771 | 0.00e+00 | 1e-12 | pass |
| K0006 | approaches the exact rate as 1/barrier | exact passage time in Airy functions (K0004) | 8.31e-01 | 1e+00 | pass |
| K0006 | transcription | Eq. (3.10) evaluated by hand | 0.00e+00 | 1e-15 | pass |
| K0007 | value and slope at beta = 0 against the exact rate | exact rate in Airy functions (K0004) | 7.41e-10 | 1e-07 | pass |
| K0007 | the printed decimals | Lindner et al. (2003), Eq. (3.11) | 0.00e+00 | 1e-12 | pass |
| K0008 | against 65-digit Hermite-basis values | independent high-precision computation | 2.15e-14 | 1e-12 | pass |
| K0008 | high barriers: equals the reciprocal mean passage time | Airy closed form (K0004) | 1.65e-13 | 1e-12 | pass |
| K0008 | slope of the log-survival | time-dependent Fokker-Planck solver | 9.21e-07 | 2e-03 | pass |
| K0009 | against the quadrature solver | Green's-function quadrature, entrance and exit tails | 1.41e-14 | 1e-10 | pass |
| K0009 | the printed limits sigma = 0 and mu = 0 | Brunel and Latham (2003), pp. 2288-2289 | 1.11e-16 | 1e-11 | pass |
| K0010 | against the quadrature solver | Ornstein-Uhlenbeck mean first-passage time by quadrature | 5.77e-15 | 1e-10 | pass |
| K0010 | against Brunel and Hakim's alternative integral | Brunel and Hakim (1999), Eq. (14) | 2.22e-16 | 1e-09 | pass |
| K0010 | against simulation | Euler-Maruyama, 20,000 interspike intervals | 5.96e-01 | 4e+00 | pass |
| K0011 | against Eq. (A.4) evaluated directly | independent triple integral in log space | 2.00e-15 | 1e-09 | pass |
| K0011 | at the bifurcation point | Lindner et al. Eq. (3.5) | 1.33e-15 | 1e-11 | pass |
| K0012 | all four quantities against the quadrature solver | Green's-function quadrature | 1.67e-14 | 1e-09 | pass |
| K0012 | against the formulas as printed | Srivastava et al. (2017) | 8.88e-16 | 1e-13 | pass |
| K0013 | printed Eq. (28b) against K0012 for v and -v | Srivastava et al. formulas (K0012) | 2.33e-15 | 1e-11 | pass |
| K0013 | no loss of digits at tiny and large drift | v -> 0 limit and sign symmetry | 1.11e-16 | 1e-09 | pass |
| K0013 | whole distributions at each end unchanged by v -> -v | time-dependent Fokker-Planck solver | 1.22e-06 | 2e-03 | pass |
| K0013 | midpoint: the two ends coincide; off-centre they differ | time-dependent Fokker-Planck solver | 1.48e-03 | 1e+00 | pass |
| K0014 | against the radial process by quadrature, d = 1, 2, 3, 5 | Green's-function quadrature with psi = r^(d-1) | 3.33e-15 | 1e-07 | pass |
| K0014 | against simulation in three dimensions | random walk with bridge correction, 20,000 paths, dt = 1e-4 | 1.72e+00 | 4e+00 | pass |
| K0015 | against the general renewal formula | Reuveni (2016) Eq. (7), entry K0017 | 2.22e-16 | 1e-13 | pass |
| K0015 | against simulation | Euler-Maruyama with Poissonian resetting, 40,000 paths | 6.42e-01 | 4e+00 | pass |
| K0016 | is the minimum of the mean time | numerical minimisation of K0015 | 1.37e-08 | 1e-06 | pass |
| K0016 | the printed digits | Evans and Majumdar; Pal, Kundu and Evans | 0.00e+00 | 1e-12 | pass |
| K0017 | free diffusion reproduces Evans and Majumdar | K0015 | 1.11e-16 | 1e-12 | pass |
| K0017 | with a transform computed numerically | first-passage density from the Fokker-Planck solver | 3.17e-05 | 1e-04 | pass |
| K0017 | drift-diffusion against simulation | Euler-Maruyama with resetting, 40,000 paths | 5.00e-01 | 4e+00 | pass |
| K0018 | slope and sign against the derivative of the renewal formula | K0017 with the inverse-Gaussian transform | 1.26e-02 | 1e+00 | pass |
| K0019 | the optimality condition at the best period | Pal, Kundu and Evans Eq. (66) | 5.69e-13 | 1e-06 | pass |
| K0019 | free diffusion against simulation | Euler-Maruyama with periodic resetting, 40,000 paths | 6.04e-01 | 4e+00 | pass |
| K0020 | generating function and asymptotics | Bray, Majumdar and Schehr Eqs. (13), (15) | 1.22e-01 | 1e+00 | pass |
| K0020 | four jump laws against simulation | 200,000 walks each: Gaussian, uniform, Cauchy, Laplace | 1.12e+00 | 4e+00 | pass |
| K0021 | no potential: F / eta | free drift | 4.44e-16 | 1e-12 | pass |
| K0021 | against L over the mean time per period | quadrature solver, natural boundary behind the tilt | 5.55e-16 | 1e-09 | pass |
| K0022 | no potential: D0 | free diffusion | 9.99e-16 | 1e-11 | pass |
| K0022 | against the passage-time moments over one period | quadrature solver, Eq. (5) | 7.88e-15 | 1e-08 | pass |
| K0023 | against event-driven simulation | exact piecewise-deterministic simulation, 100,000 particles | 5.85e-01 | 4e+00 | pass |
| K0024 | Bell rate: closed form | integral of an exponential | 1.33e-15 | 1e-06 | pass |
| K0024 | against the density printed for the Dudko-Hummer-Szabo model | Dudko, Hummer and Szabo (2006), Eq. (4) | 2.63e-14 | 1e-06 | pass |
| K0024 | mean and variance for an exponential rate | Gumbel law | 4.79e-10 | 1e-09 | pass |
| K0025 | recovers a known rate from 200,000 synthetic switches | exact sampling from the survival law | 2.57e+00 | 4e+00 | pass |
| K0026 | nu = 2/3 equals Kramers' formula for the tilted surface | K0001 | 1.89e-15 | 1e-12 | pass |
| K0026 | force dependence against exact decay rates | quadrature solver, cubic and cusp landscapes | 4.86e-01 | 1e+00 | pass |
| K0027 | against the exact moments of the switching distribution | K0024 with K0026 | 1.12e-02 | 2e-02 | pass |
| K0028 | on an exact histogram it equals its known discretisation | (dF/Fdot) / (2 tanh(lambda/2)) | 5.48e-14 | 1e-09 | pass |
| K0028 | recovers a known lifetime from 200,000 synthetic ruptures | exact sampling from the Bell rate | 2.16e+00 | 4e+00 | pass |
| K0029 | is the mode of the exact rupture-force density | K0024 with the Bell rate | 0.00e+00 | 2e-04 | pass |
| K0029 | the nu = 1 limit of the mean-force formula | Dudko, Hummer and Szabo (2006) Eq. (5) | 6.66e-16 | 1e-12 | pass |
| K0030 | grid and time-step convergence | the same computation with twice the cells and steps | 4.87e-06 | 1e-03 | pass |
| K0030 | quantiles against direct simulation | Euler-Maruyama, 6,000 paths, the same escape criterion | 1.66e+00 | 4e+00 | pass |
| K0030 | slow ramp: approaches the quasi-static median | decay rate of the frozen fold (K0008) | 2.25e-03 | 1e-02 | pass |
| K0031 | d = 1 against the one-dimensional result | entry K0015 at distance R - a | 0.00e+00 | 1e-12 | pass |
| K0031 | d = 3 against the renewal formula | Reuveni (2016) Eq. (7) with the elementary d = 3 transform (entry K0017) | 6.66e-16 | 1e-12 | pass |
| K0031 | against simulation | Euler-Maruyama for the radial (Bessel) process with Poissonian resetting, 40,000 paths | 3.71e-01 | 4e+00 | pass |
| K0032 | against the renewal formula | Reuveni (2016) Eq. (7) with the interval's Laplace transform (entry K0017) | 8.88e-16 | 1e-12 | pass |
| K0032 | the limit without resetting | entry K0012 with zero drift | 8.45e-11 | 1e-06 | pass |
| K0032 | against simulation | Euler-Maruyama with Poissonian resetting and crossing correction, 100,000 paths | 7.90e-01 | 4e+00 | pass |
| K0033 | S(0) = 1/2 and the large-time law | the limits stated with Eq. (1) | 1.92e-02 | 1e+00 | pass |
| K0033 | against event-driven simulation | exact piecewise-straight runs in d = 1, 2, 3, 200,000 particles each | 1.82e+00 | 4e+00 | pass |
| K0034 | against the renewal formula | Reuveni (2016) Eq. (7) with the drift-diffusion transform (entry K0017) | 1.39e-09 | 1e-08 | pass |
| K0034 | the limits V = 0 and r -> 0 | entry K0015; the quadrature solver without restart | 6.92e-04 | 1e+00 | pass |
| K0034 | the optimal rate and the transition at Pe = 1 | numerical minimisation of the mean | 1.07e-02 | 1e+00 | pass |
| K0034 | against simulation | Euler-Maruyama with Poissonian restart, 100,000 paths, drift towards and away | 2.07e+00 | 4e+00 | pass |
| K0035 | against the renewal formula | Reuveni (2016) Eq. (7) with the transform of Eq. (25) (entry K0017) | 2.48e-09 | 1e-08 | pass |
| K0035 | free diffusion | entry K0015 at beta U0 = 0 | 4.44e-16 | 1e-12 | pass |
| K0035 | the limit without resetting | quadrature solver with the logarithmic potential | 2.40e-01 | 1e+00 | pass |
| K0035 | the optimal rate and the transition at beta U0 = 5 | numerical minimisation; entry K0016's condition | 1.01e-02 | 1e+00 | pass |
| K0035 | against simulation | Euler-Maruyama with the 1/x drift (attractive), and exact event-driven sampling of the Bessel hitting law with exponential resets (repulsive and weakly attractive) | 6.91e-01 | 4e+00 | pass |
| K0036 | against the renewal formula | Reuveni (2016) Eq. (7) with the wall transform (entry K0017) | 2.22e-15 | 1e-12 | pass |
| K0036 | no resetting, and the critical start point | quadrature solver: mean, variance, and variance = mean^2 | 6.70e-03 | 1e+00 | pass |
| K0036 | the optimal rate | numerical minimisation of the mean | 1.93e-02 | 1e+00 | pass |
| K0036 | against simulation | Euler-Maruyama with a reflecting wall and resetting, 100,000 paths | 1.80e+00 | 4e+00 | pass |
| K0037 | two printed forms | Ahmad et al. Eqs. (21)-(22) and Dubey and Pal Eq. (41), evaluated as printed | 2.00e-15 | 1e-10 | pass |
| K0037 | the limits r -> 0 and kappa -> 0 | quadrature solver without resetting; entry K0015 | 7.93e-02 | 1e+00 | pass |
| K0037 | the transition where resetting stops helping | quadrature solver: spread = mean; both papers' printed values | 4.31e-01 | 1e+00 | pass |
| K0037 | against simulation | Euler-Maruyama with resetting, 100,000 paths | 2.66e+00 | 4e+00 | pass |
| K0038 | against the time-domain formula | Eq. (36) by quadrature, no transforms | 8.88e-16 | 1e-08 | pass |
| K0038 | without a refractory period | entry K0015 | 2.22e-16 | 1e-12 | pass |
| K0038 | against exact event-driven sampling | 200,000 paths each; exponential and fixed periods; independent and correlated cases | 7.44e-01 | 4e+00 | pass |
| K0039 | the Brownian case and the uniform-start average | entry K0012 at alpha = 2; quadrature of Eq. (8) | 2.22e-04 | 1e+00 | pass |
| K0039 | against simulation | alpha-stable steps by the Chambers-Mallows-Stuck method, 100,000 paths; the step bias shown falling | 1.86e-01 | 1e+00 | pass |
| K0040 | against the general formula | Eq. (6) by quadrature with the propagator of Eq. (8) | 4.44e-16 | 1e-09 | pass |
| K0040 | instant returns, and the optimal rate | entry K0015; numerical minimisation | 1.75e-02 | 1e+00 | pass |
| K0040 | against simulation | Euler-Maruyama with a bridge-corrected target and timed returns home, 100,000 paths | 7.91e-01 | 4e+00 | pass |
| K0041 | two printed forms | the Bessel series Eq. (5), acute and obtuse wedges | 1.40e-14 | 1e-10 | pass |
| K0041 | right angle and half-plane | product of two half-line survivals; erf of the distance to a line | 2.22e-16 | 1e-11 | pass |
| K0041 | against simulation | two-dimensional Euler-Maruyama with a bridge correction on each edge, 100,000 paths | 5.16e-01 | 4e+00 | pass |
| K0042 | against the printed Laplace transforms | Eq. (32), with the atom for a start towards the origin | 2.22e-16 | 1e-09 | pass |
| K0042 | total probability and the long-time tail | 1; Eq. (36) | 1.10e-01 | 1e+00 | pass |
| K0042 | against exact event-driven simulation | piecewise-linear runs, 200,000 particles per case | 2.79e+00 | 4e+00 | pass |
| K0043 | the hypergeometric series | mpmath's 2F1 at 30 digits | 3.66e-15 | 1e-12 | pass |
| K0043 | limits of the decay exponent and of the transform | Eq. (90); theta = 1/2 at gamma = mu = 1; s -> 0 | 5.13e-01 | 1e+00 | pass |
| K0043 | against exact event-driven simulation | piecewise-exponential runs, 200,000 to 400,000 particles; transform, mean and decay exponent | 1.45e+00 | 4e+00 | pass |
| M01 | the claim follows from the stored data | the stored grid | 0.00e+00 | 0e+00 | pass |
| M01 | the stored map is current | the same computation, at sample points | 0.00e+00 | 1e-06 | pass |
| M02 | the claim follows from the stored data | the stored grid | 0.00e+00 | 0e+00 | pass |
| M02 | the stored map is current | the same computation, at sample points | 0.00e+00 | 1e-06 | pass |
| M03 | the claim follows from the stored data | the stored grid | 0.00e+00 | 0e+00 | pass |
| M03 | the stored map is current | the same computation, at sample points | 0.00e+00 | 1e-06 | pass |
| M04 | the claim follows from the stored data | the stored grid | 0.00e+00 | 0e+00 | pass |
| M04 | the stored map is current | the same computation, at sample points | 0.00e+00 | 1e-06 | pass |
| M05 | the claim follows from the stored data | the stored grid | 0.00e+00 | 0e+00 | pass |
| M05 | the stored map is current | the same computation, at sample points | 0.00e+00 | 1e-06 | pass |

The tables behind each line are in `verification.json`.
