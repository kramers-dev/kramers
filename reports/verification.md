# Verification report

Generated 2026-10-25T11:53:51+00:00 with kramers 0.1.0.dev0, Python 3.12.13, NumPy 2.5.3, SciPy 1.18.1.

**66 of 66 checks passed** across 30 entries (including the slow simulation checks; 289.9 s).

| Entry | Check | Reference | Worst | Tolerance | Result |
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

The tables behind each line are in `verification.json`.
