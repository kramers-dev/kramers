# kramers

**Escape and first-passage results you can run and trust.**

kramers is an open handbook of first-passage and escape results in statistical
physics. Every result is a Python function, tied to the paper and equation that print
it, and checked against an independent solver every time the code changes.

```python
import kramers
from kramers.formulas import passage_time_saddle_node, kramers_rate_overdamped

kramers.search("resetting")            # find results
e = kramers.get("K0004")               # one entry: its sources, validity and checks
print(e.cite())                        # what to cite: the original paper first
e.verify()                             # run its checks yourself
```

## Sixty seconds

```text
$ python examples/quickstart.py
```

It recomputes, in a few seconds, numbers printed in four papers (a neuron's firing
rate at threshold, the Airy-function passage time through a noisy saddle-node, the
optimal resetting rate, the Sparre Andersen survival law) and shows how far Kramers'
formula is from the exact escape rate at 5, 10 and 20 kT.

## What is inside

- **The handbook.** One entry per published result: the function, the source that
  prints it (with the equation number, the page and the version that was read), where
  it holds, and the checks it passes. Thirty entries today, in seven areas: escape over
  a barrier, switching under a ramp, diffusion to a target, stochastic resetting,
  active and anomalous motion, transport in periodic potentials, neuron models.
- **Reference solvers**, the independent computations the checks rest on:
  - exact first-passage moments, splitting probabilities and decay rates of any
    one-dimensional diffusion, by quadrature in logarithms, at any barrier height
    (rates down to 1e-300), including drifts that reach infinity in finite time;
  - the time-dependent Fokker-Planck equation, for distributions and for ramped
    (time-dependent) drifts;
  - simulation with boundary-crossing correction and resetting;
  - numerical Laplace inversion.
- **Validity maps.** For approximate results, the error against the exact answer
  across the parameter range: Kramers' rate, for example, is 3% to 6% too high at a
  10 kT barrier, depending on the shape of the well.
- **Errata.** Implementing and checking published equations finds misprints. They are
  listed, with the numerical evidence, in [docs/errata.md](docs/errata.md).

## The entries

<!-- entries:start -->
| Entry | Result | Kind | Primary source |
|---|---|---|---|
| [K0001](docs/handbook/K0001.md) | Kramers' escape rate, overdamped (Smoluchowski) limit | asymptotic | P. Haenggi, P. Talkner and M. Borkovec, Rev. Mod. Phys. 62, 251 (1990) |
| [K0002](docs/handbook/K0002.md) | Exact mean first-passage time in one dimension (quadrature formula) | exact | P. Haenggi, P. Talkner and M. Borkovec, Rev. Mod. Phys. 62, 251 (1990) |
| [K0004](docs/handbook/K0004.md) | Mean passage time through a noisy saddle-node, closed form in Airy functions | exact | D. Hathcock and J. P. Sethna, Phys. Rev. Research 3, 013156 (2021) |
| [K0005](docs/handbook/K0005.md) | Noisy saddle-node exactly at the bifurcation: mean, variance and CV of the passage time | exact | B. Lindner, A. Longtin and A. Bulsara, Neural Comput. 15, 1761 (2003) |
| [K0006](docs/handbook/K0006.md) | Escape rate near a saddle-node, weak noise (Kramers limit) | asymptotic | B. Lindner, A. Longtin and A. Bulsara, Neural Comput. 15, 1761 (2003) |
| [K0008](docs/handbook/K0008.md) | Decay rate of the noisy fold at any barrier height (computed reference) | computed-reference | computed here |
| [K0024](docs/handbook/K0024.md) | Distribution of the switching point under a ramp (quasi-static approximation) | exact | O. K. Dudko, G. Hummer and A. Szabo, Phys. Rev. Lett. 96, 108101 (2006) |
| [K0025](docs/handbook/K0025.md) | Escape rates from a switching histogram (Fulton-Dunkleberger transformation) | estimator | J. N. Kaemmerer, S. Masis, K. Hambardzumyan, P. Lenhard, U. Strobel, J. Lisenfeld, H. Rotzinger, A. V. Ustinov et al., arXiv:2411.15048 (2024) |
| [K0026](docs/handbook/K0026.md) | Force-dependent escape rate of Dudko, Hummer and Szabo | asymptotic | O. K. Dudko, G. Hummer and A. Szabo, Phys. Rev. Lett. 96, 108101 (2006) |
| [K0027](docs/handbook/K0027.md) | Mean and variance of the rupture force under constant loading (Dudko, Hummer and Szabo) | asymptotic | O. K. Dudko, G. Hummer and A. Szabo, Phys. Rev. Lett. 96, 108101 (2006) |
| [K0028](docs/handbook/K0028.md) | Force-dependent lifetimes from rupture-force histograms (Dudko, Hummer and Szabo, 2008) | estimator | O. K. Dudko, G. Hummer and A. Szabo, Proc. Natl. Acad. Sci. USA 105, 15755 (2008) |
| [K0029](docs/handbook/K0029.md) | Most probable rupture force under constant loading (Bell-Evans) | exact | Z. Liu, H. Liu, A. M. Vera, R. C. Bernardi, P. Tinnefeld and M. A. Nash, Nat. Commun. 11, 4321 (2020) |
| [K0030](docs/handbook/K0030.md) | Switching of the noisy fold under a linear ramp, solved exactly (computed reference) | computed-reference | computed here |
| [K0003](docs/handbook/K0003.md) | Splitting probability between two absorbing ends in one dimension | exact | L. Dagdug and A. M. Berezhkovskii, J. Chem. Phys. 131, 056101 (2009) |
| [K0012](docs/handbook/K0012.md) | Drift-diffusion on an interval: exit probability, mean and conditional exit times | exact | S. Redner, Physica A 631, 128545 (2023) |
| [K0013](docs/handbook/K0013.md) | First-passage duality: conditional exit times do not depend on the sign of the drift | exact | P. L. Krapivsky and S. Redner, J. Stat. Mech. (2018) 093208 |
| [K0014](docs/handbook/K0014.md) | Exit time of Brownian motion from a ball in d dimensions: mean and second moment | exact | H. Chen and F. Huang, Phys. Rev. E 105, 034109 (2022) |
| [K0015](docs/handbook/K0015.md) | Diffusion with stochastic resetting: mean time to reach the target | exact | M. R. Evans and S. N. Majumdar, Phys. Rev. Lett. 106, 160601 (2011) |
| [K0016](docs/handbook/K0016.md) | Optimal resetting rate for diffusion to a target | exact | M. R. Evans and S. N. Majumdar, Phys. Rev. Lett. 106, 160601 (2011) |
| [K0017](docs/handbook/K0017.md) | Any first-passage process under Poissonian restart: mean and second moment | exact | S. Reuveni, Phys. Rev. Lett. 116, 170601 (2016) |
| [K0018](docs/handbook/K0018.md) | Restart helps at small rates exactly when the coefficient of variation exceeds one | exact | A. Pal and S. Reuveni, Phys. Rev. Lett. 118, 030603 (2017) |
| [K0019](docs/handbook/K0019.md) | Mean completion time under sharp (periodic) restart | exact | A. Pal, A. Kundu and M. R. Evans, J. Phys. A 49, 225001 (2016) |
| [K0020](docs/handbook/K0020.md) | Sparre Andersen theorem: universal survival probability of a symmetric random walk | exact | A. J. Bray, S. N. Majumdar and G. Schehr, Adv. Phys. 62, 225 (2013) |
| [K0023](docs/handbook/K0023.md) | Mean exit time of a run-and-tumble particle from an interval | exact | L. Angelani, R. Di Leonardo and M. Paoluzzi, Eur. Phys. J. E 37, 59 (2014) |
| [K0021](docs/handbook/K0021.md) | Mean velocity in a tilted periodic potential (Stratonovich's formula) | exact | P. Reimann, C. Van den Broeck, H. Linke, P. Haenggi, J. M. Rubi and A. Perez-Madrid, Phys. Rev. Lett. 87, 010602 (2001) |
| [K0022](docs/handbook/K0022.md) | Effective diffusion in a tilted periodic potential | exact | P. Reimann, C. Van den Broeck, H. Linke, P. Haenggi, J. M. Rubi and A. Perez-Madrid, Phys. Rev. Lett. 87, 010602 (2001) |
| [K0007](docs/handbook/K0007.md) | Rate near a saddle-node to first order in the bias | asymptotic | B. Lindner, A. Longtin and A. Bulsara, Neural Comput. 15, 1761 (2003) |
| [K0009](docs/handbook/K0009.md) | Firing rate of the quadratic integrate-and-fire neuron with white noise | exact | N. Brunel and P. E. Latham, Neural Comput. 15, 2281 (2003) |
| [K0010](docs/handbook/K0010.md) | Firing rate of the leaky integrate-and-fire neuron with white noise (Siegert formula) | exact | N. Brunel, J. Comput. Neurosci. 8, 183 (2000) |
| [K0011](docs/handbook/K0011.md) | Variance of the passage time through a noisy saddle-node, any bias and thresholds | exact | B. Lindner, A. Longtin and A. Bulsara, Neural Comput. 15, 1761 (2003) |
<!-- entries:end -->

## Take part

| You have | You can |
|---|---|
| Ten minutes and a paper of your own in the handbook | Check its entry: is the equation, the range of validity, the citation right? |
| An hour | Add an entry from [the open list](OPEN_ENTRIES.md); the paper, the equation and the check are named |
| An experiment | Propose a case: one published measurement, recomputed |
| A course | Use entries as self-checking exercises |

How, and how credit works: [CONTRIBUTING.md](CONTRIBUTING.md). Everyone who
contributes is named on the release. The original paper is always cited first.

## How it is made

The code and the text are written largely with AI tools (Claude, by Anthropic),
directed by a person who signs every release, and reviewed by a second AI model. That
is why nothing here asks to be trusted: every equation's source was opened and its
location recorded, and no entry is accepted unless an independent numerical check
passes. Details, and what we will and will not do: [HOW_THIS_IS_MADE.md](HOW_THIS_IS_MADE.md).

## Status

Early (0.1.0.dev0), not yet released. The verification report of the latest full run
is in [reports/verification.md](reports/verification.md). Nothing here is new
physics: each page gives the history of its result.

## Install

```bash
pip install -e .            # from a clone; NumPy and SciPy are the only dependencies
python -m pytest            # the fast checks
python tools/verify_all.py --slow   # everything, including simulations
```

MIT licence.
