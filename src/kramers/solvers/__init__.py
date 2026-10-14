"""Reference solvers that the handbook's checks rest on.

- :mod:`quadrature`: exact first-passage moments, splitting probabilities and decay
  rates of one-dimensional diffusions, at any barrier height.
- :mod:`fokker_planck`: the time-dependent Fokker-Planck equation, for distributions
  and for ramped (time-dependent) drifts.
- :mod:`simulate`: direct simulation, with statistical error bars.
- :mod:`laplace`: numerical inversion of Laplace transforms.
"""

from . import fokker_planck, laplace, quadrature, simulate
from .quadrature import FirstPassage1D

__all__ = ["FirstPassage1D", "fokker_planck", "laplace", "quadrature", "simulate"]
