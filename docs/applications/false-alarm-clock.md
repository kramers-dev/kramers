# The false-alarm clock

*How often does a threshold alarm fire when nothing is happening?*

Many monitoring rules raise an alarm when an indicator rises above a threshold: a drift detector, an anomaly score, an
early-warning indicator, a control chart. The number that matters to whoever acts on the alarm is how long the rule
runs, on average, before it fires on noise alone. If the indicator fluctuates around its normal level like an
Ornstein–Uhlenbeck process, that expected time is a mean first-passage time, and the handbook already holds it exactly.

This page adds no new result. It reads entry [K0010](../handbook/K0010.md) (the Siegert formula, published for the
firing rate of a noisy neuron) as a statement about alarms. Entry [K0002](../handbook/K0002.md) (the general
quadrature) gives the same numbers to about 1e-10. That figure comes from comparing the two on 13 November 2026, and
part of it is the rounding of the reference values.

## The mapping

An indicator with mean $m$, stationary standard deviation $s$ and correlation time $\tau$, modelled as

$$\tau\,\dot x = -(x-m) + \sqrt{2}\,s\,\sqrt{\tau}\,\eta(t),$$

is the membrane potential of K0010 with $\mu = m$ and $\sigma = \sqrt{2}\,s$. An alarm at $m + h\,s$ is the threshold,
and a start at the normal level is the reset $V_r = m$. The mean time to the first false alarm is then the
inter-spike interval:

$$T(h) = \tau\sqrt{\pi}\int_0^{h/\sqrt{2}} e^{u^2}\left(1+\operatorname{erf}u\right)du .$$

```python
import numpy as np
from kramers.formulas import lif_firing_rate

def false_alarm_time(h, tau=1.0):
    """Mean time to the first crossing of mean + h sd, from the mean, for an OU indicator with correlation time tau."""
    return 1.0 / lif_firing_rate(mu=0.0, sigma=np.sqrt(2.0), theta=h, V_r=0.0, tau=tau)

false_alarm_time(2.0, tau=30.0)   # 312.85...: a daily indicator with a 30-day memory alarms at 2 sd about every 313 days
```

## The numbers

| Threshold (standard deviations above normal) | Mean time to the first false alarm (correlation times) |
|---|---|
| 1.645 (one-sided 95th percentile) | 5.80 |
| 1.96 | 9.73 |
| 2.0 | 10.43 |
| 2.5 | 27.01 |
| 3.0 | 86.93 |
| 3.5 | 364.4 |
| 4.0 | 2,018 |

A threshold chosen as "two standard deviations" sounds strict. On a continuously monitored indicator it fires on
noise about every ten correlation times.

## What the numbers assume

- **An Ornstein–Uhlenbeck indicator.** The indicator relaxes to its normal level at a single rate, with Gaussian
  noise. Rolling-window statistics (a variance or an autocorrelation over a window) are not exactly of this kind.
  For them the table is a first estimate, and a simulation of the actual statistic is the check.
- **Continuous monitoring.** An indicator read at discrete times misses short excursions, so its false alarms come
  later than the table says.
- **A start at the normal level.** Starting from a value drawn from the stationary distribution shortens the times
  slightly (by about 5% at 2 sd and about 1% at 3 sd).
- **Known $m$, $s$ and $\tau$.** In practice they are estimated from a baseline, and the error in those estimates
  moves the clock.

## Why it matters

Choosing a threshold from the false-alarm rate you can tolerate, rather than from a conventional number of standard
deviations, turns an alarm into a statement with a known cost. The same first-passage reasoning gives the false-alarm
rate of rules that are not Ornstein–Uhlenbeck: the quadrature of K0002 accepts any one-dimensional drift and noise.
