"""A bead in a cubic optical potential: the Brno experiments, recomputed from the printed parameters.

Sources (open access or arXiv):
  P1  M. Siler, P. Jakl, O. Brzobohaty, A. Ryabov, R. Filip and P. Zemanek, Sci. Rep. 7, 1697 (2017)
  P2  M. Siler, L. Ornigotti, O. Brzobohaty, P. Jakl, A. Ryabov, V. Holubec, P. Zemanek and R. Filip,
      Phys. Rev. Lett. 121, 230601 (2018), arXiv:1803.07833v2 and its supplement (SM)
  P3  A. Ryabov, P. Zemanek and R. Filip, Phys. Rev. E 94, 042108 (2016)
Model (P1 Eq. 1, P2 Eq. 1): overdamped bead, V(x) = mu x^3 / 3, D = kT / gamma.
Numbers labelled "printed" or "read" come from the papers (figure readings to the stated
precision); everything else is computed here with kramers' quadrature and Fokker-Planck solvers.

    python cases/brno_cubic_trap/reproduce.py        (about two minutes; writes figure.png)
"""

from pathlib import Path

import numpy as np

from kramers.formulas import fold_decay_rate, ramped_fold_switching
from kramers.solvers import FirstPassage1D, fokker_planck

KT = 1.380649e-23 * 295.0  # J, room temperature assumed (not printed)


def moments(mu, x0, x_abs, D=1.0):
    """Mean and second moment of the time from x0 to x_abs < x0 in V = mu x^3/3 (natural boundary at +inf)."""
    fp = FirstPassage1D(lambda x: mu * x**3 / 3.0, D=D, domain=(x_abs, np.inf), left="absorbing", right="natural")
    return fp.moments(x0, n=2).value


def survival(mu, D, x0, x_abs, t_end):
    sol = fokker_planck.solve(lambda x: -D * mu * x * x, D, (x_abs, 3.0), left="absorbing", right="reflecting",
                              x0=x0, t_end=t_end, n_cells=2000, dt=t_end / 3000)
    return sol.t, sol.survival


# ---------------------------------------------------------------- P1, Sci. Rep. 2017
print("P1. x0 = 1.2 um (Fig. 4; Fig. 1 marks about 1.47), x1 = -2.1 um, mu3 = 6.65 +- 0.05 kT/um^3 (printed).")
print("    Measured (Fig. 4, read; N = 26): <tau(x0->0)> = 1.16 +- 0.14 s, <tau(x0->x1)> = 3.61 +- 0.29 s, R = 0.32 +- 0.05,")
print("    mean/SD of tau(x0->x1) = 2.3 +- 1.6; their simulation at these points: 1.62 (2.32 cut at 8 s).")
D_stokes = KT / (6 * np.pi * 0.95e-3 * 0.497e-6) * 1e12  # um^2/s: water at 22 C assumed; radius 0.497 um printed
print("\n    Reading A: mu3 is the potential itself; D is free.")
for x0 in (1.2, 1.47):
    t0, _ = moments(6.65, x0, 0.0)
    t1, s1 = moments(6.65, x0, -2.1)
    print(f"      x0 = {x0}: R = {t0 / t1:.3f}, mean/SD = {t1 / np.sqrt(s1 - t1 * t1):.3f}; the measured means need "
          f"D = {t0 / 1.16:.3f} +- {0.14 / 1.16 * t0 / 1.16:.3f} and {t1 / 3.61:.3f} +- {0.29 / 3.61 * t1 / 3.61:.3f} um^2/s "
          f"(a wall factor of {D_stokes / (t1 / 3.61):.1f} against the Stokes value {D_stokes:.3f})")
print("\n    Reading B (P1 Methods: the drift was measured near the wall, the wall factor gamma_c enters only the noise):")
print("    drift coefficient kappa = 6.65 x D_Stokes fixed; D = D_Stokes / gamma_c; effective mu = 6.65 gamma_c.")
print("      gamma_c   <tau(x0->0)>   <tau(x0->x1)>    R      mean/SD    (x0 = 1.2; times scale with the assumed D_Stokes)")
for gc in (2.0, 3.0, 4.2, 4.6):
    t0, _ = moments(6.65 * gc, 1.2, 0.0, D_stokes / gc)
    t1, s1 = moments(6.65 * gc, 1.2, -2.1, D_stokes / gc)
    print(f"        {gc:3.1f}       {t0:6.3f} s       {t1:6.3f} s      {t0 / t1:.3f}    {t1 / np.sqrt(s1 - t1 * t1):.3f}")
print("    Asymptotic values for distant points (P3 Eqs. 27, 32): R = 1/3, mean/SD = sqrt(3) = 1.732.")

# ---------------------------------------------------------------- P2, PRL 2018
print("\nP2. mu = 4.07 +- 0.03 kT/um^3 and lambda0 = 0.44 +- 0.01 1/s (printed); D not printed.")
E0 = fold_decay_rate(0.0)
D2 = 0.44 / (E0 * 4.07 ** (2.0 / 3.0))
print(f"    lambda0 = E0(0) D mu^(2/3) with E0(0) = {E0:.6f} gives D = {D2:.4f} +- {0.01 / 0.44 * D2:.4f} um^2/s "
      f"(Stokes, 1 um bead, water, 22 C: {KT / (6 * np.pi * 0.95e-3 * 0.5e-6) * 1e12:.3f}).")
sims = {  # SM Fig. S2, simulated (dashed) and measured S(t), read from the plot to +-0.005
    -1.0: ((0.2, 0.3, 0.5, 0.75, 1.0, 2.0), (0.85, 0.50, 0.17, 0.08, 0.05, 0.02), (0.84, 0.59, 0.22, 0.10, 0.09, 0.06)),
    -0.5: ((0.3, 0.5, 0.75, 1.0, 2.0, 4.0), (0.94, 0.68, 0.46, 0.34, 0.17, 0.06), (0.96, 0.72, 0.48, 0.33, 0.19, 0.10)),
    0.0: ((0.5, 0.75, 1.0, 2.0, 3.0, 4.0), (0.95, 0.82, 0.69, 0.40, 0.24, 0.14), (0.92, 0.78, 0.60, 0.31, 0.23, 0.15)),
}
curves = {}
print("    Survival S(t) with divergence counted at a = -10 um (as we read the text) and at a = -2.75 um (SM Fig. S3):")
print("      x0     t(s)   a=-10    a=-2.75    P2 sim.   P2 meas.")
for x0, (t, s_sim, s_meas) in sims.items():
    tt, S10 = survival(4.07, D2, x0, -10.0, 6.0)
    tt2, S275 = survival(4.07, D2, x0, -2.75, 6.0)
    curves[x0] = (tt2, S275, tt, S10)
    for ti, a, b, c, d in zip(t, np.interp(t, tt, S10), np.interp(t, tt2, S275), s_sim, s_meas, strict=True):
        print(f"      {x0:5.2f}  {ti:4.2f}   {a:.3f}     {b:.3f}      {c:.2f}      {d:.2f}")

# ---------------------------------------------------------------- the ramp
ell = 4.07 ** (-1.0 / 3.0)  # um
tau = ell * ell / D2  # s
f_unit = KT / (ell * 1e-6) * 1e15  # fN
y_esc = 2.75 / ell
print(f"\nThe ramp nobody has measured, in P2's potential: tilt lowered linearly to zero (the fold) at scaled rate Sigma.")
print(f"    Units: length {ell:.3f} um, time {tau:.3f} s, tilt force {f_unit:.2f} fN; start 4 units from the fold "
      f"({4 * f_unit:.1f} fN, well at {2 * ell:.2f} um, inside the calibrated range); escape at -2.75 um.")
print("      Sigma   tilt at which 10% / 50% / 90% have escaped (fN)   quasi-static median   ramp (fN/s)   run length (s)")
u_grid = np.linspace(4.0, 0.0, 401)
lam = fold_decay_rate(u_grid)
for sigma in (0.02, 0.05, 0.1, 0.3):
    u = ramped_fold_switching(sigma, (0.1, 0.5, 0.9), u0=4.0, y_escape=y_esc)
    H = np.concatenate([[0.0], np.cumsum(0.5 * (lam[1:] + lam[:-1]) * (-np.diff(u_grid)))]) / sigma
    qs = float(np.interp(np.log(2.0), H, u_grid)) if H[-1] > np.log(2.0) else float("nan")
    print(f"      {sigma:5.2f}    {u[0] * f_unit:6.2f} / {u[1] * f_unit:6.2f} / {u[2] * f_unit:6.2f}               "
          f"{qs * f_unit:6.2f}            {sigma * f_unit / tau:.3f}         {4.0 / sigma * tau:.0f}")
print("    Negative tilt: past the fold. The quasi-static column integrates the frozen escape rate (entry K0024 with K0008).")

# ---------------------------------------------------------------- figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    colors = {-1.0: "#1f77b4", -0.5: "#2ca02c", 0.0: "#d62728"}
    for x0, (t, s_sim, s_meas) in sims.items():
        tt2, S275, tt, S10 = curves[x0]
        ax[0].plot(tt2, S275, color=colors[x0], lw=1.6, label=f"x0 = {x0:g} um, divergence at -2.75 um")
        ax[0].plot(tt, S10, color=colors[x0], lw=1.0, ls=":")
        ax[0].plot(t, s_sim, "o", mfc="none", color=colors[x0], ms=5)
        ax[0].plot(t, s_meas, "x", color=colors[x0], ms=5)
    ax[0].set_xlim(0, 4.2)
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("survival S(t)")
    ax[0].set_title("P2: computed (lines; dotted: divergence at -10 um),\nP2's simulation (o) and measurement (x)", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)
    sig = np.array([0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3])
    q = np.array([ramped_fold_switching(s, (0.1, 0.5, 0.9), u0=4.0, y_escape=y_esc) for s in sig]) * f_unit
    ax[1].fill_between(sig * f_unit / tau, q[:, 2], q[:, 0], color="#bbbbbb", label="10% to 90% escaped")
    ax[1].plot(sig * f_unit / tau, q[:, 1], "k-", label="median")
    ax[1].axhline(0, color="k", lw=0.6, ls="--")
    ax[1].set_xscale("log")
    ax[1].set_xticks([0.2, 0.5, 1.0, 2.0])
    ax[1].set_xticklabels(["0.2", "0.5", "1", "2"])
    ax[1].minorticks_off()
    ax[1].set_xlabel("ramp of the tilt force (fN/s)")
    ax[1].set_ylabel("tilt force at escape (fN)")
    ax[1].set_title("Prediction: escape under a ramped tilt (P2's potential)", fontsize=9)
    ax[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out = Path(__file__).with_name("figure.png")
    fig.savefig(out, dpi=150)
    print(f"\nFigure: {out}")
except ImportError:
    print("\n(matplotlib not installed: no figure)")
