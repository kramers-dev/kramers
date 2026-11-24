#!/usr/bin/env python3
"""Compute the validity atlas and draw its figures.

    python tools/build_atlas.py              # compute every map, store it, draw its figure
    python tools/build_atlas.py M03 M04      # only these maps
    python tools/build_atlas.py --figures    # redraw the figures from the stored data only

Each map is stored as src/kramers/data/atlas_<id>.json and drawn as docs/atlas/<id>.png.
The pages themselves are written by tools/build_handbook.py, which also reports the
latest checks. Needs matplotlib (``pip install -e ".[dev]"``).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

from kramers import atlas  # noqa: E402

# Colours: categorical slots 1-3 and a blue-red diverging ramp with a grey midpoint,
# from the project's validated palette. Shapes and labels carry identity as well.
CAT = ("#2a78d6", "#eb6834", "#1baf7a")
MARK = ("o", "s", "^")
SURFACE = "#fcfcfb"
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e6e5e1"
DIVERGING = LinearSegmentedColormap.from_list(
    "kramers_diverging", ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#f2a3a0", "#e34948", "#8f2323"])
BLANK = "#d9d8d4"  # cells the map leaves out

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titlesize": 10, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def _arr(v):
    return np.array([np.nan if x is None else x for x in np.ravel(np.asarray(v, dtype=object))], dtype=float)


def _grid(rows):
    return np.array([_arr(r) for r in rows])


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig_m01(d, path):
    B = _arr(d["axes"]["barrier_kT"])
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    for (name, vals), c, m in zip(d["series"].items(), CAT, MARK, strict=True):
        e = 100 * _arr(vals)
        ax.plot(B, e, color=c, marker=m, ms=5, lw=1.8, label=name)
        ax.annotate(name, (B[-1], e[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8,
                    color=INK)
    for lvl in (1, 5):
        ax.axhline(lvl, color=MUTED, lw=0.7, ls=":")
        ax.text(2.05, lvl * 1.08, f"{lvl}%", color=MUTED, fontsize=7.5)
    ax.set(xscale="log", yscale="log", xlim=(1.8, 200), ylim=(0.2, 40), xlabel="barrier (kT)",
           ylabel="Kramers' rate too high by (%)")
    ax.set_xticks([2, 5, 10, 20, 50, 80], ["2", "5", "10", "20", "50", "80"])
    ax.set_yticks([0.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"])
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.set_title("Kramers' overdamped rate against the exact decay rate", loc="left")
    _style(ax)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_m02(d, path):
    b = _arr(d["axes"]["b"])
    f = d["facts"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    ax.axvspan(f["gap10_lo"], f["gap10_hi"], color=GRID, lw=0)
    ax.text(0.5 * (f["gap10_lo"] + f["gap10_hi"]), 36, "neither\nwithin 10%", ha="center", va="top", fontsize=7.5,
            color=MUTED)
    for lvl in (-10, 10):
        ax.axhline(lvl, color=MUTED, lw=0.7, ls=":")
    ax.axhline(0, color=MUTED, lw=0.7)
    for (name, vals), c in zip(d["series"].items(), CAT[:2], strict=True):
        e = 100 * _arr(vals)
        ax.plot(b, np.clip(e, -40, 40), color=c, lw=2.0, label=name)
    ax.annotate("weak-bias line\nturns negative", (f["lin_zero"], -40), xytext=(-2.2, -30), fontsize=7.5,
                color=INK, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.7})
    ax.set(xlim=(-10, 6), ylim=(-40, 40), xlabel=r"scaled bias  $b=\beta/D^{2/3}$  (b < 0: below the bifurcation)",
           ylabel="formula / exact rate - 1 (%), clipped at 40%")
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.set_title("Rate of the noisy saddle-node: two approximations against the exact rate", loc="left")
    _style(ax)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _heat(ax, x, y, z, lim, levels):
    zz = np.ma.masked_invalid(100 * z)
    cmap = DIVERGING.copy()
    cmap.set_bad(BLANK)
    pc = ax.pcolormesh(x, y, zz, cmap=cmap, vmin=-lim, vmax=lim, shading="nearest")
    cs = ax.contour(x, y, zz, levels=levels, colors=INK, linewidths=0.7, negative_linestyles="solid")
    ax.clabel(cs, fmt=lambda v: f"{v:+.0f}%", fontsize=7)
    return pc


def fig_m03(d, path):
    dG = _arr(d["axes"]["barrier_kT"])
    f = _arr(d["axes"]["force_over_critical"])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), sharey=True, constrained_layout=True)
    for ax, (name, rows), nu in zip(axes, d["series"].items(), (2.0 / 3.0, 0.5), strict=True):
        pc = _heat(ax, f, dG, _grid(rows), 20, [-10, 5, 10])
        left = dG[:, None] * (1 - f[None, :]) ** (1 / nu)
        ax.contour(f, dG, left, levels=[3.0], colors=INK, linewidths=1.1, linestyles="--")
        ax.set(yscale="log", xlabel="force / critical force")
        ax.set_yticks([5, 10, 20, 40, 80], ["5", "10", "20", "40", "80"])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(name, loc="left")
    axes[0].set_ylabel("barrier at zero force (kT)")
    axes[0].plot([], [], color=INK, lw=1.1, ls="--", label="3 kT of barrier left (right of it, less)")
    axes[0].legend(loc="lower left", frameon=False, fontsize=7.5)
    cb = fig.colorbar(pc, ax=axes, shrink=0.9, extend="both")
    cb.set_label("error of k(F)/k(0) (%)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_m04(d, path):
    dG = _arr(d["axes"]["barrier_kT"])
    s = _arr(d["axes"]["log_loading"])
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), sharey=True, constrained_layout=True)
    for ax, (name, rows) in zip(axes, d["series"].items(), strict=True):
        pc = _heat(ax, s, dG, _grid(rows), 30, [-10, -5, 5, 10])
        ax.set(yscale="log", xlabel=r"loading  $s=\ln(r x/k_0k_BT)\,/\,(\Delta G/k_BT)$")
        ax.set_yticks([7, 10, 20, 40, 80], ["7", "10", "20", "40", "80"])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(name, loc="left", fontsize=9)
    axes[0].set_ylabel("barrier at zero force (kT)")
    cb = fig.colorbar(pc, ax=axes, shrink=0.9, extend="both")
    cb.set_label("formula / exact - 1 (%); grey: not mapped")
    fig.savefig(path, dpi=180)
    plt.close(fig)


SEQUENTIAL = LinearSegmentedColormap.from_list(
    "kramers_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def fig_m05(d, path):
    N = _arr(d["axes"]["entering"])
    mu = _arr(d["axes"]["hazard_per_bin"])
    g = {k: _grid(v) for k, v in d["series"].items()}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True, constrained_layout=True)
    panels = (("relative bias of the rate", DIVERGING, TwoSlopeNorm(vcenter=0.0, vmin=-20, vmax=20), "bias (%)", 100),
              ("relative spread of the rate", SEQUENTIAL, LogNorm(vmin=0.5, vmax=300), "relative s.d. (%)", 100),
              ("coverage of the reported standard error", DIVERGING.reversed(),
               TwoSlopeNorm(vcenter=68.3, vmin=0, vmax=100), "coverage (%); 68.3 is right", 100))
    switches = N[:, None] * (1 - np.exp(-mu[None, :]))
    survivors = N[:, None] * np.exp(-mu[None, :])
    for ax, (name, cmap, norm, label, scale) in zip(axes, panels, strict=True):
        cm = cmap.copy()
        cm.set_bad(BLANK)
        pc = ax.pcolormesh(mu, N, np.ma.masked_invalid(scale * g[name]), cmap=cm, norm=norm, shading="nearest")
        cs = ax.contour(mu, N, np.minimum(switches, survivors), levels=[1, 10, 100], colors=INK, linewidths=0.7)
        ax.clabel(cs, fmt=lambda v: f"{v:g}", fontsize=7)
        ax.set(xscale="log", yscale="log", xlabel="expected hazard in the bin, mu")
        ax.set_title(name, loc="left", fontsize=9)
        fig.colorbar(pc, ax=ax, orientation="horizontal", shrink=0.8, pad=0.02, aspect=30).set_label(label, fontsize=8)
    axes[0].set_ylabel("switches still to come when the bin starts")
    fig.suptitle("Black contours: the expected number of switches inside the bin or after it, whichever is smaller",
                 x=0.01, ha="left", fontsize=8.5, color=INK)
    fig.savefig(path, dpi=180)
    plt.close(fig)


FIGURES = {"M01": fig_m01, "M02": fig_m02, "M03": fig_m03, "M04": fig_m04, "M05": fig_m05}


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    figures_only = "--figures" in sys.argv
    out = ROOT / "docs" / "atlas"
    out.mkdir(parents=True, exist_ok=True)
    for m in atlas.maps():
        if args and m.id not in args:
            continue
        if not figures_only:
            t0 = time.time()
            data = m.build()
            m.path.parent.mkdir(parents=True, exist_ok=True)
            m.path.write_text(json.dumps(data, indent=1) + "\n")
            print(f"{m.id}: computed in {time.time() - t0:.0f} s -> {m.path.relative_to(ROOT)}", flush=True)
        FIGURES[m.id](m.load(), out / f"{m.id}.png")
        print(f"{m.id}: figure -> docs/atlas/{m.id}.png")
        print(f"   {m.claim()}")


if __name__ == "__main__":
    main()
