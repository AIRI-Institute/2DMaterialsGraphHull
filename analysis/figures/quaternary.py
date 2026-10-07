#!/usr/bin/env python3
"""On-hull map for the 21 vacancy-free quaternary compositions -- VACANCY-FREE HULL.

Companion to figure_quaternary_onhull.py (full 420-composition hull). Here the hull is
rebuilt from the vacancy-free branch alone, the same frame the vibrational tie-plane
analysis uses. At 0 K the two hulls are identical; above it the vacancy-bearing phases are
absent, so every quaternary reaches the hull and the onsets are lower.

Left  : dGhull(T) as a strip per composition; black bars mark the temperature windows in
        which the composition actually lies ON the free-energy hull; circles mark onsets.
Right : dGhull at 0 K -- the starting distance that decides whether entropy suffices.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

ON = 1e-6
VMAX = 12.0


def plot_quaternary(hull_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(hull_path)))
    T = np.array([int(c.split("_")[1][:-1]) for c in rows[0] if c.startswith("dGhull_")])
    recs = []
    for r in rows:
        if r["n_elements"] != "4" or r["n_vac_Me"] != "0" or r["n_vac_X"] != "0":
            continue
        g = np.array([float(r[f"dGhull_{t}K"]) for t in T])
        idx = np.where(g <= ON)[0]
        recs.append({"k": (int(r["W"]), int(r["Mo"]), int(r["Se"]), int(r["S"])),
                     "g": g, "onset": T[idx[0]] if len(idx) else None,
                     "frac": len(idx) / len(T)})
    recs.sort(key=lambda d: (d["onset"] is None, d["onset"] if d["onset"] is not None
                             else d["g"].min()))
    print(f"{len(recs)} vacancy-free quaternary compositions")

    def tex(k):
        return (rf"W$_{{{k[0]}}}$Mo$_{{{k[1]}}}$Se$_{{{k[2]}}}$S$_{{{k[3]}}}$")

    n = len(recs)
    fig = plt.figure(figsize=(11.4, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.05, 1.0], wspace=0.075,
                          left=0.150, right=0.905, top=0.885, bottom=0.115)
    ax, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    Z = np.array([d["g"] for d in recs])
    te = np.concatenate([T - 5, [T[-1] + 5]])
    ye = np.arange(n + 1) - 0.5
    cmap = plt.get_cmap("viridis").copy(); cmap.set_over("#fcffa4")
    m = ax.pcolormesh(te, ye, Z, cmap=cmap, norm=Normalize(0, VMAX),
                      shading="flat", rasterized=True)

    # on-hull windows + onsets
    for i, d in enumerate(recs):
        on = d["g"] <= ON
        if on.any():
            edges = np.diff(np.concatenate([[0], on.astype(int), [0]]))
            for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
                ax.plot([T[a], T[b - 1]], [i, i], color="w", lw=4.2,
                        solid_capstyle="butt", zorder=4)
            ax.plot(d["onset"], i, "o", ms=6.5, mfc="#d95f02", mec="w", mew=1.3,
                    zorder=6)
            ax.annotate(f"{d['onset']:.0f} K", (d["onset"], i), fontsize=8,
                        textcoords="offset points", xytext=(-8, 0), ha="right",
                        va="center", zorder=7, color="w",
                        path_effects=[pe.withStroke(linewidth=2.0, foreground="0.15")])

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([tex(d["k"]) for d in recs], fontsize=8.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(0, 1200)
    ax.set_xlabel("Temperature (K)", fontsize=10)
    ax.set_title("on the finite-temperature convex hull  "
                 "(white bar = ΔG$_{\\rm hull}$ = 0)", fontsize=10, loc="left")
    n_never = sum(1 for d in recs if d["onset"] is None)
    if n_never:
        div = n - n_never - 0.5
        ax.axhline(div, color="0.4", lw=1.0, ls="--")
        ax.text(20, div + 0.25, "never reach the hull", fontsize=8.5, ha="left",
                va="top", color="w", style="italic",
                path_effects=[pe.withStroke(linewidth=2.0, foreground="0.15")])

    # right: starting distance at 0 K
    g0 = [d["g"][0] for d in recs]
    cols = ["#1b7837" if d["onset"] is not None else "#b2182b" for d in recs]
    axb.barh(np.arange(n), g0, color=cols, height=0.66)
    for i, v in enumerate(g0):
        axb.text(v + 0.25, i, f"{v:.1f}", va="center", fontsize=7.5)
    axb.set_yticks(np.arange(n)); axb.set_yticklabels([])
    axb.set_xticks([5, 10])
    axb.set_ylim(n - 0.5, -0.5); axb.set_xlim(0, 13)
    axb.set_xlabel(r"$\Delta G_{\rm hull}$(0 K), meV/atom", fontsize=9)
    if n_never:
        axb.axhline(div, color="0.4", lw=1.0, ls="--")
    axb.set_title("0 K starting distance", fontsize=10, loc="left")

    cax = fig.add_axes([0.918, 0.115, 0.015, 0.77])
    cb = fig.colorbar(m, cax=cax, extend="max")
    cb.set_label(r"$\Delta G_{\rm hull}$ (meV/atom)", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    fig.suptitle("Vacancy-free quaternary W–Mo–Se–S monolayer alloys, 2×2×1 CCS — hull built "
                 "from the vacancy-free branch only", fontsize=11, x=0.150,
                 ha="left", y=0.975)
    for a in (ax, axb):
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(labelsize=8.5)

    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"figure_quaternary_onhull_vacancyfree.{ext}", dpi=300, bbox_inches="tight")
    print("wrote figure_quaternary_onhull_vacancyfree.pdf / .png")
    got = [d for d in recs if d["onset"] is not None]
    print(f"  reach the hull: {len(got)}/{n}; earliest {min(d['onset'] for d in got):.0f} K")
    print(f"  widest on-hull window: "
          f"{max(got, key=lambda d: d['frac'])['frac']*100:.0f} % of the 0-1200 K range")


    return fig
