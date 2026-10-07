#!/usr/bin/env python3
"""Figure 6d-style onset chart for the 4x4x1 second inference set.

One column per composition; the bar spans the temperature range over which the
composition lies ABOVE the free-energy hull, so the grey remainder is the range over
which it defines the hull. Compositions that never reach the hull are drawn full height
in a separate colour, which the original style cannot distinguish from a 1200 K onset.
"""
from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.patches import Patch

ON = 1e-6
ABOVE = "#3f7fb5"      # above the hull
NEVER = "#b2182b"      # never reaches it
ALLQ = 180.0
ONHULL = "#dcdcdc"     # background = on the hull


def above_intervals(g, T):
    """All (T_start, T_end) intervals where the composition lies ABOVE the hull."""
    a = g > ON
    e = np.diff(np.concatenate([[0], a.astype(int), [0]]))
    step = T[1] - T[0]
    return [(T[i], T[j - 1] + step) for i, j in zip(np.where(e == 1)[0], np.where(e == -1)[0])]


def draw_bars(ax, cols):
    """Blue segments for every above-hull episode; red full height if never on hull."""
    for i, r in enumerate(cols):
        if r["onset"] is None:
            ax.bar(i, 1200, bottom=0, width=0.86, color=NEVER, linewidth=0)
            continue
        for t0, t1 in r["ivals"]:
            ax.bar(i, t1 - t0, bottom=t0, width=0.86, color=ABOVE, linewidth=0)


def load(path="fi_onsets.csv"):
    rows = list(csv.DictReader(open(path)))
    T = np.array([int(c.split("_")[1][:-1]) for c in rows[0] if c.startswith("dGhull_")])
    out = []
    for r in rows:
        w, mo, se, s = (int(r[k]) for k in ("W", "Mo", "Se", "S"))
        g = np.array([float(r[f"dGhull_{t}K"]) for t in T])
        idx = np.where(g <= ON)[0]
        out.append({"W": w, "Mo": mo, "Se": se, "S": s,
                    "vme": 16 - w - mo, "vx": 32 - se - s,
                    "onset": T[idx[0]] if len(idx) else None,
                    "ivals": above_intervals(g, T)})
    return out


def tex(r):
    p = ""
    for el, n in (("W", r["W"]), ("Mo", r["Mo"]), ("Se", r["Se"]), ("S", r["S"])):
        if n:
            p += rf"{el}$_{{{n}}}$"
    return p


def plot_mow(data_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recs = load(data_path)
    vf = [r for r in recs if r["vme"] == 0 and r["vx"] == 0]
    fams = [((32, 0), r"Se$_{32}$"), ((31, 1), r"Se$_{31}$S$_1$"),
            ((1, 31), r"Se$_1$S$_{31}$"), ((0, 32), r"S$_{32}$")]
    cols, bounds = [], []
    for key, _ in fams:
        sub = sorted([r for r in vf if (r["Se"], r["S"]) == key], key=lambda r: r["Mo"])
        bounds.append((len(cols), len(cols) + len(sub)))
        cols += sub

    fig, ax = plt.subplots(figsize=(13.67, 4.37))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.30)
    ax.set_facecolor(ONHULL)

    x = np.arange(len(cols))
    draw_bars(ax, cols)

    tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, (a, b) in enumerate(bounds):
        if i:
            ax.axvline(a - 0.5, color="w", lw=2.0)
        ax.text((a + b - 1) / 2, 1.035, fams[i][1], transform=tr, ha="center",
                va="bottom", fontsize=11, fontweight="bold")

    # ---- median and spread of the QUATERNARY compositions --------------------
    import statistics as st
    quat_idx, spans = [], []
    for (a, b), (key, _) in zip(bounds, fams):
        if min(key) == 0:                      # pure chalcogen family -> ternary
            continue
        inner = [i for i in range(a, b) if 0 < cols[i]["Mo"] < 16]
        quat_idx += inner
        spans.append((min(inner) - 0.45, max(inner) + 0.45))
    qon = sorted(cols[i]["onset"] for i in quat_idx if cols[i]["onset"] is not None)
    med = st.median(qon); q1 = qon[len(qon) // 4]; q3 = qon[(3 * len(qon)) // 4]
    for lo, hi in spans:
        ax.fill_between([lo, hi], q1, q3, color="#d95f02", alpha=0.20, zorder=3, lw=0)
        ax.plot([lo, hi], [med, med], color="#d95f02", lw=2.2, zorder=4)
    ax.annotate(f"quaternary median {med:.0f} K  (IQR {q1:.0f}–{q3:.0f} K)",
                xy=(spans[0][0], med), xytext=(2, 7), textcoords="offset points",
                fontsize=8.5, color="#a5490a", va="bottom", ha="left", zorder=7,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.70))
    ALLQ = 180.0
    # global quaternary reference: drawn ONLY across the quaternary families
    for lo, hi in spans:
        ax.plot([lo, hi], [ALLQ, ALLQ], color="0.35", lw=1.2, ls=(0, (5, 3)), zorder=3)
    ax.annotate("43 of 88 vacancy-free quaternaries reach the hull: median onset 180 K",
                xy=(spans[-1][1], ALLQ), xytext=(-2, -4), textcoords="offset points",
                fontsize=8.5, color="0.30", ha="right", va="top", zorder=7,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.70))

    ax.set_xticks(x)
    ax.set_xticklabels([tex(r) for r in cols], rotation=90, fontsize=6.5)
    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.set_ylim(1200, 0)
    ax.set_yticks(np.arange(0, 1201, 200))
    ax.set_ylabel("T, K", fontsize=11)
    ax.tick_params(labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax.legend(handles=[Patch(facecolor=ABOVE, label="above hull"),
                       Patch(facecolor=ONHULL, label="on hull"),
                       Patch(facecolor=NEVER, label="never on hull")],
              loc="lower left", bbox_to_anchor=(0.006, 0.02), fontsize=9,
              handlelength=1.3, frameon=True, framealpha=0.88, edgecolor="none",
              facecolor="white", ncol=1, columnspacing=1.2)


    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig441_6d_MoW.{ext}", dpi=300, bbox_inches="tight")
    nn = sum(1 for r in cols if r["onset"] is None)
    ons = [r["onset"] for r in cols if r["onset"] is not None]
    print(f"wrote fig441_6d_MoW: {len(cols)} compositions, {nn} never on hull, "
          f"median onset {np.median(ons):.0f} K, max {max(ons):.0f} K")


    return fig

def draw_sse(ax, vf, families, title):
    cols, bounds = [], []
    for sel, _, _ in families:
        sub = sorted([r for r in vf if sel(r)], key=lambda r: r["S"])
        bounds.append((len(cols), len(cols) + len(sub)))
        cols += sub

    ax.set_facecolor(ONHULL)
    x = np.arange(len(cols))
    draw_bars(ax, cols)

    tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, (a, b) in enumerate(bounds):
        if i:
            ax.axvline(a - 0.5, color="w", lw=2.0)
        ax.text((a + b - 1) / 2, 1.04, families[i][1], transform=tr, ha="center",
                va="bottom", fontsize=10.5, fontweight="bold")

    # median + IQR of the quaternary family in this panel
    qspans = []
    for (a, b), (_, _, is_quat) in zip(bounds, families):
        if not is_quat:
            continue
        # the family spans z = 0..32; its two endpoints are TERNARY (one chalcogen
        # absent), so restrict the statistic and the band to the quaternary columns
        qi = [i for i in range(a, b)
              if min(cols[i]["Se"], cols[i]["S"]) > 0 and min(cols[i]["W"], cols[i]["Mo"]) > 0]
        lo, hi = min(qi) - 0.45, max(qi) + 0.45
        qspans.append((lo, hi))
        on = sorted(cols[i]["onset"] for i in qi if cols[i]["onset"] is not None)
        med, q1, q3 = st.median(on), on[len(on) // 4], on[(3 * len(on)) // 4]
        ax.fill_between([lo, hi], q1, q3, color="#d95f02", alpha=0.20,
                        zorder=3, lw=0)
        ax.plot([lo, hi], [med, med], color="#d95f02", lw=2.2, zorder=4)
        # NB: no per-panel reach ratio here. The Mo/W and S/Se figures share four
        # compositions, so panel ratios do not sum to the global 43 of 88 and invite
        # exactly that mistake; the global count is on the dashed reference line.
        ax.annotate(f"quaternary median {med:.0f} K  (IQR {q1:.0f}–{q3:.0f} K)",
                    xy=(lo + 0.15, q3), xytext=(2, -6), textcoords="offset points",
                    fontsize=8.5, color="#a5490a", va="top", ha="left", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.70))

    # global quaternary reference: drawn ONLY across the quaternary families
    for lo, hi in qspans:
        ax.plot([lo, hi], [ALLQ, ALLQ], color="0.35", lw=1.2, ls=(0, (5, 3)), zorder=3)
    if qspans:
        ax.annotate("43 of 88 vacancy-free quaternaries reach the hull: "
                    "median onset 180 K",
                    xy=(qspans[-1][1], ALLQ), xytext=(-2, -4),
                    textcoords="offset points", fontsize=8.5, color="0.30",
                    ha="right", va="top", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none",
                              alpha=0.70))
    ax.set_xticks(x)
    ax.set_xticklabels([tex(r) for r in cols], rotation=90, fontsize=6.0)
    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.set_ylim(1200, 0)
    ax.set_yticks(np.arange(0, 1201, 200))
    ax.set_ylabel("T, K", fontsize=11)
    ax.tick_params(labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def plot_sse(data_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recs = load(data_path)
    vf = [r for r in recs if r["vme"] == 0 and r["vx"] == 0]
    fig, axes = plt.subplots(2, 1, figsize=(13.67, 8.93))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.965, bottom=0.145, hspace=0.80)

    draw_sse(axes[0], vf,
         [(lambda r: r["W"] == 15 and r["Mo"] == 1, r"W$_{15}$Mo$_1$Se$_{32-z}$S$_z$", True),
          (lambda r: r["W"] == 16 and r["Mo"] == 0, r"W$_{16}$Se$_{32-z}$S$_z$", False)],
         "Se/S alloys (W-rich) — 4×4×1 second inference set")
    draw_sse(axes[1], vf,
         [(lambda r: r["Mo"] == 16 and r["W"] == 0, r"Mo$_{16}$Se$_{32-z}$S$_z$", False),
          (lambda r: r["W"] == 1 and r["Mo"] == 15, r"W$_1$Mo$_{15}$Se$_{32-z}$S$_z$", True)],
         "Se/S alloys (Mo-rich) — 4×4×1 second inference set")

    axes[0].legend(handles=[Patch(facecolor=ABOVE, label="above hull"),
                            Patch(facecolor=ONHULL, label="on hull"),
                            Patch(facecolor=NEVER, label="never on hull")],
                   loc="lower right", bbox_to_anchor=(0.998, 0.02), fontsize=8,
                   handlelength=1.2, frameon=True, framealpha=0.9, edgecolor="none",
                   facecolor="white", ncol=1, labelspacing=0.45, borderpad=0.5)
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig441_6d_SSe.{ext}", dpi=300, bbox_inches="tight")
    print("wrote fig441_6d_SSe")


    return fig
