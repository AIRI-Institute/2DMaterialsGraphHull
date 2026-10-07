#!/usr/bin/env python3
"""Supplementary Figure 25 -- vibrational cancellation on the tie planes.

Selection rule: every supporting tie plane, plus the first
computable competing plane per target. Solid = supporting regime (T <= onset, a genuine
decomposition); dashed = competing regime (the target is a hull vertex, and the plane
defines a stability margin).
"""
from pathlib import Path
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
def plot_vibrational(vibrational, selected, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    d, F = vibrational, selected
    with plt.rc_context({"font.family": "Arial", "font.size": 11, "axes.linewidth": .9,
                         "xtick.labelsize": 10, "ytick.labelsize": 10}):
        def pretty(s):
            o=""
            for e,n in re.findall(r"([A-Z][a-z]?)(\d+)",s): o+=e+("" if n=="1" else "$_{%s}$"%n)
            return o
        SUPCOL=["#1f4e79","#2e7d32","#7b8ea3","#8a6d1f"]; COMPCOL="#b5651d"
        order=["W1Mo3Se1S7","W3Mo1Se7S1","W1Mo3Se2S6","W2Mo2Se7S1","W2Mo2Se2S6","W1Mo3Se4S4"]
        fig,axes=plt.subplots(2,3,figsize=(16.5,9.9)); axes=axes.ravel()
        sup=[]
        for ax,t in zip(axes,order):
            info=F[t]; on=info["onset"]; runs=sorted(info["runs"],key=lambda r:r["lo"])
            ax.axvspan(on,1200,color="#2e7d32",alpha=0.045,lw=0,zorder=0)
            handles=[]; isup=0
            for r in runs:
                mem=" + ".join(f"{v:.3f}·{pretty(c)}" for c,v in r["members"].items())
                if not r["ok"]:
                    miss=", ".join(pretty(c) for c in r["members"] if c not in d.columns)
                    ax.axvspan(r["lo"],r["hi"],color="0.86",alpha=0.6,lw=0,zorder=1)
                    handles.append(Patch(fc="0.86",ec="none",label=f"{r['lo']}–{r['hi']} K   ({miss} pending)\n{mem}"))
                    continue
                competing = r["role"]=="competing"
                col = COMPCOL if competing else SUPCOL[isup % len(SUPCOL)]
                if not competing: isup+=1
                dF=d[t]-sum(v*d[c] for c,v in r["members"].items())
                ax.plot(dF.index,dF.values,color=col,lw=1.0,alpha=0.28,zorder=2)
                win=(dF.index>=r["lo"])&(dF.index<=r["hi"])
                s_=win&(dF.index<=on); m_=win&(dF.index>on)
                for mask,kw in ((s_,dict(lw=3.0,solid_capstyle="round")),
                                (m_,dict(lw=2.5,ls=(0,(3.5,1.6))))):
                    if not mask.any(): continue
                    ax.plot(dF.index[mask],dF.values[mask],color=col,zorder=4,**kw)
                # Every window is marked at its ends, in every panel, so a window spanning a single
                # grid point (e.g. 300-300 K) -- which draws as nothing at any line width -- appears
                # under the same rule as the other windows.
                ends=[r["lo"]] if r["lo"]==r["hi"] else [r["lo"],r["hi"]]
                ax.plot(ends,[dF.loc[e] for e in ends],color=col,ls="none",
                        marker="o",ms=4.5,zorder=5)
                if competing:
                    lab=f"{r['lo']}–{r['hi']} K   comp.\n{mem}"
                else:
                    v=dF.loc[r["lo"]:min(r["hi"],on)].abs().max(); sup.append((t,r["lo"],v))
                    tail=f" / {dF.loc[on:r['hi']].abs().max():.3f}" if r["hi"]>on else ""
                    lab=f"{r['lo']}–{r['hi']} K   |Δ|max {v:.3f}{tail}\n{mem}"
                handles.append(Line2D([],[],color=col,lw=3.0,marker="o",ms=4.5,
                                      ls="-" if not competing else (0,(3.5,1.6)),label=lab))
            ax.axhline(0,color="k",lw=0.8,zorder=3)
            ax.axvline(on,color="#c00000",ls="--",lw=1.6,zorder=5)
            ax.annotate(f"{on:.0f} K",xy=(on,-0.106),xytext=(5,2),textcoords="offset points",
                        fontsize=10,color="#c00000",rotation=90,va="bottom",ha="left",zorder=6,
                        bbox=dict(fc="white",ec="none",alpha=0.8,pad=0.8))
            ax.set_xlim(0,1200); ax.set_ylim(-0.11,0.11)
            nm=sum(r["role"]=="competing" for r in runs)
            ax.set_title(f"{pretty(t)}    (tie planes: all {info['n_sup']} supp."
                         f" + {nm}/{info['n_mar']} comp.)", fontsize=13,pad=4)
            ax.set_xlabel("temperature (K)",fontsize=11.5,labelpad=3)
            ax.set_ylabel("$\\Delta F_{vib}^{d}$ (meV/atom)",fontsize=11.5,labelpad=3)
            ax.legend(handles=handles,fontsize=8.2,loc="upper center",bbox_to_anchor=(0.5,-0.205),
                      frameon=True,framealpha=1.0,borderpad=0.3,handlelength=1.8,labelspacing=0.32,
                      handletextpad=0.6)
            ax.tick_params(length=4)
        fig.suptitle("Vibrational cancellation on the tie planes around each stabilization onset",fontsize=17,y=0.995)
        fig.tight_layout(w_pad=1.0,h_pad=1.0,rect=[0,0,1,0.975])
        for ext in ("pdf","png"):
            fig.savefig(output_dir / f"figure_dfvib_onset_window.{ext}",dpi=200,bbox_inches="tight")
        print(f"  {len(sup)} supporting planes drawn, max |dF_vib^d| = {max(x[2] for x in sup):.3f} meV/atom")

        return fig
