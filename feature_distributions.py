"""Per-feature distributions (CaOx reference vs uric acid) with the 90% tolerance intervals."""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from stone_benchmark import load, Reference, logit, inv_logit

df = load(sys.argv[1])
ok = df[df.excluded_reason == ""]
ref = ok[ok.composition_class == "Calcium oxalate"]
ua = ok[ok.composition_class == "Uric acid"]
ti = Reference(ref).tolerance_intervals().set_index("feature")

INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
C_REF, C_UA = "#2f6fb0", "#d97b29"
panels = [("porosity_pct", "Porosity (%)", lambda v: logit(v / 100), lambda z: 100 * inv_logit(z),
           [0.02, 0.1, 0.5, 2, 10], "porosity_pct"),
          ("sphericity", "Sphericity (1 = perfect sphere)", logit, inv_logit,
           [0.1, 0.2, 0.4, 0.6, 0.8, 0.9], "sphericity"),
          ("volume_mm3", "Volume (mm³) — size covariate", np.log10, lambda z: 10 ** z,
           [10, 30, 100, 300, 1000], None)]

fig, axes = plt.subplots(3, 1, figsize=(7.6, 8.4), dpi=200)
for ax, (col, label, fwd, back, ticks, tikey) in zip(axes, panels):
    zr, zu = fwd(ref[col].values), fwd(ua[col].values)
    lo, hi = min(zr.min(), zu.min()) - 0.8, max(zr.max(), zu.max()) + 0.8
    grid = np.linspace(lo, hi, 400)
    for z, c, name in [(zr, C_REF, f"Calcium oxalate (reference), n={len(zr)}"),
                       (zu, C_UA, f"Uric acid (control), n={len(zu)}")]:
        dens = stats.gaussian_kde(z)(grid)
        ax.fill_between(grid, dens, color=c, alpha=0.22, lw=0)
        ax.plot(grid, dens, color=c, lw=1.3, label=name)
    ymax = ax.get_ylim()[1]
    ax.plot(zr, np.full_like(zr, -0.06 * ymax), "|", color=C_REF, ms=9, mew=1.2)
    ax.plot(zu, np.full_like(zu, -0.13 * ymax), "|", color=C_UA, ms=9, mew=1.2)
    if tikey:
        a, b = fwd(ti.loc[tikey, "TI90_low"]), fwd(ti.loc[tikey, "TI90_high"])
        ax.axvspan(a, b, color=C_REF, alpha=0.06, lw=0)
        for x in (a, b):
            ax.axvline(x, color=C_REF, ls="--", lw=1)
        ax.text(b, ymax * 0.92, "  90% tolerance\n  interval edge", color=C_REF, fontsize=7.5, va="top")
    ax.set_ylim(-0.18 * ymax, ymax)
    ax.set_xlim(lo, hi)
    ax.set_xticks(fwd(np.array(ticks, float)))
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_yticks([])
    ax.set_xlabel(label, color=INK, fontsize=9.5)
    ax.grid(axis="x", color=GRID, lw=0.6)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
axes[0].legend(loc="upper left", fontsize=8, frameon=False)
fig.suptitle("Distribution of each micro-CT feature, patient stones", x=0.04, ha="left",
             fontsize=12, color=INK)
fig.text(0.04, 0.945, "Smoothed density (KDE) on the analysis scale; ticks under each curve = individual stones",
         ha="left", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("benchmark_outputs/feature_distributions.png", bbox_inches="tight")
