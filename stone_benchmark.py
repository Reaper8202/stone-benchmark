"""
Reference distributions for a stone-farm fidelity benchmark
(internal architecture dimension: porosity + sphericity from micro-CT).

Usage
-----
    python stone_benchmark.py "MASTER STUDY DATA - ALL 1.csv"                  # build reference
    python stone_benchmark.py "MASTER STUDY DATA - ALL 1.csv" farm_stones.csv  # also score farm stones

farm_stones.csv needs columns: stone_id, volume_mm3, porosity_pct, surface_area_mm2
(surface area from the SAME estimator as the reference; Lindblad by default).

Design choices (see README section printed in the report):
  * Unit = stone, but stones from the same patient are correlated, so every
    resampling / leave-out step is done by PATIENT (cluster), and tolerance
    intervals use an effective sample size from the intraclass correlation.
  * Features are analysed on the logit scale (both are bounded proportions).
  * Volume is a covariate (size), not a fidelity feature.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(20261016)
SA_COL = "Surface Area Lindblad (mm^2)"   # primary SA estimator
SA_COL_ALT = "Surface Area Lorensen (mm^2)"  # sensitivity analysis
ENVELOPE = 0.90                            # pre-registered envelope level
OUT = Path("benchmark_outputs")


# --------------------------------------------------------------------------- #
# 1. Load and clean
# --------------------------------------------------------------------------- #
def num(x):
    """Parse '18,157.74', '2.09%', ' 100 ' -> float; anything else -> NaN."""
    if pd.isna(x):
        return np.nan
    s = str(x).replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def composition_class(r):
    """Dominant-component class from the analysis percentages."""
    caox = r["COM"] + r["COD"]
    ua = r["UA"] + r["UAD"]
    parts = {"Calcium oxalate": caox, "Uric acid": ua,
             "Calcium phosphate": r["Apatite"] + r["Brushite"],
             "Struvite": r["Struvite"], "Cystine": r["Cystine"]}
    best = max(parts, key=parts.get)
    return best if parts[best] > 0 else "Unknown"


def load(path):
    raw = pd.read_csv(path, header=1, dtype=str)
    pid_col = raw.columns[0]
    raw = raw[raw[pid_col].fillna("").str.match(r"^\d")]          # stone rows only
    df = pd.DataFrame({
        "stone_id": raw[pid_col].str.strip(),
        "patient": raw[pid_col].str.extract(r"^(\d+)")[0],
        "COM": raw["COM %"].map(num), "COD": raw["COD %"].map(num),
        "UA": raw["UA %"].map(num), "UAD": raw["UAD %"].map(num),
        "Apatite": raw["Apatite (Ca5(PO4)3) %"].map(num),
        "Brushite": raw["Brushite (CaHPO4.2H2O)"].map(num),
        "Struvite": raw["Struvite %"].map(num), "Cystine": raw["Cystine"].map(num),
        "reported_type": raw["Stone Type"].str.strip(),
        "volume_mm3": raw["Total Stone V (mm^3)"].map(num),
        "pore_volume_mm3": raw["V Pore (mm^3)"].map(num),
        "porosity_pct": raw["Porosity %"].map(num),
        "sa_lindblad": raw[SA_COL].map(num),
        "sa_lorensen": raw[SA_COL_ALT].map(num),
    })
    df[["COM", "COD", "UA", "UAD", "Apatite", "Brushite", "Struvite", "Cystine"]] = \
        df[["COM", "COD", "UA", "UAD", "Apatite", "Brushite", "Struvite", "Cystine"]].fillna(0)

    # exclusion log
    reasons = []
    for _, r in df.iterrows():
        if not (r.volume_mm3 > 0):
            reasons.append("no micro-CT volume (blank/0/'REDO'/'HARD TO SEGMENT')")
        elif np.isnan(r.porosity_pct):
            reasons.append("no porosity value (row misaligned or missing)")
        elif not (r.sa_lindblad > 0):
            reasons.append("no surface area")
        else:
            reasons.append("")
    df["excluded_reason"] = reasons

    # internal consistency check: porosity should equal V_pore / V_total
    calc = 100 * df.pore_volume_mm3 / df.volume_mm3
    df["porosity_check_diff"] = (calc - df.porosity_pct).round(3)

    df["composition_class"] = df.apply(composition_class, axis=1)
    df["sphericity"] = (np.pi ** (1 / 3) * (6 * df.volume_mm3) ** (2 / 3)) / df.sa_lindblad
    df["sphericity_lorensen"] = (np.pi ** (1 / 3) * (6 * df.volume_mm3) ** (2 / 3)) / df.sa_lorensen
    return df


# --------------------------------------------------------------------------- #
# 2. Transforms and statistics
# --------------------------------------------------------------------------- #
EPS = 1e-4


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def inv_logit(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))


def features(d):
    """Return n x 2 matrix on analysis scale: logit(porosity), logit(sphericity)."""
    return np.column_stack([logit(d.porosity_pct / 100), logit(d.sphericity)])


def icc_oneway(y, groups):
    """ANOVA ICC(1) for unbalanced clusters."""
    g = pd.Series(y).groupby(np.asarray(groups))
    k = g.ngroups
    n = len(y)
    if k < 2 or n == k:
        return 0.0
    n_i = g.size().values
    grand = np.mean(y)
    ssb = np.sum(n_i * (g.mean().values - grand) ** 2)
    ssw = sum(((v - v.mean()) ** 2).sum() for _, v in g)
    msb, msw = ssb / (k - 1), ssw / (n - k)
    n0 = (n - np.sum(n_i ** 2) / n) / (k - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    return float(max(icc, 0.0))


def n_effective(n, groups, icc):
    m_bar = n / pd.Series(groups).nunique()
    return n / (1 + (m_bar - 1) * icc)


def tolerance_k(n_eff, coverage=ENVELOPE, conf=0.95):
    """Howe (1969) two-sided normal tolerance factor."""
    nu = n_eff - 1
    z = stats.norm.ppf((1 + coverage) / 2)
    chi = stats.chi2.ppf(1 - conf, nu)
    return z * np.sqrt(nu * (1 + 1 / n_eff) / chi)


def cluster_boot(d, fn, B=5000):
    pats = d.patient.unique()
    by = {p: d[d.patient == p] for p in pats}
    out = []
    for _ in range(B):
        samp = RNG.choice(pats, len(pats), replace=True)
        out.append(fn(pd.concat([by[p] for p in samp])))
    return np.array(out)


def describe(d, col, scale=1.0):
    y = d[col].values * scale
    return dict(n=len(y), median=np.median(y), q1=np.percentile(y, 25),
                q3=np.percentile(y, 75), min=y.min(), max=y.max())


def size_dependence(d):
    """OLS slope of each logit feature on log10(volume), patient-cluster bootstrap CI."""
    rows = []
    x = np.log10(d.volume_mm3.values)
    F = features(d)
    for j, name in enumerate(["porosity", "sphericity"]):
        slope = np.polyfit(x, F[:, j], 1)[0]
        boots = cluster_boot(d, lambda s, j=j: np.polyfit(np.log10(s.volume_mm3.values),
                                                           features(s)[:, j], 1)[0], B=3000)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        r, _ = stats.spearmanr(x, F[:, j])
        rows.append(dict(feature=name, slope_logit_per_log10V=slope, ci_low=lo, ci_high=hi,
                         spearman_rho=r, size_dependent=not (lo <= 0 <= hi)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. Reference model: marginal tolerance intervals + bivariate envelope
# --------------------------------------------------------------------------- #
class Reference:
    def __init__(self, ref):
        self.ref = ref.reset_index(drop=True)
        self.F = features(self.ref)
        self.mu = self.F.mean(0)
        self.cov = np.cov(self.F, rowvar=False)
        self.cov_inv = np.linalg.inv(self.cov)
        self.icc = [icc_oneway(self.F[:, j], self.ref.patient) for j in range(2)]
        self.n_eff = [n_effective(len(self.ref), self.ref.patient, i) for i in self.icc]
        self.null_d2 = self._lopo_null()
        self.envelope_cut = float(np.quantile(self.null_d2, ENVELOPE))

    def _lopo_null(self):
        """Leave-one-PATIENT-out Mahalanobis distances of real reference stones."""
        d2 = np.empty(len(self.ref))
        for p in self.ref.patient.unique():
            out = (self.ref.patient == p).values
            Fin = self.F[~out]
            mu, ci = Fin.mean(0), np.linalg.inv(np.cov(Fin, rowvar=False))
            diff = self.F[out] - mu
            d2[out] = np.einsum("ij,jk,ik->i", diff, ci, diff)
        return d2

    def d2(self, F):
        diff = np.atleast_2d(F) - self.mu
        return np.einsum("ij,jk,ik->i", diff, self.cov_inv, diff)

    def tolerance_intervals(self):
        rows = []
        for j, (name, back) in enumerate([("porosity_pct", lambda z: 100 * inv_logit(z)),
                                          ("sphericity", inv_logit)]):
            m, s = self.F[:, j].mean(), self.F[:, j].std(ddof=1)
            k = tolerance_k(self.n_eff[j])
            # nonparametric cluster bootstrap of the 5th/95th percentiles (cross-check)
            col = "porosity_pct" if j == 0 else "sphericity"
            bl = cluster_boot(self.ref, lambda s: np.percentile(s[col], 5), 3000)
            bh = cluster_boot(self.ref, lambda s: np.percentile(s[col], 95), 3000)
            rows.append(dict(feature=name, n_stones=len(self.ref),
                             n_patients=self.ref.patient.nunique(), icc=self.icc[j],
                             n_effective=self.n_eff[j], k_factor=k,
                             TI90_low=float(back(m - k * s)), TI90_high=float(back(m + k * s)),
                             boot_p5_median=np.median(bl), boot_p5_lower95=np.percentile(bl, 2.5),
                             boot_p95_median=np.median(bh), boot_p95_upper95=np.percentile(bh, 97.5)))
        return pd.DataFrame(rows)

    def score(self, d):
        """Score any set of stones (farm, controls) against the reference."""
        F = features(d)
        out = d[["stone_id", "volume_mm3", "porosity_pct", "sphericity"]].copy()
        for j, name in enumerate(["porosity", "sphericity"]):
            out[f"{name}_percentile_in_ref"] = [100 * np.mean(self.F[:, j] <= v) for v in F[:, j]]
            out[f"{name}_z"] = (F[:, j] - self.F[:, j].mean()) / self.F[:, j].std(ddof=1)
        D2 = self.d2(F)
        out["mahalanobis_d2"] = D2
        # % of REAL stones that are more typical than this stone (0 = centre, 100 = beyond all)
        out["atypicality_percentile"] = [100 * np.mean(self.null_d2 < v) for v in D2]
        out["inside_90_envelope"] = D2 <= self.envelope_cut
        return out


# --------------------------------------------------------------------------- #
# 4. Figure
# --------------------------------------------------------------------------- #
def make_figure(R, others, farm_scores, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter

    INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
    C_REF, C_UA, C_OTH, C_FARM = "#2f6fb0", "#d97b29", "#8e8e8e", "#c0392b"

    fig = plt.figure(figsize=(8.6, 7.4), dpi=200)
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    axx = fig.add_subplot(gs[0, 0], sharex=ax)
    axy = fig.add_subplot(gs[1, 1], sharey=ax)

    # envelope ellipse at the empirical LOPO cut
    t = np.linspace(0, 2 * np.pi, 400)
    vals, vecs = np.linalg.eigh(R.cov)
    circ = np.vstack([np.cos(t), np.sin(t)]) * np.sqrt(R.envelope_cut)
    ell = (vecs @ np.diag(np.sqrt(vals)) @ circ).T + R.mu
    ax.fill(ell[:, 0], ell[:, 1], color=C_REF, alpha=0.08, lw=0)
    ax.plot(ell[:, 0], ell[:, 1], color=C_REF, lw=1.2, ls="--",
            label=f"{int(ENVELOPE*100)}% reference envelope")

    ax.scatter(R.F[:, 0], R.F[:, 1], s=34, color=C_REF, edgecolor="white", lw=0.6,
               label=f"Calcium oxalate (reference), n={len(R.ref)}", zorder=3)
    for name, d, c, m in others:
        if len(d):
            F = features(d)
            ax.scatter(F[:, 0], F[:, 1], s=40, color=c, marker=m, edgecolor="white",
                       lw=0.6, label=f"{name}, n={len(d)}", zorder=3)
    if farm_scores is not None and len(farm_scores):
        F = features(farm_scores)
        ax.scatter(F[:, 0], F[:, 1], s=70, marker="*", color=C_FARM, edgecolor="white",
                   lw=0.6, label=f"Farm stones, n={len(F)}", zorder=4)

    # marginal KDEs
    for j, a in enumerate([axx, axy]):
        z = R.F[:, j]
        grid = np.linspace(z.min() - 1.2, z.max() + 1.2, 300)
        dens = stats.gaussian_kde(z)(grid)
        if j == 0:
            a.fill_between(grid, dens, color=C_REF, alpha=0.25, lw=0)
            a.plot(grid, dens, color=C_REF, lw=1)
        else:
            a.fill_betweenx(grid, dens, color=C_REF, alpha=0.25, lw=0)
            a.plot(dens, grid, color=C_REF, lw=1)
        a.axis("off")

    # axes labelled in original units
    pt = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    st = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ax.xaxis.set_major_locator(FixedLocator(logit(np.array(pt) / 100)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100*inv_logit(v):.2g}"))
    ax.yaxis.set_major_locator(FixedLocator(logit(st)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{inv_logit(v):.1f}"))
    ax.set_xlabel("Porosity (%, logit scale)", color=INK)
    ax.set_ylabel("Sphericity (logit scale)", color=INK)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle("Micro-CT architecture envelope of patient calcium oxalate stones",
                 x=0.07, ha="left", fontsize=12, color=INK)
    fig.text(0.07, 0.915, "Envelope = 90th percentile of leave-one-patient-out Mahalanobis "
             "distance; margins show reference density", ha="left", fontsize=8.5, color=MUTED)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. Main
# --------------------------------------------------------------------------- #
def main(master, farm=None):
    OUT.mkdir(exist_ok=True)
    df = load(master)
    df.to_csv(OUT / "01_all_stones_parsed.csv", index=False)
    ok = df[df.excluded_reason == ""].copy()

    ref = ok[ok.composition_class == "Calcium oxalate"]
    ua = ok[ok.composition_class == "Uric acid"]
    oth = ok[~ok.composition_class.isin(["Calcium oxalate", "Uric acid"])]

    R = Reference(ref)

    # ---- summaries
    counts = ok.groupby("composition_class").agg(stones=("stone_id", "size"),
                                                 patients=("patient", "nunique"))
    desc = pd.DataFrame([dict(group=g, feature=f, **describe(d, c))
                         for g, d in [("Calcium oxalate", ref), ("Uric acid", ua)]
                         for f, c in [("porosity_pct", "porosity_pct"),
                                      ("sphericity", "sphericity"),
                                      ("volume_mm3", "volume_mm3")]])
    ti = R.tolerance_intervals()
    sd = size_dependence(ref)
    shapiro = [dict(feature=n, W=stats.shapiro(R.F[:, j])[0], p=stats.shapiro(R.F[:, j])[1])
               for j, n in enumerate(["logit porosity", "logit sphericity"])]
    corr = stats.spearmanr(R.F[:, 0], R.F[:, 1])

    # ---- validation: real stones vs wrong-type stones
    ref_self = R.score(ref)
    ref_self["mahalanobis_d2"] = R.null_d2  # LOPO, not in-sample
    ref_self["inside_90_envelope"] = R.null_d2 <= R.envelope_cut
    ua_sc = R.score(ua)
    # sensitivity: Lorensen surface area
    ref_lor = ref.assign(sphericity=ref.sphericity_lorensen).dropna(subset=["sphericity"])
    R_lor = Reference(ref_lor)
    # sensitivity: pure COM only
    R_com = Reference(ref[ref.COM == 100])

    with pd.ExcelWriter(OUT / "reference_benchmark.xlsx") as xw:
        counts.to_excel(xw, sheet_name="counts")
        desc.to_excel(xw, sheet_name="descriptives", index=False)
        ti.to_excel(xw, sheet_name="tolerance_intervals", index=False)
        sd.to_excel(xw, sheet_name="size_dependence", index=False)
        ref_self.to_excel(xw, sheet_name="reference_LOPO_scores", index=False)
        ua_sc.to_excel(xw, sheet_name="uric_acid_control_scores", index=False)
        df[df.excluded_reason != ""][["stone_id", "reported_type", "excluded_reason"]] \
            .to_excel(xw, sheet_name="excluded", index=False)

    farm_sc = None
    if farm:
        f = pd.read_csv(farm)
        f["patient"] = f["stone_id"].astype(str)
        f["sphericity"] = (np.pi ** (1 / 3) * (6 * f.volume_mm3) ** (2 / 3)) / f.surface_area_mm2
        farm_sc = R.score(f)
        farm_sc.to_csv(OUT / "farm_scores.csv", index=False)

    make_figure(R, [("Uric acid (wrong-type control)", ua, "#d97b29", "s"),
                    ("Other", oth, "#8e8e8e", "^")], farm_sc,
                OUT / "envelope_figure.png")

    # ---- report
    L = []
    p = L.append
    p("REFERENCE BENCHMARK REPORT\n==========================")
    p(f"Stones in file: {len(df)}; with usable micro-CT: {len(ok)} "
      f"from {ok.patient.nunique()} patients")
    p("\nUsable stones by composition class:\n" + counts.to_string())
    p(f"\nReference = calcium oxalate: {len(ref)} stones, {ref.patient.nunique()} patients")
    p("\nDescriptives (original units):\n" + desc.round(3).to_string(index=False))
    p("\nNormality on logit scale (Shapiro-Wilk):\n" +
      pd.DataFrame(shapiro).round(3).to_string(index=False))
    p(f"\nPorosity vs sphericity (logit), Spearman rho = {corr[0]:.2f}, p = {corr[1]:.3f}")
    p("\nSize dependence (slope of logit feature on log10 volume, patient bootstrap 95% CI):\n"
      + sd.round(3).to_string(index=False))
    p("\n90%/95% tolerance intervals (normal on logit scale, ICC-adjusted n_eff) "
      "+ bootstrap percentile check:\n" + ti.round(4).T.to_string())
    p(f"\nBivariate envelope: 90th pct of LOPO D2 = {R.envelope_cut:.2f} "
      f"(chi2(2) 90% would be {stats.chi2.ppf(.9, 2):.2f})")
    p(f"Reference stones inside envelope (LOPO): {ref_self.inside_90_envelope.mean()*100:.0f}%")
    p(f"Uric-acid control stones inside CaOx envelope: {ua_sc.inside_90_envelope.sum()}/{len(ua_sc)} "
      f"({ua_sc.inside_90_envelope.mean()*100:.0f}%)  -> discriminative check")
    p(f"\nSensitivity - Lorensen SA: sphericity median {ref_lor.sphericity.median():.3f} vs "
      f"Lindblad {ref.sphericity.median():.3f}; envelope cut {R_lor.envelope_cut:.2f}")
    p(f"Sensitivity - pure COM only ({len(R_com.ref)} stones, {R_com.ref.patient.nunique()} pts): "
      f"envelope cut {R_com.envelope_cut:.2f}")
    chk = ok.porosity_check_diff.abs()
    p(f"\nData check: |reported porosity - 100*Vpore/Vtotal| max = {chk.max():.3f} "
      f"percentage points (stones > 0.05: {list(ok.loc[chk > 0.05, 'stone_id'])})")
    # discrimination: can each score separate reference (LOPO) from uric-acid stones?
    def auc(a, b):  # P(score_b > score_a)
        return stats.mannwhitneyu(b, a).statistic / (len(a) * len(b))
    p("\nDiscrimination, calcium oxalate (LOPO) vs uric acid, AUC (0.5 = no separation):")
    p(f"  bivariate D2 (UA more atypical): {auc(R.null_d2, ua_sc.mahalanobis_d2):.2f}")
    p(f"  sphericity (UA lower):           {auc(-ref.sphericity, -ua.sphericity):.2f}")
    p(f"  porosity (UA higher):            {auc(ref.porosity_pct, ua.porosity_pct):.2f}")
    p(f"  UA inside envelope at chi2(2) 90% cut ({stats.chi2.ppf(.9, 2):.2f}): "
      f"{(ua_sc.mahalanobis_d2 <= stats.chi2.ppf(.9, 2)).sum()}/{len(ua_sc)}; "
      f"reference inside at that cut (LOPO): {(R.null_d2 <= stats.chi2.ppf(.9, 2)).mean()*100:.0f}%")
    p(f"  Reference stones driving the envelope (LOPO D2 > 5): "
      f"{list(ref_self.loc[ref_self.mahalanobis_d2 > 5, 'stone_id'])}")
    ex = df[df.excluded_reason != ""]
    p(f"\nExcluded stones: {len(ex)} (see 'excluded' sheet). By reason:\n"
      + ex.excluded_reason.value_counts().to_string())
    if farm_sc is not None:
        p("\nFARM STONE SCORES:\n" + farm_sc.round(3).to_string(index=False))
    txt = "\n".join(L)
    (OUT / "report.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main(*sys.argv[1:3])
