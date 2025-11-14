# -*- coding: utf-8 -*-
"""
Numerical Experiment 3 (BPP vs PPP) — Final code with latest fixes
- F1: Boundary alignment (PPP theory vs BPP MC)  [ylabel θ; DS better at (0.67,0.10)]
- F2: Boundary deviation e(rho) with ADAPTIVE y-limits (main: Nc>=5; full: include Nc=1)
- F3a: Through-origin regression for 1 - Dtilde_min vs 1/N
- F3b: Collapse N*(1 - Dtilde_min) vs rho, with 3/8 band
- F4: Kappa slices (kappa_BPP/kappa - 1 vs 1/Nc at fixed rho)

Outputs in ./outputs_bpp_final/
"""

import os
import math
import numpy as np
import pandas as pd
import mpmath as mp
from functools import lru_cache
from scipy.spatial import cKDTree
import matplotlib as mpl
import matplotlib.pyplot as plt

# ------------- Aesthetics -------------
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = True
mpl.rcParams["figure.dpi"] = 140
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["legend.frameon"] = False
TAB10 = mpl.colormaps.get_cmap("tab10")

# ------------- Global config -------------
SEED = 20251024
rng = np.random.default_rng(SEED)

OUTDIR = "./outputs_bpp_final"
os.makedirs(OUTDIR, exist_ok=True)

# Spatial domain (torus)
L = 20.0
A = L * L

# Patients per instance & repeats
M = 10000
K = 200
K_MAP = {1: 500}
M_MAP = {1: 20000}

# Baseline grid (NC=10 exact integer rho points)
NC_BASE = 10
RHO_LIST_BASE = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0], dtype=float)
NP_LIST_BASE = (RHO_LIST_BASE * NC_BASE).astype(int)

# Sensitivity grid (include NC=1 for appendix/full plots)
NC_SENS = [1, 5, 15, 20]
NP_UNIVERSAL = np.array([1, 2, 5, 10, 15, 20, 50, 100, 150, 200], dtype=int)

# Theta grid (kept for potential later use)
THETA_MIN, THETA_MAX, THETA_N = 0.05, 0.60, 56
THETA_GRID = np.round(np.linspace(THETA_MIN, THETA_MAX, THETA_N), 4)

# Kappa theory integration precision
mp.mp.dps = 80
PI = mp.pi
THEORY_RHO_MIN, THEORY_RHO_MAX, THEORY_RHO_N = 1e-2, 30.0, 300
RHO_TAB = np.logspace(np.log10(THEORY_RHO_MIN), np.log10(THEORY_RHO_MAX), THEORY_RHO_N)
THETA_TAB = None  # filled in main()

# Kappa slice settings
KAPPA_SLICE_RHOS = [0.5, 1.0, 2.0, 10.0]
KAPPA_SLICE_TOL_REL = 0.12  # accept rows within ±12% of target rho

# Optional scale invariance (disabled by default)
RUN_SCALE_INVARIANCE = False
SCALES_L = [10.0, 20.0, 40.0]

# ------------- Torus helpers -------------
def build_periodic_tree(points, L):
    n = points.shape[0]
    if n == 0:
        raise RuntimeError("Empty point set.")
    offsets = np.array([[dx, dy] for dx in (-L, 0.0, L) for dy in (-L, 0.0, L)], dtype=float)
    tiled = (points[None, :, :] + offsets[:, None, :]).reshape(-1, 2)
    tree = cKDTree(tiled)
    return tree, n

def query_periodic_tree(tree, base_size, query_pts):
    dist, idx = tree.query(query_pts, k=1)
    base_idx = idx % base_size
    return dist, base_idx

# ------------- BPP samplers -------------
def sample_bpp_points(N, L, rng):
    if N <= 0:
        return np.zeros((0, 2), dtype=float)
    return rng.uniform(0.0, L, size=(N, 2))

def simulate_one_instance_bpp(NC, NP, L, M, rng):
    """Return a_k, b_k, sanity metrics."""
    assert NC > 0 and NP > 0
    CSC = sample_bpp_points(NC, L, rng)
    PSC = sample_bpp_points(NP, L, rng)

    tree_csc, nC = build_periodic_tree(CSC, L)
    tree_psc, nP = build_periodic_tree(PSC, L)

    # PSC → nearest CSC
    dist_psc_to_csc, _ = query_periodic_tree(tree_csc, nC, PSC)

    # Random patients
    PTS = rng.uniform(0.0, L, size=(M, 2))
    D_C, _ = query_periodic_tree(tree_csc, nC, PTS)
    D_P, psc_idx_base = query_periodic_tree(tree_psc, nP, PTS)

    D = np.minimum(D_C, D_P)
    nearest_is_psc = (D_P < D_C)

    Y = np.zeros(M, dtype=float)
    if np.any(nearest_is_psc):
        Y[nearest_is_psc] = dist_psc_to_csc[psc_idx_base[nearest_is_psc]]

    a_k = float(np.mean(D_C - D))                          # E[D_C - D_min]
    b_k = float(np.mean(nearest_is_psc.astype(float) * Y)) # E[ 1{near PSC} * Y ]

    sanity = dict(
        N_C=NC, N_P=NP, rho_bpp=NP / NC,
        p_near_psc_hat=float(np.mean(nearest_is_psc.astype(float))),
        D_mean=float(np.mean(D)),            # E[D_min]
        D_C_mean=float(np.mean(D_C)),        # E[D_C]
    )
    return a_k, b_k, sanity

# ------------- Theory: kappa(rho), alpha, b, theta* -------------
def _clamp_unit(x): return min(1.0, max(-1.0, x))

def h_dimless(x):
    x = mp.mpf(x)
    if x < 2:
        a = _clamp_unit(float(x/2))
        b = _clamp_unit(float(1 - x**2/2))
        term = (x**2 * mp.acos(a) + mp.acos(b) - (x/2) * mp.sqrt(max(0, 4 - x**2)))
        return x**2 - (1/PI) * term
    else:
        return x**2 - 1

@lru_cache(maxsize=512)
def kappa_rho_cached(rho_val):
    rho = mp.mpf(rho_val)
    g = lambda t: ((1+rho) + h_dimless(t))**(-mp.mpf('1.5'))
    I0_2 = mp.quad(g, [0, 0.9, 1.8, 1.95, 1.99, 2.0])
    tail = (1/rho) * (1 - 2/mp.sqrt(rho + 4))
    return float((1+rho) * (I0_2 + tail))

def kappa_rho(rho): return kappa_rho_cached(float(rho))

def alpha_theory(rho):
    rho = float(rho)
    return 1.0 - 1.0 / math.sqrt(1.0 + rho)

def b_theory(rho):
    rho = float(rho)
    return (rho / (1.0 + rho)) * kappa_rho(rho)

def theta_star_theory(rho):
    return alpha_theory(rho) / b_theory(rho)

# ------------- Aggregation & SE (delta method) -------------
def aggregate_theta_star(a_list, b_list):
    a_arr = np.array(a_list, dtype=float)
    b_arr = np.array(b_list, dtype=float)
    a_bar = float(a_arr.mean()); b_bar = float(b_arr.mean())
    theta_mc = (a_bar / b_bar) if b_bar > 0 else np.nan

    if len(a_arr) > 1 and b_bar > 0:
        s_a2 = float(a_arr.var(ddof=1))
        s_b2 = float(b_arr.var(ddof=1))
        s_ab = float(np.cov(a_arr, b_arr, ddof=1)[0, 1])
        var_theta = (s_a2 / (len(a_arr) * b_bar**2)
                     + (a_bar**2) / (len(a_arr) * b_bar**4) * s_b2
                     - (2 * a_bar) / (len(a_arr) * b_bar**3) * s_ab)
        var_theta = max(0.0, var_theta)
        se_theta = math.sqrt(var_theta)
        ci_lo = theta_mc - 1.96 * se_theta
        ci_hi = theta_mc + 1.96 * se_theta
    else:
        se_theta = np.nan; ci_lo = np.nan; ci_hi = np.nan

    return a_bar, b_bar, theta_mc, se_theta, ci_lo, ci_hi

# ------------- Run BPP grid -------------
def run_bpp_grid(configs, L, M, K, rng):
    rows, sanity_rows = [], []
    unique_rhos = sorted({c['NP'] / c['NC'] for c in configs})
    for r in unique_rhos: _ = kappa_rho(r)  # warm cache

    for cfg in configs:
        NC, NP = int(cfg['NC']), int(cfg['NP'])
        rho = NP / NC
        K_cur = K_MAP.get(NC, K)
        M_cur = M_MAP.get(NC, M)

        a_list, b_list, S = [], [], []
        for _ in range(K_cur):
            a_k, b_k, sanity = simulate_one_instance_bpp(NC, NP, L, M_cur, rng)
            a_list.append(a_k); b_list.append(b_k); S.append(sanity)

        a_bar, b_bar, theta_mc, se_theta, ci_lo, ci_hi = aggregate_theta_star(a_list, b_list)
        theta_th = theta_star_theory(rho)

        s_df = pd.DataFrame(S)
        D_C_bar = float(s_df["D_C_mean"].mean())
        D_min_bar = float(s_df["D_mean"].mean())
        lambda_C = NC / A
        lambda_T = (NC + NP) / A

        # Dimensionless tilde (PPP baseline = 1)
        tilde_Dc = 2.0 * math.sqrt(lambda_C) * D_C_bar
        tilde_Dmin = 2.0 * math.sqrt(lambda_T) * D_min_bar

        # PPP baselines (explicit values)
        Dc_ppp = 1.0 / (2.0 * math.sqrt(lambda_C))
        Dmin_ppp = 1.0 / (2.0 * math.sqrt(lambda_T))

        # Ratios for explicit PPP vs BPP comparison
        ratio_Dc = D_C_bar / Dc_ppp
        ratio_Dmin = D_min_bar / Dmin_ppp

        # kappa_hat (BPP) and relative deviation
        kappa_hat = ((1.0 + rho) / rho) * (2.0 * math.sqrt(lambda_C)) * b_bar
        kappa_th = kappa_rho(rho)
        kappa_rel = (kappa_hat / kappa_th) - 1.0

        rows.append(dict(
            NC=NC, NP=NP, rho=rho,
            a_bar=a_bar, b_bar=b_bar,
            theta_star_mc=theta_mc, theta_star_mc_se=se_theta,
            theta_star_mc_lo=ci_lo, theta_star_mc_hi=ci_hi,
            theta_star_th=theta_th,
            tilde_Dc=tilde_Dc, tilde_Dmin=tilde_Dmin,
            Dc_ppp=Dc_ppp, Dmin_ppp=Dmin_ppp,
            Dc_bpp=D_C_bar, Dmin_bpp=D_min_bar,
            ratio_Dc=ratio_Dc, ratio_Dmin=ratio_Dmin,
            kappa_hat=kappa_hat, kappa_th=kappa_th,
            kappa_rel=kappa_rel
        ))

        sanity_rows.append(dict(
            NC=NC, NP=NP, rho=rho,
            N_C_mean=float(s_df["N_C"].mean()),
            N_P_mean=float(s_df["N_P"].mean()) if "N_P" in s_df else NP,
            D_mean=D_min_bar, D_C_mean=D_C_bar,
            p_near_psc_hat=float(s_df["p_near_psc_hat"].mean()),
            p_near_psc_theory=rho / (1.0 + rho),
            p_near_psc_err=float(s_df["p_near_psc_hat"].mean() - (rho / (1.0 + rho))),
        ))

        print(f"[N_c={NC:>2}, N_p={NP:>3}, rho={rho:>6.3f}] "
              f"ā={a_bar:.6f}, b̄={b_bar:.6f}, "
              f"theta*MC={theta_mc:.4f} [{ci_lo:.4f},{ci_hi:.4f}] | theta*th={theta_th:.4f}")

    df_main = pd.DataFrame(rows).sort_values(["NC", "rho"]).reset_index(drop=True)

    # Eval with errors (boundary)
    df_eval = df_main.copy()
    df_eval["e_rho"] = df_eval["theta_star_mc"] - df_eval["theta_star_th"]
    df_eval["abs_e"] = df_eval["e_rho"].abs()
    df_eval["z_rho"] = df_eval["e_rho"] / df_eval["theta_star_mc_se"]
    df_eval.loc[~np.isfinite(df_eval["z_rho"]), "z_rho"] = np.nan
    df_eval["in_95_CI"] = df_eval["theta_star_th"].between(df_eval["theta_star_mc_lo"],
                                                           df_eval["theta_star_mc_hi"])
    df_sanity = pd.DataFrame(sanity_rows).sort_values(["NC", "rho"]).reset_index(drop=True)
    return df_main, df_eval, df_sanity

# ------------- Theory table (rho -> theta*_th) -------------
def build_theta_table():
    return np.array([theta_star_theory(float(r)) for r in RHO_TAB], dtype=float)

# ------------- Plot helpers -------------
def moves_legend(ax, loc="best"):
    leg = ax.legend(loc=loc)
    if leg is not None:
        leg.set_frame_on(False)

def pick_rep_points_for_errors(sub_df, targets=[0.2, 1.0, 5.0]):
    idxs = []
    rho_vals = sub_df["rho"].values
    for t in targets:
        j = int(np.argmin(np.abs(rho_vals - t)))
        if j not in idxs:
            idxs.append(j)
    return idxs

# ------------- F1: Boundary alignment -------------
def plot_boundary_alignment(df_main, outdir, theta_tab):
    fig = plt.figure(figsize=(7.6, 5.0))
    ax = fig.add_subplot(111)

    # Theory curve
    ax.plot(RHO_TAB, theta_tab, lw=2.6, color="black", label=r"PPP $\theta_{\mathrm{th}}^*(\rho)$")

    # BPP curves grouped by N_c
    for i, NC in enumerate(sorted(df_main["NC"].unique())):
        sub = df_main[df_main["NC"] == NC].sort_values("rho")
        ax.plot(sub["rho"], sub["theta_star_mc"], marker="o", linestyle="-",
                lw=1.9, ms=5.0, color=TAB10(i), label=fr"BPP $\theta_{{\mathrm{{sim}}}}^*(\rho)$, $N_c={NC}$")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\rho = N_p/N_c$ (log scale)")
    ax.set_ylabel(r"$\theta$")
    ax.set_title("Boundary alignment: PPP theory vs BPP (finite-size)")

    # Region labels (updated DS position)
    ax.text(0.30, 0.84, "MS better\n($\\theta>\\theta^*(\\rho)$)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7))
    ax.text(0.58, 0.10, "DS better\n($\\theta<\\theta^*(\\rho)$)",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7))

    moves_legend(ax, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "boundary_alignment_bpp.png"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "boundary_alignment_bpp.pdf"), bbox_inches="tight")
    plt.close(fig)

# ------------- F2: Error vs rho (main & full) -------------
def plot_error_vs_rho(df_eval, outdir):
    # MAIN: Nc >= 5, with adaptive y-limits and ±2SE at representative points
    fig = plt.figure(figsize=(7.6, 5.0))
    ax = fig.add_subplot(111)

    # first pass: compute adaptive y-limits across plotted curves & chosen error bars
    y_lo, y_hi = np.inf, -np.inf
    drawn = []
    for i, NC in enumerate(sorted(df_eval["NC"].unique())):
        if NC < 5:
            continue
        sub = df_eval[df_eval["NC"] == NC].sort_values("rho")
        if sub.empty:
            continue
        ys = sub["e_rho"].values
        ses = sub["theta_star_mc_se"].values
        y_lo = min(y_lo, float(np.min(ys)))
        y_hi = max(y_hi, float(np.max(ys)))
        idxs = pick_rep_points_for_errors(sub)
        if len(idxs) > 0:
            y_lo = min(y_lo, float(np.min(ys[idxs] - 2*ses[idxs])))
            y_hi = max(y_hi, float(np.max(ys[idxs] + 2*ses[idxs])))
        drawn.append((i, NC, sub, idxs))

    # plotting
    for (i, NC, sub, idxs) in drawn:
        ax.plot(sub["rho"], sub["e_rho"], marker="o", linestyle="-", lw=1.9, ms=5,
                color=TAB10(i), label=fr"$N_c={NC}$")
        for j in idxs:
            x = float(sub["rho"].iloc[j])
            y = float(sub["e_rho"].iloc[j])
            se = float(sub["theta_star_mc_se"].iloc[j])
            ax.errorbar([x], [y], yerr=[[2*se],[2*se]], fmt='none',
                        ecolor=TAB10(i), elinewidth=1.0, capsize=2)

    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_xscale("log")
    # adaptive ylim with 8% padding
    if np.isfinite(y_lo) and np.isfinite(y_hi) and y_hi > y_lo:
        pad = 0.08 * (y_hi - y_lo)
        ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.set_xlabel(r"$\rho = N_p/N_c$ (log scale)")
    ax.set_ylabel(r"$e(\rho)=\theta^*_{\rm BPP}-\theta^*_{\rm th}$")
    ax.set_title("Finite-size boundary deviation by $\\rho$ (BPP − theory)")
    ax.text(0.02, 0.93, "negative ⇒ BPP boundary lower than PPP",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7))
    moves_legend(ax, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "error_vs_rho_bpp_main.png"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "error_vs_rho_bpp_main.pdf"), bbox_inches="tight")
    plt.close(fig)

    # FULL: include Nc = 1 for appendix
    fig = plt.figure(figsize=(7.6, 5.0))
    ax = fig.add_subplot(111)
    for i, NC in enumerate(sorted(df_eval["NC"].unique())):
        sub = df_eval[df_eval["NC"] == NC].sort_values("rho")
        if sub.empty:
            continue
        ax.plot(sub["rho"], sub["e_rho"], marker="o", linestyle="-", lw=1.7, ms=4.5,
                color=TAB10(i), label=fr"$N_c={NC}$")
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\rho = N_p/N_c$ (log scale)")
    ax.set_ylabel(r"$e(\rho)$")
    ax.set_title("Finite-size boundary deviation")
    moves_legend(ax, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "error_vs_rho_bpp.png"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "error_vs_rho_bpp.pdf"), bbox_inches="tight")
    plt.close(fig)

# ------------- F3a: through-origin regression for 1 - Dtilde_min -------------
def fit_origin_regression(x, y):
    """Through-origin OLS: y = beta * x."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    n = len(x)
    Sxx = float(np.dot(x, x))
    if Sxx <= 1e-15:
        return dict(slope=np.nan, lo=np.nan, hi=np.nan, R2=np.nan, se=np.nan, n=n)
    beta = float(np.dot(x, y) / Sxx)
    resid = y - beta * x
    dof = max(n - 1, 1)
    s2 = float(np.dot(resid, resid) / dof)
    se_beta = math.sqrt(s2 / max(Sxx, 1e-15))
    z = 1.96
    denom = float(np.dot(y, y))
    R2 = 1.0 - (float(np.dot(resid, resid)) / max(denom, 1e-15))
    return dict(slope=beta, lo=beta - z*se_beta, hi=beta + z*se_beta, R2=R2, se=se_beta, n=n)

def plot_tildeDmin_origin_and_collapse(df_main, outdir):
    # Use Nc >= 5 for asymptotic fit
    df = df_main[df_main["NC"] >= 5].copy()
    df["N_tot"] = df["NC"] + df["NP"]
    x = 1.0 / df["N_tot"].values
    y = 1.0 - df["tilde_Dmin"].values
    res = fit_origin_regression(x, y)

    # Through-origin regression
    fig = plt.figure(figsize=(6.8, 4.8))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=18, color="#2ca02c", alpha=0.85, label="BPP per-config")
    xs = np.linspace(0, x.max()*1.05, 300)
    ax.plot(xs, res["slope"]*xs, color="#2ca02c", lw=2.2,
            label=fr"Through-origin fit: slope={res['slope']:.3f} "
                  fr"[{res['lo']:.3f},{res['hi']:.3f}], $R^2$={res['R2']:.3f}")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"$1/N$  where  $N=N_c+N_p$")
    ax.set_ylabel(r"$1-\tilde{D}_{\min}$")
    ax.set_title("Verification (through-origin): $1-\\tilde{D}_{\\min}$ vs $1/N$")
    moves_legend(ax, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "verify_B_tildeDmin_origin.png"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "verify_B_tildeDmin_origin.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Collapse: N*(1 - Dtilde_min) ~ 3/8
    fig = plt.figure(figsize=(6.8, 4.8))
    ax = fig.add_subplot(111)
    y_col = df["N_tot"].values * (1.0 - df["tilde_Dmin"].values)
    # soft reference band around 3/8
    ax.axhspan(0.375-0.03, 0.375+0.03, color="#ff7f0e", alpha=0.10, lw=0)
    ax.scatter(df["rho"].values, y_col, s=20, color="#1f77b4", alpha=0.85)
    ax.axhline(0.375, color="#ff7f0e", lw=1.6, ls="--", label="Reference 3/8 = 0.375")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\rho = N_p/N_c$ (log scale)")
    ax.set_ylabel(r"$N\cdot(1-\tilde{D}_{\min})$")
    ax.set_title("Collapse of first-order constant: $N\\cdot(1-\\tilde{D}_{\\min})$")
    moves_legend(ax, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "collapse_B_tildeDmin.png"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "collapse_B_tildeDmin.pdf"), bbox_inches="tight")
    plt.close(fig)

# ------------- F4: Kappa slices (fixed rho) -------------
def plot_kappa_slices(df_main, outdir, slice_rhos=KAPPA_SLICE_RHOS, tol_rel=KAPPA_SLICE_TOL_REL):
    fig = plt.figure(figsize=(7.2, 5.0))
    ax = fig.add_subplot(111)
    used = 0
    for j, rt in enumerate(slice_rhos):
        rows = []
        for NC in sorted(df_main["NC"].unique()):
            sub_nc = df_main[df_main["NC"] == NC]
            if len(sub_nc) == 0:
                continue
            diffs = np.abs(sub_nc["rho"].values - rt)
            k = int(np.argmin(diffs))
            rho_star = float(sub_nc["rho"].iloc[k])
            if abs(rho_star - rt) <= tol_rel * rt:
                invNC = 1.0 / float(NC)
                y = float(sub_nc["kappa_rel"].iloc[k])  # relative deviation
                rows.append((invNC, y, rho_star, NC))
        if len(rows) >= 2:
            rows = sorted(rows, key=lambda t: t[0])
            x = [r[0] for r in rows]
            y = [r[1] for r in rows]
            lbl = fr"$\rho\approx{rt:g}$"
            ax.plot(x, y, marker="o", lw=1.9, ms=5, color=TAB10(j), label=lbl)
            used += 1

    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_xlabel(r"$1/N_c$")
    ax.set_ylabel(r"$\kappa_{\rm BPP}/\kappa - 1$")
    ax.set_title(r"Kappa finite-size slices (fixed $\rho$): convergence as $1/N_c\to 0$")
    moves_legend(ax, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "kappa_ratio_vs_invNC_slices.png"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "kappa_ratio_vs_invNC_slices.pdf"), bbox_inches="tight")
    plt.close(fig)
    if used == 0:
        print("[Warn] No usable kappa slices found (rho grid may not match tolerance).")

# ------------- Optional: Scale invariance (appendix) -------------
def run_scale_invariance(df_main_base, outdir):
    pass  # optional; kept minimal per current scope

# ------------- Main -------------
def main():
    global THETA_TAB
    THETA_TAB = build_theta_table()

    # Build config list
    configs = []
    # NC=10 base (exact integer rho)
    for NP in NP_LIST_BASE:
        configs.append({"NC": NC_BASE, "NP": int(NP)})
    # Sensitivity NC in {1,5,15,20} with universal NP grid
    for NC in NC_SENS:
        for NP in NP_UNIVERSAL:
            if not (NC == NC_BASE and NP in NP_LIST_BASE):
                configs.append({"NC": int(NC), "NP": int(NP)})

    # Run grid
    df_main, df_eval, df_sanity = run_bpp_grid(configs, L=L, M=M, K=K, rng=rng)

    # Save tables
    df_main.to_csv(os.path.join(OUTDIR, "bpp_boundary_main.csv"), index=False)
    df_eval.to_csv(os.path.join(OUTDIR, "bpp_boundary_eval.csv"), index=False)
    cov_by_nc = df_eval.groupby("NC")["in_95_CI"].mean().rename("coverage").reset_index()
    cov_by_nc.to_csv(os.path.join(OUTDIR, "coverage_by_NC.csv"), index=False)
    print("\n[Coverage by N_c]")
    print(cov_by_nc)

    # F1: boundary alignment
    plot_boundary_alignment(df_main, OUTDIR, THETA_TAB)

    # F2: error curves (adaptive y-limits)
    plot_error_vs_rho(df_eval, OUTDIR)

    # F3: 3/8 verification (through-origin + collapse with band)
    plot_tildeDmin_origin_and_collapse(df_main, OUTDIR)

    # F4: kappa slices at fixed rho
    plot_kappa_slices(df_main, OUTDIR, slice_rhos=KAPPA_SLICE_RHOS, tol_rel=KAPPA_SLICE_TOL_REL)

    if RUN_SCALE_INVARIANCE:
        run_scale_invariance(df_main, OUTDIR)

    print(f"\nDone. Results saved to: {OUTDIR}")

if __name__ == "__main__":
    main()
